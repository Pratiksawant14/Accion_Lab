"""Safe Unpickler that stubs out missing torch/ultralytics classes.
Allows reading metadata dictionaries (names, train_args, epoch, etc.) without having ultralytics installed.
"""

from __future__ import annotations

import io
import json
import pickle
from pprint import pprint
import zipfile


class MockObject:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.__dict__.update(
            state if isinstance(state, dict) else {"state": state}
        )

    def __repr__(self):
        return f"<Mock {self.__class__.__name__}>"


class SafeMetadataUnpickler(pickle.Unpickler):

    def persistent_load(self, pid):
        return None

    def find_class(self, module, name):
        # Stub out ultralytics and torch nn classes so deserialization doesn't fail
        if module.startswith("ultralytics") or module.startswith("torch.nn"):
            return type(name, (MockObject,), {})
        if module == "torch._utils" and name in (
            "_rebuild_tensor_v2",
            "_rebuild_parameter",
        ):
            return lambda *args: None
        if module.startswith("torch") and ("Storage" in name or "Size" in name):
            return type(name, (MockObject,), {})
        return super().find_class(module, name)


def inspect_checkpoint_dict(pt_path: str):
    with zipfile.ZipFile(pt_path) as z:
        pkl_name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        pkl_bytes = z.read(pkl_name)

    unpickler = SafeMetadataUnpickler(io.BytesIO(pkl_bytes))
    data = unpickler.load()

    model_obj = getattr(data, "get", lambda k, d=None: None)("model")
    names = getattr(data, "get", lambda k, d=None: None)("names")
    if not names and model_obj:
        names = getattr(model_obj, "names", None)
        if not names and hasattr(model_obj, "yaml") and isinstance(model_obj.yaml, dict):
            names = model_obj.yaml.get("names")

    summary = {
        "file": pt_path,
        "epoch": getattr(data, "get", lambda k, d=None: None)("epoch"),
        "names": names,
        "date": getattr(data, "get", lambda k, d=None: None)("date"),
        "version": getattr(data, "get", lambda k, d=None: None)("version"),
    }

    train_args = getattr(data, "get", lambda k, d=None: None)("train_args")
    if train_args and isinstance(train_args, dict):
        summary["train_args"] = {
            k: train_args.get(k)
            for k in [
                "task",
                "mode",
                "model",
                "data",
                "epochs",
                "batch",
                "imgsz",
                "patience",
                "optimizer",
                "lr0",
                "lrf",
                "device",
            ]
            if k in train_args
        }

    return summary


if __name__ == "__main__":
    for pt in [
        "checkpoints/Agriculture_bestweight.pt",
        "checkpoints/best.pt",
        "checkpoints/space_electronics.pt",
    ]:
        print(f"\n========================================================")
        print(f"METADATA EXTRACTION: {pt}")
        print(f"========================================================")
        try:
            res = inspect_checkpoint_dict(pt)
            pprint(res)
        except Exception as e:
            print(f"Error inspecting {pt}: {e}")
