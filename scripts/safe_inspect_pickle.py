"""Safe Pickle Introspector for PyTorch Checkpoints without executing code.
Uses Python standard library `pickletools` and `zipfile`.
"""

import io
import pickletools
import zipfile
from pathlib import Path


def extract_pickle_strings_and_globals(pt_path: Path):
    with zipfile.ZipFile(pt_path) as z:
        # Find data.pkl
        pkl_names = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            print(f"No data.pkl in {pt_path}")
            return
        pkl_bytes = z.read(pkl_names[0])

    print(f"\n========================================================")
    print(f"SAFE INTROSPECTION: {pt_path.name} ({len(pkl_bytes):,} bytes)")
    print(f"========================================================")

    globals_found = set()
    strings_found = []

    for opcode, arg, pos in pickletools.genops(io.BytesIO(pkl_bytes)):
        if opcode.name == "GLOBAL":
            globals_found.add(arg)
        elif opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            if isinstance(arg, str) and len(arg) > 1:
                strings_found.append(arg)

    print("--- REFERENCED PYTHON MODULES & CLASSES (GLOBAL OPCODES) ---")
    for g in sorted(globals_found):
        if any(
            x in g
            for x in [
                "ultralytics",
                "torch",
                "models",
                "yolo",
                "tasks",
                "Detection",
            ]
        ):
            print(f"  {g}")

    print("\n--- SIGNIFICANT STRINGS / METADATA DETECTED ---")
    # Filter for interesting strings
    interesting = [
        s
        for s in strings_found
        if any(
            k in s.lower()
            for k in [
                "yolo",
                "crop",
                "ship",
                "open",
                "short",
                "names",
                "imgsz",
                "epoch",
                "version",
                "nc",
                "cls",
                "class",
            ]
        )
    ]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in interesting:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    for s in deduped[:50]:
        print(f"  {s}")


if __name__ == "__main__":
    for pt in [
        "checkpoints/Agriculture_bestweight.pt",
        "checkpoints/best.pt",
        "checkpoints/space_electronics.pt",
    ]:
        p = Path(pt)
        if p.exists():
            extract_pickle_strings_and_globals(p)
        else:
            print(f"File not found: {pt}")
