"""Unit and integration tests for edge model training pipeline & evaluation."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.training.config import TrainingConfig
from src.training.runner import TrainingRunner, TrainingResult
from src.training.evaluator import ModelEvaluator, EvaluationResult


class TestTrainingConfig:
    """Test suite for TrainingConfig schema and validation."""

    def test_valid_config_creation(self, tmp_path):
        cfg = TrainingConfig(
            domain="agriculture",
            model_variant="yolov8n.pt",
            data_yaml="configs/datasets/agriculture_data.yaml",
            imgsz=512,
            epochs=20,
            batch=8,
        )
        assert cfg.domain == "agriculture"
        assert cfg.imgsz == 512
        assert cfg.epochs == 20

        # Test YAML serialization roundtrip
        yaml_path = tmp_path / "test_config.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded_cfg = TrainingConfig.from_yaml(yaml_path)
        assert loaded_cfg.domain == cfg.domain
        assert loaded_cfg.imgsz == cfg.imgsz
        assert loaded_cfg.epochs == cfg.epochs

    def test_invalid_domain_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                domain="invalid_domain_xyz",
                data_yaml="data.yaml",
            )

    def test_negative_parameters_raise(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                domain="ships",
                data_yaml="data.yaml",
                epochs=-5,
            )

    def test_ultralytics_args_generation(self):
        cfg = TrainingConfig(
            domain="space_electronics",
            model_variant="yolo11n.pt",
            data_yaml="configs/datasets/space_electronics_data.yaml",
            imgsz=640,
            epochs=30,
            batch=16,
            device="cpu",
        )
        args = cfg.to_ultralytics_args()
        assert args["data"] == "configs/datasets/space_electronics_data.yaml"
        assert args["imgsz"] == 640
        assert args["epochs"] == 30
        assert args["batch"] == 16
        assert args["device"] == "cpu"


class TestTrainingRunner:
    """Test suite for TrainingRunner execution and pre-flight validation."""

    def test_dry_run_training(self, tmp_path):
        dummy_data = tmp_path / "data.yaml"
        dummy_data.write_text("names: [a, b]", encoding="utf-8")

        cfg = TrainingConfig(
            domain="ships",
            model_variant="yolo11n.pt",
            data_yaml=str(dummy_data),
            validate_dataset_before_train=False,
            project=str(tmp_path / "runs"),
            name="exp_test",
        )

        runner = TrainingRunner(cfg)
        result = runner.train(dry_run=True)

        assert isinstance(result, TrainingResult)
        assert result.domain == "ships"
        assert result.validation_passed is True
        assert result.best_checkpoint_path is not None
        assert Path(result.best_checkpoint_path).exists()
        assert Path(result.last_checkpoint_path).exists()
        assert result.duration_seconds >= 0.0

    def test_pre_flight_corrupt_dataset_aborts(self, tmp_path):
        # Create a mock corrupt dataset (empty image + out of bounds label)
        ds_dir = tmp_path / "corrupt_ds"
        (ds_dir / "images" / "train").mkdir(parents=True)
        (ds_dir / "labels" / "train").mkdir(parents=True)

        corrupt_img = ds_dir / "images" / "train" / "bad.jpg"
        corrupt_img.write_bytes(b"not an image binary header")

        corrupt_lbl = ds_dir / "labels" / "train" / "bad.txt"
        corrupt_lbl.write_text("0 1.5 0.5 0.2 0.2", encoding="utf-8")

        cfg = TrainingConfig(
            domain="agriculture",
            data_yaml=str(tmp_path / "data.yaml"),
            dataset_root=str(ds_dir),
            validate_dataset_before_train=True,
        )

        runner = TrainingRunner(cfg)
        with pytest.raises(RuntimeError, match="Pre-flight validation failed"):
            runner.run_pre_flight_check()


class TestModelEvaluator:
    """Test suite for ModelEvaluator metrics calculation and reporting."""

    def test_nonexistent_checkpoint_raises(self, tmp_path):
        missing_ckpt = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            ModelEvaluator(missing_ckpt, domain="agriculture")

    def test_mock_evaluation_and_report_saving(self, tmp_path):
        dummy_ckpt = tmp_path / "dummy_best.pt"
        dummy_ckpt.write_text("MOCK_WEIGHTS", encoding="utf-8")
        dummy_data = tmp_path / "data.yaml"
        dummy_data.write_text("dummy: true", encoding="utf-8")

        evaluator = ModelEvaluator(dummy_ckpt, domain="ships")
        result = evaluator.evaluate(
            data_yaml=dummy_data,
            split="val",
            mock_result={
                "map50": 0.912,
                "map50_95": 0.745,
                "precision": 0.895,
                "recall": 0.864,
                "fitness": 0.761,
                "speed_ms": {"preprocess": 1.1, "inference": 142.3, "postprocess": 1.9},
                "per_class_map50": {"cargo_ship": 0.94, "passenger_ship": 0.88},
            },
        )

        assert isinstance(result, EvaluationResult)
        assert result.domain == "ships"
        assert result.map50 == 0.912
        assert result.map50_95 == 0.745
        assert result.per_class_map50["cargo_ship"] == 0.94

        # Verify saving report
        report_path = tmp_path / "eval_report.json"
        result.save_report(report_path)
        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["map50"] == 0.912
        assert saved_data["domain"] == "ships"
