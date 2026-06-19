from copy import deepcopy
from pathlib import Path
import yaml


DEFAULT_CONFIG = {
    "seed": 42,
    "dataset": {
        "root_dir": "./data/toolwear",
        "train_index": "./data/toolwear/train.csv",
        "valid_index": "./data/toolwear/valid.csv",
        "test_index": "./data/toolwear/test.csv",
        "sampling_rate": 10240,
        "window_length": 1024,
        "normalization": {
            "enabled": True,
            "id": "global-zscore-v1",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    },
    "model": {
        "in_channels": 3,
        "base_channels": 64,
        "attn_heads": 4,
        "dropout": 0.1,
        "latent_channels": 48,
        "latent_tokens": 8,
        "use_reference_decoder": True,
        "decoder_hidden_dim": 1024,
    },
    "train": {
        "batch_size": 64,
        "num_workers": 4,
        "epochs": 50,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "use_quantization": True,
        "quant_bits": 8,
        "save_dir": "./checkpoints/industrial_semantic",
        "loss_weights": {
            "contact": 1.0,
            "wear": 1.0,
            "mapped": 1.0,
            "reconstruction": 0.3,
        },
    },
    "policy": {
        "idle_conf_threshold": 0.90,
        "idle_occupancy_threshold": 0.20,
        "cutting_occupancy_threshold": 0.80,
        "acceleration_alarm_threshold": 0.75,
        "uncertainty_threshold": 0.55,
        "ood_threshold": 0.60,
        "normal_bits": 8,
        "mixed_bits": 10,
        "alarm_bits": 12,
        "occupancy_splits": 4,
        "occupancy_energy_ratio": 0.35,
        "allow_anchor_only_for_steady": False,
        "anchor_only_conf_threshold": 0.98,
        "send_raw_on_alarm": True,
        "ema_alpha": 0.60,
        "promotion_threshold": 0.55,
    },
    "receiver": {
        "module": "",
        "class_name": "",
        "checkpoint": "",
        "init_kwargs": {},
        "method_candidates": [
            "decode_packet",
            "decode",
            "reconstruct",
            "forward",
            "inference",
        ],
        "use_reference_decoder_fallback": True,
        "reference_checkpoint": "",
    },
    "runtime": {
        "session_id": "SESSION-001",
        "machine_id": "M001",
        "spindle_id": "S001",
        "tool_id": "T001",
        "encoder_version": "semtx-1.0",
    },
}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path=None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    return _deep_update(cfg, user_cfg)