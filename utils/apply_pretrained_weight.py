import torch
from pathlib import Path

DEFAULT_SKIP_PREFIXES = ("feature_boost.", "depth_head.", "normal_head.")


def _load_pretrained_weights(weight_filepath):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(PROJECT_ROOT / weight_filepath, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def _strip_module_prefix(key):
    if key.startswith("module."):
        return key[len("module."):]
    return key


def apply_weight(net, weight_filepath, skip_prefixes=DEFAULT_SKIP_PREFIXES, verbose=True):
    checkpoint = _load_pretrained_weights(weight_filepath)
    model_state = net.state_dict()

    filtered_checkpoint = {}
    skipped_by_prefix = []
    skipped_missing = []
    skipped_shape = []

    for key, value in checkpoint.items():
        key = _strip_module_prefix(key)

        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
            continue

        if key not in model_state:
            skipped_missing.append(key)
            continue

        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue

        filtered_checkpoint[key] = value

    load_info = net.load_state_dict(filtered_checkpoint, strict=False)

    report = {
        "loaded": sorted(filtered_checkpoint.keys()),
        "skipped_by_prefix": sorted(skipped_by_prefix),
        "skipped_missing": sorted(skipped_missing),
        "skipped_shape": sorted(skipped_shape),
        "missing_keys": sorted(load_info.missing_keys),
        "unexpected_keys": sorted(load_info.unexpected_keys),
    }

    if verbose:
        print(f"Loaded pretrained weights from {weight_filepath}")
        print(f"  loaded tensors: {len(report['loaded'])}")
        print(f"  skipped by prefix: {len(report['skipped_by_prefix'])}")
        print(f"  skipped missing: {len(report['skipped_missing'])}")
        print(f"  skipped shape mismatch: {len(report['skipped_shape'])}")
        print(f"  model tensors left randomly initialized: {len(report['missing_keys'])}")

    return net, report