#!/usr/bin/env python3
"""Create Orin AP model directories by rewriting dataset paths only."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path("/home/jichengzhi/lane_c_orin_stage6_five_config_20260724")
DATASET = Path(
    "/home/jichengzhi/lane_c_orin_backbone_parity_20260723/"
    "dair_val_full/cooperative-vehicle-infrastructure"
)
SOURCES = (
    ("pyramid", "064x128x256", "net_epoch_bestval_at23.pth"),
    ("pyramid", "024x128x064", "net_epoch_bestval_at1.pth"),
    ("pyramid", "032x048x128", "net_epoch_bestval_at1.pth"),
    ("pyramid", "016x032x064", "net_epoch_bestval_at31.pth"),
    ("codriving", "064x128x256", "net_epoch29.pth"),
    ("codriving", "032x064x096", "net_epoch_bestval_at11.pth"),
    ("codriving", "032x080x128", "net_epoch_bestval_at5.pth"),
    ("codriving", "016x032x064", "net_epoch_bestval_at9.pth"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_config(source: Path, target: Path, model: str) -> dict[str, object]:
    values = {
        "data_dir": str(DATASET),
        "root_dir": str(DATASET / "train.json"),
        "validate_dir": str(DATASET / "val.json"),
        "test_dir": str(DATASET / "val.json"),
    }
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    changed: list[str] = []
    rewritten = []
    for line in lines:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        key = match.group(1) if match else None
        if key in values:
            rewritten.append(f"{key}: {values[key]}\n")
            changed.append(key)
        else:
            rewritten.append(line)
    if set(changed) != set(values):
        raise ValueError(
            f"dataset rewrite mismatch for {source}: {changed} != {sorted(values)}"
        )
    target.write_text("".join(rewritten), encoding="utf-8")
    return {
        "source_config": str(source),
        "source_config_sha256": sha256(source),
        "derived_config": str(target),
        "derived_config_sha256": sha256(target),
        "allowed_changes": sorted(values),
        "actual_changes": sorted(changed),
        "derived_values": values,
    }


def main() -> int:
    if not (DATASET / "val.json").is_file():
        raise FileNotFoundError(DATASET / "val.json")
    audits = []
    for model, width, checkpoint_name in SOURCES:
        source_dir = ROOT / "source" / model / width
        model_dir = ROOT / "models" / model / width
        model_dir.mkdir(parents=True, exist_ok=True)
        audit = rewrite_config(
            source_dir / "config.yaml", model_dir / "config.yaml", model
        )
        source_checkpoint = source_dir / checkpoint_name
        target_checkpoint = model_dir / checkpoint_name
        shutil.copy2(source_checkpoint, target_checkpoint)
        epoch_match = re.search(r"bestval_at(\d+)", checkpoint_name)
        alias = None
        if epoch_match is not None:
            alias = model_dir / f"net_epoch{epoch_match.group(1)}.pth"
            shutil.copy2(source_checkpoint, alias)
        audit.update(
            {
                "model": model,
                "width": width,
                "source_checkpoint": str(source_checkpoint),
                "checkpoint_sha256": sha256(source_checkpoint),
                "derived_checkpoint": str(target_checkpoint),
                "derived_checkpoint_sha256": sha256(target_checkpoint),
                "checkpoint_alias": str(alias) if alias else None,
                "checkpoint_alias_sha256": sha256(alias) if alias else None,
                "unexpected_changes": [],
            }
        )
        (model_dir / "derivation_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audits.append(audit)
    output = ROOT / "contracts" / "ap_model_derivation_audit.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": "lane_c_stage6_orin_ap_model_derivation_v1",
                "dataset": str(DATASET),
                "rows": audits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
