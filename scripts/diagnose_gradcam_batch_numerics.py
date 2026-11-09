#!/usr/bin/env python3
"""Diagnose batch-shape numerical differences for one frozen Grad-CAM example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_subgroups_and_gradcam import CONFIG_PATH, ROOT, build_frozen_model, load_image_tensors
from train_simple_cnn import CLASS_ORDER, seed_everything


def main() -> None:
    run = ROOT / "experiments/subgroups_gradcam/primary_test_20261021_162618"
    config = json.loads(CONFIG_PATH.read_text())
    selected = pd.read_csv(run / "selected_examples.csv")
    predictions = pd.read_csv(ROOT / config["inputs"]["test_predictions"])
    first = selected.iloc[0]
    image_id = str(first["image_id"])
    row_index = int(predictions.index[predictions["image_id"] == image_id][0])
    saved = predictions.loc[row_index, [f"probability_calibrated_{label}" for label in CLASS_ORDER]].to_numpy(float)
    device = torch.device("cuda:0")
    seed_everything(43, True)
    model, _ = build_frozen_model(ROOT / config["system"]["checkpoint"], device)
    tensor, _ = load_image_tensors(ROOT / "data" / str(first["relative_path"]))
    results = []
    for size in (1, 2, 32):
        batch = tensor.repeat(size, 1, 1, 1).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch) / config["system"]["temperature"], dim=1)[0].cpu().numpy()
        results.append({"batch": f"repeated_{size}", "max_abs_difference": float(np.max(np.abs(probabilities - saved)))})
    start = row_index // 32 * 32
    stop = min(start + 32, len(predictions))
    tensors = []
    for relative_path in predictions.iloc[start:stop]["relative_path"]:
        value, _ = load_image_tensors(ROOT / "data" / str(relative_path)); tensors.append(value)
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.cat(tensors).to(device)) / config["system"]["temperature"], dim=1)[row_index - start].cpu().numpy()
    results.append({"batch": f"original_manifest_rows_{start}_{stop - 1}", "max_abs_difference": float(np.max(np.abs(probabilities - saved)))})
    output = {"image_id": image_id, "row_index": row_index, "saved_evaluation_batch_size": 32, "results": results}
    (run / "failed_attempt_batch1_diagnosis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
