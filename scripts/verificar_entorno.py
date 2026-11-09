#!/usr/bin/env python3
"""Comprueba el entorno del TFM sin descargar datos ni iniciar entrenamientos."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def command_output(command: list[str]) -> dict[str, object]:
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    return {"ok": process.returncode == 0, "returncode": process.returncode, "output": output}


def main() -> int:
    import cv2
    import matplotlib
    import numpy
    import pandas
    import scipy
    import seaborn
    import sklearn
    import skimage
    import streamlit
    import torch
    import torchvision
    import yaml
    from pytorch_grad_cam import GradCAM  # noqa: F401
    from torchvision.models import efficientnet_b0, mobilenet_v3_small

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "streamlit": streamlit.__version__,
            "pyyaml": yaml.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    }

    test_input = torch.zeros(1, 3, 224, 224)
    model_shapes: dict[str, list[int]] = {}
    with torch.inference_mode():
        for name, model in {
            "mobilenet_v3_small": mobilenet_v3_small(weights=None, num_classes=7),
            "efficientnet_b0": efficientnet_b0(weights=None, num_classes=7),
        }.items():
            model_shapes[name] = list(model.eval()(test_input).shape)
    report["model_smoke_test"] = model_shapes

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        tensor = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
        result = tensor @ tensor
        torch.cuda.synchronize(device)
        report["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "matmul_sum": float(result.sum().cpu()),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if torch.cuda.is_available() else 2


if __name__ == "__main__":
    raise SystemExit(main())

