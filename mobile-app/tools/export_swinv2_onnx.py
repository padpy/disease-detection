"""Export the SwinV2 grape-leaf classifier to ONNX for on-device use.

Mirrors the server-side classifier in
``server/app/classification.py`` (HuggingFace ``AutoModelForImageClassification``,
input is ``[1, 3, 224, 224]`` after ``transforms.functional.to_tensor`` +
``Resize((224, 224))``). The export keeps that 224 input size and the
3-class label order from ``server/app/application.py``:

    label2id = {'Healthy-Leaf': 0, 'Downy-Leaf': 1, 'Powdery-Leaf': 2}

Output schema::

    input:  pixel_values  float32[1, 3, 224, 224]   (RGB, scaled to [0, 1])
    output: logits        float32[1, 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification

OPSET = 17
INPUT_SIZE = 224
LABEL2ID = {"Healthy-Leaf": 0, "Downy-Leaf": 1, "Powdery-Leaf": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the SwinV2 model directory "
                             "(contains config.json + model.safetensors)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination ONNX path")
    parser.add_argument("--imgsz", type=int, default=INPUT_SIZE)
    parser.add_argument("--opset", type=int, default=OPSET)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"checkpoint dir not found: {args.checkpoint}")

    print(f"loading {args.checkpoint}")
    model = AutoModelForImageClassification.from_pretrained(
        args.checkpoint.as_posix(),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.m(pixel_values=pixel_values).logits

    wrapped = LogitsOnly(model).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting to ONNX (imgsz={args.imgsz}, opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            args.output.as_posix(),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
