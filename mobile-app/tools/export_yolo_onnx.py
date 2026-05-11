"""Export an Ultralytics YOLO ``.pt`` checkpoint to ONNX for on-device use.

The Flutter pipeline in ``lib/services/wheat_head_pipeline.dart`` consumes
the standard Ultralytics ONNX schema:

    input:  images   float32[1, 3, 640, 640]   (RGB, scaled to [0, 1])
    output: output0  float32[1, 4 + nc, N]     (raw head)

Invoked from ``install_sam_model.sh``. Standalone use is fine if you have
``ultralytics`` installed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

OPSET = 17
IMG_SIZE = 640


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the YOLO .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination ONNX path")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--opset", type=int, default=OPSET)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    print(f"loading {args.checkpoint}")
    model = YOLO(args.checkpoint.as_posix())

    # Ultralytics exports next to the .pt; we move the result afterwards.
    print(f"exporting to ONNX (imgsz={args.imgsz}, opset={args.opset})")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
    )
    exported_path = Path(exported)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != args.output.resolve():
        shutil.move(exported_path.as_posix(), args.output.as_posix())
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
