"""Export MobileSAM image encoder + prompt decoder to ONNX.

Output schema matches the standard SAM ONNX contract that the Flutter
service in `lib/services/local_segmentation_service.dart` consumes:

    encoder:
        input:  image           float32[1,3,1024,1024]
        output: image_embeddings

    decoder:
        inputs: image_embeddings, point_coords, point_labels,
                mask_input, has_mask_input, orig_im_size
        output: masks (mask logits at original image size)

Invoked by ``install_sam_model.sh`` — not meant to be run standalone unless
you already have torch + mobile_sam installed.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

OPSET = 17
INPUT_SIZE = 1024


def export_encoder(sam, output: Path) -> None:
    """Trace the TinyViT image encoder to ONNX."""
    encoder = sam.image_encoder
    encoder.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            encoder,
            dummy,
            output.as_posix(),
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"  wrote encoder → {output}")


def export_decoder(sam, output: Path) -> None:
    """Wrap the prompt encoder + mask decoder for ONNX export."""
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=True,
        use_stability_score=False,
        return_extra_metrics=False,
    )

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "point_coords": torch.randint(low=0, high=INPUT_SIZE, size=(1, 5, 2), dtype=torch.float32),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
        "mask_input": torch.randn(1, 1, 4 * embed_size[0], 4 * embed_size[1], dtype=torch.float32),
        "has_mask_input": torch.tensor([1], dtype=torch.float32),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float32),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            output.as_posix(),
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"  wrote decoder → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to mobile_sam.pt")
    parser.add_argument("--encoder-out", type=Path, required=True)
    parser.add_argument("--decoder-out", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_t", help="MobileSAM uses vit_t")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    print(f"loading {args.model_type} from {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint.as_posix())
    sam.eval()

    export_encoder(sam, args.encoder_out)
    export_decoder(sam, args.decoder_out)


if __name__ == "__main__":
    main()
