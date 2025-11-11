#!/usr/bin/env python3
"""
Quantize the Depth Anything ONNX model to reduce memory footprint on device.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic-quantize a Depth Anything ONNX")
    parser.add_argument("--input", required=True, type=Path, help="Path to fp32 ONNX model")
    parser.add_argument("--output", required=True, type=Path, help="Output path for quantized ONNX")
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization (slower to build, better accuracy)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model written to {args.output}")


if __name__ == "__main__":
    main()
