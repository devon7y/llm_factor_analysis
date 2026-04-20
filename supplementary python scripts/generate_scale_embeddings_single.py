#!/usr/bin/env python3
"""Generate embeddings for a single scale CSV in the repo's cache format."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def detect_device() -> tuple[str, int]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-csv", required=True, help="Path to scale CSV")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-8B",
        help="Model name or local path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .npz path; defaults to embeddings/{stem}_{size}.npz",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving",
    )
    args = parser.parse_args()

    scale_csv = Path(args.scale_csv)
    if not scale_csv.exists():
        raise FileNotFoundError(f"Scale CSV not found: {scale_csv}")

    scale_df = pd.read_csv(scale_csv)
    required_cols = ["code", "item"]
    missing_cols = [col for col in required_cols if col not in scale_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    codes = scale_df["code"].tolist()
    items = scale_df["item"].tolist()
    factors = scale_df["factor"].tolist() if "factor" in scale_df.columns else None
    scoring = scale_df["scoring"].tolist() if "scoring" in scale_df.columns else None

    model_size = args.model_name.split("-")[-1]
    data_name = scale_csv.stem
    output_path = (
        Path(args.output)
        if args.output
        else Path("embeddings") / f"{data_name}_{model_size}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device, num_devices = detect_device()
    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}")
    if device == "cuda":
        for idx in range(num_devices):
            print(f"  GPU {idx}: {torch.cuda.get_device_name(idx)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model_kwargs: dict[str, object] = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    print(f"Encoding {len(items)} items from {scale_csv} ...")
    batches = []
    for start in range(0, len(items), args.batch_size):
        end = min(start + args.batch_size, len(items))
        print(f"  Batch {start + 1}-{end}")
        batch_text = items[start:end]
        batch_dict = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
        with torch.inference_mode():
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],
            )
            if args.normalize_embeddings:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        batches.append(batch_embeddings.float().cpu().numpy())
    embeddings = np.vstack(batches)
    print(f"Generated embeddings: {embeddings.shape}")

    metadata = {
        "model_name": args.model_name,
        "model_size": model_size,
        "embedding_dim": int(embeddings.shape[1]),
        "num_items": len(items),
        "embedding_mode": "scale",
        "normalized": bool(args.normalize_embeddings),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device": device,
        "multi_gpu": False,
        "num_gpus": num_devices,
        "scale_csv_path": str(scale_csv),
        "data_name": data_name,
        "num_codes": len(codes),
        "backend": "transformers",
    }

    save_dict = {
        "embeddings": embeddings,
        "scale_embeddings": embeddings,
        "codes": np.array(codes),
        "items": np.array(items),
        "metadata": np.array(metadata, dtype=object),
    }
    if factors is not None:
        save_dict["factors"] = np.array(factors)
    if scoring is not None:
        save_dict["scoring"] = np.array(scoring)

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
