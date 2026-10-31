
#!/usr/bin/env python3
"""DeiT token pruning experiment.

Compare:
  - Full tokens
  - Random pruning
  - Representation Shift pruning

Dependencies:
  pip install torch torchvision timm pillow

Examples:
  python exp2_deit_pruning.py --image cat.jpg
  python exp2_deit_pruning.py --manifest val.csv
  python exp2_deit_pruning.py --synthetic --no-pretrained
"""

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--synthetic", action="store_true")

    parser.add_argument(
        "--model",
        default="deit_tiny_patch16_224.fb_in1k",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--prune-after", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--num-synthetic", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if not 0 < args.keep <= 1:
        parser.error("--keep must be between 0 and 1")

    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    return args


def load_data(args, transform, device):
    if args.synthetic:
        n = min(args.num_synthetic, args.max_images)
        x = torch.randn(n, 3, 224, 224, device=device)
        return [(x, None)]

    if args.image:
        paths = [args.image]
        labels = None

    else:
        paths = []
        labels = []

        with args.manifest.open(
            newline="", encoding="utf-8-sig"
        ) as fp:
            reader = csv.DictReader(fp)

            if not reader.fieldnames or not {
                "path", "label"
            }.issubset(reader.fieldnames):
                raise ValueError(
                    "CSV must contain path,label"
                )

            for row in reader:
                path = Path(row["path"])

                if not path.is_absolute():
                    path = args.manifest.parent / path

                label = int(row["label"])

                if not 0 <= label < 1000:
                    raise ValueError(
                        "ImageNet label must be 0..999"
                    )

                paths.append(path)
                labels.append(label)

                if len(paths) >= args.max_images:
                    break

    if not paths:
        raise ValueError("No input images")

    batches = []

    for start in range(0, len(paths), args.batch_size):
        images = []

        for path in paths[start:start + args.batch_size]:
            with Image.open(path) as img:
                images.append(
                    transform(img.convert("RGB"))
                )

        x = torch.stack(images).to(device)

        y = None
        if labels is not None:
            y = torch.tensor(
                labels[start:start + args.batch_size],
                device=device,
            )

        batches.append((x, y))

    return batches


def embed(model, images):
    # timm VisionTransformer preprocessing
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)

    return x


def prune_tokens(
    before,
    after,
    method,
    fraction,
    generator=None,
):
    """
    before, after: [B, N, C]

    CLS token (index 0) is always preserved.
    Patch tokens are selected independently per batch item.
    """

    batch, n, dim = after.shape
    patches = n - 1

    k = max(
        1,
        min(patches, round(patches * fraction))
    )

    if method == "shift":
        scores = (
            after[:, 1:] - before[:, 1:]
        ).float().norm(dim=-1)

    elif method == "random":
        scores = torch.rand(
            batch,
            patches,
            device=after.device,
            generator=generator,
        )

    else:
        raise ValueError(method)

    # Select top-k patch tokens independently for each image
    indexes = torch.topk(
        scores,
        k=k,
        dim=1,
        sorted=False,
    ).indices + 1

    # Restore original token position order
    indexes = indexes.sort(dim=1).values

    # Preserve CLS token at position 0
    cls_index = torch.zeros(
        batch,
        1,
        dtype=torch.long,
        device=after.device,
    )

    indexes = torch.cat(
        [cls_index, indexes],
        dim=1,
    )

    # [B, K+1] -> [B, K+1, C]
    gather_index = indexes.unsqueeze(-1).expand(
        -1, -1, dim
    )

    return after.gather(
        dim=1,
        index=gather_index,
    )


@torch.inference_mode()
def forward(
    model,
    images,
    method,
    keep,
    prune_after,
    seed,
):
    x = embed(model, images)

    for i, block in enumerate(model.blocks):
        before = x
        x = block(x)

        if i == prune_after and method != "full":
            generator = None

            if method == "random":
                generator = torch.Generator(
                    device=x.device
                ).manual_seed(seed)

            x = prune_tokens(
                before,
                x,
                method,
                keep,
                generator,
            )

    x = model.norm(x)
    return model.forward_head(x)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, images, args, method):
    for _ in range(args.warmup):
        forward(
            model,
            images,
            method,
            args.keep,
            args.prune_after,
            args.seed,
        )

    synchronize(images.device)

    if images.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            images.device
        )

    times = []

    for _ in range(args.repeats):
        synchronize(images.device)
        t0 = time.perf_counter()

        forward(
            model,
            images,
            method,
            args.keep,
            args.prune_after,
            args.seed,
        )

        synchronize(images.device)

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        times.append(elapsed_ms)

    peak = None

    if images.device.type == "cuda":
        peak = (
            torch.cuda.max_memory_allocated(
                images.device
            ) / 1024**2
        )

    return {
        "median_ms": round(
            statistics.median(times), 3
        ),
        "min_ms": round(min(times), 3),
        "peak_cuda_allocated_mib": (
            round(peak, 1) if peak is not None else None
        ),
    }


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    try:
        import timm
        from timm.data import (
            resolve_model_data_config,
            create_transform,
        )
    except ImportError as exc:
        raise SystemExit(
            "Run: pip install timm torchvision pillow"
        ) from exc

    model = timm.create_model(
        args.model,
        pretrained=not args.no_pretrained,
    ).eval().to(device)

    required = [
        "blocks",
        "cls_token",
        "patch_embed",
        "forward_head",
    ]

    if not all(hasattr(model, k) for k in required):
        raise ValueError(
            "This experiment requires a standard DeiT/ViT"
        )

    if getattr(model, "dist_token", None) is not None:
        raise ValueError(
            "Use non-distilled DeiT with one CLS token"
        )

    if not 0 <= args.prune_after < len(model.blocks) - 1:
        raise ValueError(
            "--prune-after must leave subsequent blocks"
        )

    config = resolve_model_data_config(model)

    transform = create_transform(
        **config,
        is_training=False,
    )

    batches = load_data(
        args,
        transform,
        device,
    )

    methods = ["full", "random", "shift"]

    totals = {
        method: {
            "correct": 0,
            "agreed": 0,
            "n": 0,
            "confidence_sum": 0.0,
        }
        for method in methods
    }

    for batch_idx, (x, y) in enumerate(batches):
        logits = {}

        for method in methods:
            logits[method] = forward(
                model,
                x,
                method,
                args.keep,
                args.prune_after,
                args.seed + batch_idx,
            )

        for method in methods:
            probabilities = logits[method].softmax(dim=-1)
            values, indices = probabilities.topk(5, dim=-1)

            print(f"\n{method} top-5:")
            for class_id, prob in zip(indices[0].tolist(), values[0].tolist()):
                print(f"  class={class_id}, probability={prob:.4f}")

        full_pred = logits["full"].argmax(dim=-1)

        for method in methods:
            pred = logits[method].argmax(dim=-1)
            conf = logits[method].softmax(dim=-1)
            conf = conf.max(dim=-1).values

            row = totals[method]

            row["n"] += len(x)
            row["agreed"] += int(
                (pred == full_pred).sum()
            )
            row["confidence_sum"] += float(
                conf.sum()
            )

            if y is not None:
                row["correct"] += int(
                    (pred == y).sum()
                )

    for method in methods:
        row = totals[method]
        n = row["n"]

        row["agreement_with_full_pct"] = round(
            100 * row["agreed"] / n, 2
        )

        row["top1_accuracy_pct"] = (
            round(100 * row["correct"] / n, 2)
            if args.manifest
            else None
        )

        row["mean_confidence_pct"] = round(
            100 * row["confidence_sum"] / n, 2
        )

        for key in [
            "correct",
            "agreed",
            "confidence_sum",
        ]:
            row.pop(key)

        row["benchmark"] = benchmark(
            model,
            batches[0][0],
            args,
            method,
        )

    result = {
        "model": args.model,
        "pretrained": not args.no_pretrained,
        "device": str(device),
        "keep_fraction": args.keep,
        "prune_after_block": args.prune_after,
        "results": totals,
        "note": (
            "Educational Representation Shift implementation. "
            "Not the exact official paper evaluation."
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
