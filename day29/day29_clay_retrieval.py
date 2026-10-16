"""Day29: CLAY-inspired conditional image retrieval with real pretrained CLIP.

Not the official CLAY implementation. Reproduces its key mechanism:
 text-conditioned tangent-space SVD -> visual subspace projection -> cosine retrieval.
Uses one Householder reflection for mean alignment (official CLAY uses a two-step rotation).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def norm(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def log_map(x, mu):
    """Logarithm map from the unit sphere to tangent space at unit vector mu."""
    dots = (x * mu).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dots)
    tangent = x - dots * mu
    return tangent * (theta / torch.sin(theta).clamp_min(1e-6))


def conditioned_features(images, texts, k):
    """CLAY-inspired: textual manifold subspace + visual alignment + projection.

    All inputs are precomputed CLIP unit vectors [N,D]/[M,D].
    This is a teaching approximation: one reflection rather than paper's H2 H1.
    """
    mu_t = norm(texts.mean(dim=0, keepdim=True)).squeeze(0)
    z_t = log_map(texts, mu_t)
    _, _, vh = torch.linalg.svd(z_t, full_matrices=False)
    basis = vh[: min(k, len(texts) - 1)].T  # [D, k], excludes null-space vectors

    # Orthogonal Householder transformation: align visual mean to textual mean.
    mu_v = norm(images.mean(dim=0, keepdim=True)).squeeze(0)
    v = mu_v - mu_t
    if v.norm() > 1e-7:
        u = v / v.norm()
        aligned = images - 2.0 * (images @ u)[:, None] * u[None, :]
    else:
        aligned = images

    projected = log_map(norm(aligned), mu_t) @ basis
    return norm(projected)


def mean_average_precision(sim, relevant_labels):
    """AP over all other gallery examples, averaged over queries."""
    n = len(relevant_labels)
    scores = sim.detach().cpu().numpy().copy()
    np.fill_diagonal(scores, -np.inf)
    labels = np.asarray(relevant_labels)
    aps = []
    for q in range(n):
        ranked = np.argsort(-scores[q])
        ranked = ranked[ranked != q]
        flags = labels[ranked] == labels[q]
        if not flags.any():
            continue
        precision_at_rank = np.cumsum(flags) / np.arange(1, len(flags) + 1)
        aps.append(float(precision_at_rank[flags].mean()))
    return float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-dir', type=str, default='./data')
    args = parser.parse_args()

    from torchvision.datasets import CIFAR10
    from transformers import CLIPModel, AutoProcessor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    print('Loading CIFAR-10 test set and pretrained CLIP (first run downloads weights)...')
    ds = CIFAR10(root=args.data_dir, train=False, download=True)
    classes = list(ds.classes)
    picks = defaultdict(list)
    for idx, target in enumerate(ds.targets):
        if len(picks[target]) < args.per_class:
            picks[target].append(idx)
        if len(picks) == len(classes) and all(len(x) >= args.per_class for x in picks.values()):
            break

    indices = [i for cls in range(len(classes)) for i in picks[cls]]
    pil_images = [ds[i][0] for i in indices]
    fine_labels = np.asarray([ds.targets[i] for i in indices])
    animals = {'bird', 'cat', 'deer', 'dog', 'frog', 'horse'}
    coarse_labels = np.asarray(['animal' if classes[i] in animals else 'vehicle' for i in fine_labels])

    model_id = 'openai/clip-vit-base-patch32'
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    vectors = []
    with torch.inference_mode():
        for s in range(0, len(pil_images), args.batch_size):
            image_inputs = processor(images=pil_images[s:s+args.batch_size], return_tensors='pt')
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            v = model.get_image_features(**image_inputs)
            vectors.append(norm(v).cpu())
    visual = torch.cat(vectors, dim=0)

    banks = {
        'coarse': ['a photo of an animal', 'a photo of a vehicle'],
        'fine': [f'a photo of a {c}' for c in classes],
    }
    text_banks = {}
    with torch.inference_mode():
        for condition, prompts in banks.items():
            tx = processor(text=prompts, padding=True, truncation=True, return_tensors='pt')
            tx = {key: val.to(device) for key, val in tx.items()}
            text_banks[condition] = norm(model.get_text_features(**tx)).cpu()

    # All image features are cached once; changing condition only changes the projection.
    embeddings = {
        'CLIP original': visual,
        'Condition: coarse': conditioned_features(visual, text_banks['coarse'], k=1),
        'Condition: fine': conditioned_features(visual, text_banks['fine'], k=8),
    }
    similarities = {name: emb @ emb.T for name, emb in embeddings.items()}

    print(f'\nimages={len(indices)} classes={len(classes)} visual_embedding_dim={visual.shape[1]}')
    print('=== MEAN AVERAGE PRECISION (higher is better) ===')
    print(f"{'Method':23s} {'same fine class':>15s} {'same coarse group':>20s}")
    for name, sim in similarities.items():
        fine = mean_average_precision(sim, fine_labels)
        coarse = mean_average_precision(sim, coarse_labels)
        print(f'{name:23s} {fine:15.3f} {coarse:20.3f}')

    # Query classes: one animal and one vehicle; inspect ranking per condition.
    print('\n=== SAME QUERY, DIFFERENT SIMILARITY CRITERION ===')
    for target in ['cat', 'truck']:
        q = next(i for i, cls in enumerate(fine_labels) if classes[cls] == target)
        print(f'\nQUERY #{q} = {target}')
        for name, sim in similarities.items():
            scores = sim[q].clone()
            scores[q] = -float('inf')
            top = torch.topk(scores, k=7).indices.tolist()
            ranked_names = [classes[fine_labels[i]] for i in top]
            print(f'  {name:23s}: {ranked_names}')

    print('\nNote: changing the condition alters retrieval geometry, not the cached image embeddings.')
    print('No improvement is guaranteed. Inspect where the projection helps or loses information.')


if __name__ == '__main__':
    main()
