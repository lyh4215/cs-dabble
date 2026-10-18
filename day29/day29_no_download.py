"""Day29: conditional visual similarity without external downloads.

This is NOT CLIP or the official CLAY method: it trains conditional similarity
from scratch on 1,797 real handwritten digits bundled with scikit-learn.

Requires: torch, numpy, scikit-learn.
No model or dataset downloads.
"""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ==================================================
# 1. Settings
# ==================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(torch.get_num_threads(), 4))


# ==================================================
# 2. Dataset
#
# 8x8 handwritten digit images
# 0 ~ 9
#
# Condition 0: same digit
# Condition 1: same parity (odd/even)
# ==================================================

X, y = load_digits(return_X_y=True)

X = torch.tensor(
    X / 16.0,
    dtype=torch.float32,
)

y = torch.tensor(
    y,
    dtype=torch.long,
)

train, test = train_test_split(
    np.arange(len(X)),
    test_size=0.23,
    stratify=y.numpy(),
    random_state=SEED,
)

xtrain, xtest = X[train], X[test]
ytrain, ytest = y[train], y[test]


# ==================================================
# 3. Conditional embedding model
#
# Shared encoder:
#   image -> 64D embedding
#
# Conditional projection:
#   condition 0 -> digit similarity space
#   condition 1 -> parity similarity space
# ==================================================

class ConditionedMetric(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )

        # Two condition-specific projection matrices
        # [2, 64, 20]

        self.proj = torch.nn.Parameter(
            torch.randn(2, 64, 20) * 0.11
        )

    def forward(self, images, condition):

        z = self.encoder(images)

        # Select projection for the condition
        z = z @ self.proj[condition]

        # L2 normalization for cosine similarity
        return torch.nn.functional.normalize(
            z,
            dim=-1,
            eps=1e-8,
        )


# ==================================================
# 4. Supervised contrastive loss
#
# Positive pairs:
#   condition 0 -> same digit
#   condition 1 -> same odd/even group
#
# Negative pairs:
#   different labels under selected condition
# ==================================================

def supervised_contrastive(
    z,
    labels,
    temperature=0.15,
):

    sims = (z @ z.T) / temperature

    diag = torch.eye(
        len(z),
        dtype=torch.bool,
        device=z.device,
    )

    pos = (
        labels[:, None] == labels[None, :]
    ) & ~diag

    logits = sims.masked_fill(
        diag,
        -1e9,
    )

    log_probs = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    counts = pos.sum(dim=1)
    valid = counts > 0

    return -(
        (
            (log_probs * pos).sum(dim=1)[valid]
            / counts[valid]
        ).mean()
    )


# ==================================================
# 5. Mean Average Precision
#
# For each test image:
#   rank all other images by similarity
#   evaluate retrieval precision
#
# Higher mAP = better retrieval
# ==================================================

def average_precision(
    similarities,
    labels,
):

    scores = similarities.numpy().copy()

    # Exclude the query image itself
    np.fill_diagonal(
        scores,
        -np.inf,
    )

    labels = labels.numpy()
    aps = []

    for i in range(len(labels)):

        ids = np.argsort(
            -scores[i]
        )

        ids = ids[ids != i]

        positive = (
            labels[ids] == labels[i]
        )

        ranks = np.arange(
            1,
            len(ids) + 1,
        )

        precisions = (
            np.cumsum(positive)
            / ranks
        )

        aps.append(
            precisions[positive].mean()
        )

    return np.mean(aps)


# ==================================================
# 6. Training
#
# Alternate between two conditions:
#
# condition 0: distinguish digits 0 ~ 9
# condition 1: distinguish odd/even
# ==================================================

model = ConditionedMetric()

optim = torch.optim.Adam(
    model.parameters(),
    lr=0.002,
)

for step in range(401):

    ids = torch.randint(
        len(train),
        (192,),
    )

    xx = xtrain[ids]
    yy = ytrain[ids]

    cond = step % 2

    if cond == 0:
        labels = yy
    else:
        labels = yy % 2

    z = model(
        xx,
        cond,
    )

    loss = supervised_contrastive(
        z,
        labels,
    )

    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 100 == 0:

        condition_name = (
            "digit"
            if cond == 0
            else "parity"
        )

        print(
            f"step={step:3d} "
            f"condition={condition_name:6s} "
            f"loss={loss.item():.3f}"
        )


# ==================================================
# 7. Generate test embeddings
#
# Same images, different similarity conditions
# ==================================================

model.eval()

with torch.no_grad():

    baseline = torch.nn.functional.normalize(
        xtest,
        dim=1,
    )

    digit_z = model(
        xtest,
        0,
    )

    parity_z = model(
        xtest,
        1,
    )


embeds = {
    "Raw pixels": baseline,
    "Condition: same digit": digit_z,
    "Condition: same parity": parity_z,
}


# ==================================================
# 8. Retrieval evaluation
# ==================================================

print(
    f"\nTest: {len(test)} unseen handwritten digit images"
)

print(
    f'{"Method":25s} '
    f'{"digit mAP":>11s} '
    f'{"parity mAP":>12s}'
)

for name, z in embeds.items():

    sims = z @ z.T

    digit_map = average_precision(
        sims,
        ytest,
    )

    parity_map = average_precision(
        sims,
        ytest % 2,
    )

    print(
        f"{name:25s} "
        f"{digit_map:11.3f} "
        f"{parity_map:12.3f}"
    )


# ==================================================
# 9. Same query, different retrieval results
#
# Query: digit 3
#
# Compare top-12 nearest neighbors
# under different similarity conditions
# ==================================================

q = int(
    torch.where(
        ytest == 3
    )[0][0]
)

print(
    f"\nSame query: handwritten digit {ytest[q].item()}"
)

for name, z in embeds.items():

    scores = z @ z[q]

    scores[q] = -float("inf")

    ranked = torch.topk(
        scores,
        12,
    ).indices

    found = [
        str(int(n))
        for n in ytest[ranked]
    ]

    print(
        f'{name:25s} '
        f'top12: {", ".join(found)}'
    )


print(
    "\nInterpretation: conditional heads can change "
    "retrieval geometry, even on fixed images."
)

print(
    "Not a reproduction of CLIP or CLAY: "
    "condition labels are supervised, not encoded text."
)