
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(torch.get_num_threads(), 4))

# ==========================================
# 1. Dataset
# ==========================================

X, y = load_digits(return_X_y=True)

X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.23,
    stratify=y.numpy(),
    random_state=SEED,
)

xtrain, xtest = X[train_idx], X[test_idx]
ytrain, ytest = y[train_idx], y[test_idx]


# ==========================================
# 2. Shared Encoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. Supervised Contrastive Loss
# ==========================================

def contrastive_loss(z, labels, temperature=0.15):
    z = F.normalize(z, dim=1)

    logits = z @ z.T / temperature

    n = len(z)
    eye = torch.eye(n, dtype=torch.bool)

    positive = (
        labels[:, None] == labels[None, :]
    ) & ~eye

    logits = logits.masked_fill(eye, -1e9)

    log_probs = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )

    counts = positive.sum(dim=1)
    valid = counts > 0

    return -(
        (positive * log_probs).sum(dim=1)[valid]
        / counts[valid]
    ).mean()


# ==========================================
# 4. Train Encoder with Digit Labels
# ==========================================

encoder = Encoder()

optimizer = torch.optim.Adam(
    encoder.parameters(),
    lr=0.002,
)

print("\n=== STAGE 1: ENCODER PRETRAINING ===")

for step in range(401):
    ids = torch.randint(len(xtrain), (192,))

    z = encoder(xtrain[ids])
    labels = ytrain[ids]

    loss = contrastive_loss(z, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"step={step:3d} "
            f"loss={loss.item():.4f}"
        )


# ==========================================
# 5. Freeze Encoder
# ==========================================

encoder.eval()

for p in encoder.parameters():
    p.requires_grad_(False)

# Cache embeddings: no encoder training from now on
with torch.no_grad():
    ztrain = encoder(xtrain)
    ztest = encoder(xtest)

print("\nEncoder frozen!")
print("Embedding dimension:", ztrain.shape[1])


# ==========================================
# 6. Condition-specific Projections
# ==========================================

class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 20, bias=False)

    def forward(self, z):
        return F.normalize(self.linear(z), dim=1)


digit_proj = Projection()
parity_proj = Projection()


def train_projection(projection, labels, name):
    optimizer = torch.optim.Adam(
        projection.parameters(),
        lr=0.003,
    )

    print(f"\n=== TRAIN {name} PROJECTION ===")

    for step in range(401):
        ids = torch.randint(len(ztrain), (192,))

        zz = projection(ztrain[ids])
        yy = labels[ids]

        loss = contrastive_loss(zz, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"step={step:3d} "
                f"loss={loss.item():.4f}"
            )


train_projection(
    digit_proj,
    ytrain,
    "DIGIT",
)

train_projection(
    parity_proj,
    ytrain % 2,
    "PARITY",
)


# ==========================================
# 7. Retrieval mAP
# ==========================================

def calculate_map(z, labels):
    z = F.normalize(z, dim=1)

    scores = (z @ z.T).detach().numpy()
    labels = labels.numpy()

    np.fill_diagonal(scores, -np.inf)

    aps = []

    for i in range(len(labels)):
        ranked = np.argsort(-scores[i])
        ranked = ranked[ranked != i]

        relevant = labels[ranked] == labels[i]

        precision = (
            np.cumsum(relevant)
            / np.arange(1, len(ranked) + 1)
        )

        aps.append(precision[relevant].mean())

    return np.mean(aps)


# ==========================================
# 8. Evaluation
# ==========================================

with torch.no_grad():
    results = {
        "Raw pixels": xtest,
        "Frozen encoder": ztest,
        "Frozen + digit proj": digit_proj(ztest),
        "Frozen + parity proj": parity_proj(ztest),
    }

print("\n=== RETRIEVAL RESULTS ===")
print(
    f'{"Method":25s} '
    f'{"Digit mAP":>10s} '
    f'{"Parity mAP":>11s}'
)

for name, z in results.items():
    digit_map = calculate_map(z, ytest)
    parity_map = calculate_map(z, ytest % 2)

    print(
        f"{name:25s} "
        f"{digit_map:10.3f} "
        f"{parity_map:11.3f}"
    )


# ==========================================
# 9. Same Query, Different Rankings
# ==========================================

query = int(torch.where(ytest == 3)[0][0])

print("\n=== SAME QUERY: DIGIT 3 ===")

for name, z in results.items():
    z = F.normalize(z, dim=1)
    sims = z @ z[query]
    sims[query] = -float("inf")

    top = torch.topk(sims, 12).indices

    matches = ytest[top].tolist()

    print(f"{name:25s} {matches}")


# ==========================================
# 10. Verify Encoder Is Frozen
# ==========================================

print("\n=== FREEZE CHECK ===")

print(
    "Trainable encoder parameters:",
    sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    ),
)

print(
    "Trainable projection parameters:",
    sum(
        p.numel()
        for p in digit_proj.parameters()
    ),
)
