import json
import difflib
import random
import re
import urllib.request

from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE = (
    "https://raw.githubusercontent.com/"
    "waroad/ReDef/main/dataset/"
)

URLS = {
    "train": BASE + "1_train.jsonl",
    "valid": BASE + "1_valid.jsonl",
    "test": BASE + "1_test.jsonl",
}

LIMITS = {
    "train": 5000,
    "valid": 1500,
    "test": 1500,
}


MAX_VOCAB = 20000
MAX_LEN = 160

EMBED_DIM = 96
CNN_CHANNELS = 64
REP_DIM = 128

BATCH_SIZE = 128
EPOCHS = 5

LR = 1e-3
WEIGHT_DECAY = 1e-4

PAIR_MARGIN = 0.5

LAMBDAS = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
]


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD ReDef
# ============================================================

def load_jsonl(
    url,
    limit,
):

    rows = []

    print(
        f"loading {limit}: {url}"
    )

    with urllib.request.urlopen(
        url
    ) as response:

        for i, raw in enumerate(
            response
        ):

            if i >= limit:
                break

            rows.append(
                json.loads(
                    raw.decode("utf-8")
                )
            )

    return rows


# ============================================================
# EXTRACT ADDED / DELETED CODE
# ============================================================

def extract_change(ex):

    before = ex[
        "function_before"
    ].splitlines()

    after = ex[
        "function_after"
    ].splitlines()


    matcher = difflib.SequenceMatcher(
        None,
        before,
        after,
    )


    added = []
    deleted = []


    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():


        if tag == "delete":

            deleted.extend(
                before[i1:i2]
            )


        elif tag == "insert":

            added.extend(
                after[j1:j2]
            )


        elif tag == "replace":

            deleted.extend(
                before[i1:i2]
            )

            added.extend(
                after[j1:j2]
            )


    return (
        "\n".join(added),
        "\n".join(deleted),
    )


# ============================================================
# SIMPLE CODE TOKENIZER
#
# whitespace split보다 조금 낫게:
#
# identifier
# number
# operator
# punctuation
#
# 등을 분리.
# ============================================================

TOKEN_RE = re.compile(
    r"""
    [A-Za-z_][A-Za-z0-9_]*
    |
    0[xX][0-9A-Fa-f]+
    |
    \d+
    |
    ==|!=|<=|>=|->|&&|\|\||<<|>>
    |
    \+\+|--
    |
    [-+*/%&|^~!=<>?:;,.()\[\]{}]
    """,
    re.VERBOSE,
)


def tokenize(code):

    return TOKEN_RE.findall(
        code
    )


# ============================================================
# PREPARE RAW DATA
# ============================================================

def prepare_raw(examples):

    rows = []


    for ex in examples:

        added, deleted = (
            extract_change(ex)
        )


        if (
            not added.strip()
            and not deleted.strip()
        ):
            continue


        rows.append(
            {
                "added": added,
                "deleted": deleted,

                "label": int(
                    ex[
                        "defective_modification"
                    ]
                ),
            }
        )


    return rows


# ============================================================
# VOCAB
#
# ONLY training set.
# ============================================================

PAD = "<PAD>"
UNK = "<UNK>"
EMPTY = "<EMPTY>"


def build_vocab(
    train_rows,
):

    counter = Counter()


    for row in train_rows:

        counter.update(
            tokenize(
                row["added"]
            )
        )

        counter.update(
            tokenize(
                row["deleted"]
            )
        )


    vocab = {
        PAD: 0,
        UNK: 1,
        EMPTY: 2,
    }


    for token, count in (
        counter.most_common(
            MAX_VOCAB
            - len(vocab)
        )
    ):

        vocab[token] = len(
            vocab
        )


    return vocab


# ============================================================
# ENCODE
# ============================================================

def encode_code(
    code,
    vocab,
):

    tokens = tokenize(
        code
    )


    if not tokens:

        return [
            vocab[EMPTY]
        ]


    ids = [

        vocab.get(
            token,
            vocab[UNK],
        )

        for token in tokens[
            :MAX_LEN
        ]
    ]


    return ids


def encode_rows(
    rows,
    vocab,
):

    encoded = []


    for row in rows:

        encoded.append(
            {
                "added":
                encode_code(
                    row["added"],
                    vocab,
                ),

                "deleted":
                encode_code(
                    row["deleted"],
                    vocab,
                ),

                "label":
                row["label"],
            }
        )


    return encoded


# ============================================================
# DATASET
# ============================================================

class ChangeDataset(Dataset):

    def __init__(
        self,
        rows,
    ):

        self.rows = rows


    def __len__(self):

        return len(
            self.rows
        )


    def __getitem__(
        self,
        idx,
    ):

        return self.rows[
            idx
        ]


# ============================================================
# COLLATE
# ============================================================

def pad_sequences(
    sequences,
):

    lengths = torch.tensor(

        [
            len(x)
            for x in sequences
        ],

        dtype=torch.long,
    )


    max_len = int(
        lengths.max()
    )


    batch = torch.zeros(

        (
            len(sequences),
            max_len,
        ),

        dtype=torch.long,
    )


    for i, seq in enumerate(
        sequences
    ):

        batch[
            i,
            :len(seq)
        ] = torch.tensor(
            seq,
            dtype=torch.long,
        )


    return (
        batch,
        lengths,
    )


def collate_fn(
    batch,
):

    added = [
        x["added"]
        for x in batch
    ]


    deleted = [
        x["deleted"]
        for x in batch
    ]


    labels = torch.tensor(

        [
            x["label"]
            for x in batch
        ],

        dtype=torch.float32,
    )


    added_ids, added_len = (
        pad_sequences(
            added
        )
    )


    deleted_ids, deleted_len = (
        pad_sequences(
            deleted
        )
    )


    return (
        added_ids,
        added_len,
        deleted_ids,
        deleted_len,
        labels,
    )


def make_loader(
    rows,
    shuffle,
    seed=0,
):

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )


    return DataLoader(

        ChangeDataset(rows),

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        collate_fn=collate_fn,

        generator=generator,
    )


# ============================================================
# CODE ENCODER
#
# tokens
#   ↓
# Embedding
#   ↓
# Conv kernel 3
# Conv kernel 5
# Conv kernel 7
#   ↓
# global max pool
#   ↓
# learned representation h
# ============================================================

class CodeEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
    ):

        super().__init__()


        self.embedding = nn.Embedding(

            vocab_size,

            EMBED_DIM,

            padding_idx=0,
        )


        self.convs = nn.ModuleList(

            [

                nn.Conv1d(
                    EMBED_DIM,
                    CNN_CHANNELS,
                    kernel_size=k,
                    padding=k // 2,
                )

                for k in (
                    3,
                    5,
                    7,
                )
            ]
        )


        self.projection = nn.Linear(

            CNN_CHANNELS * 3,

            REP_DIM,
        )


        self.norm = nn.LayerNorm(
            REP_DIM
        )


    def forward(
        self,
        ids,
        lengths,
    ):

        # -----------------------------------------
        # [B, L]
        #   ↓
        # [B, L, E]
        # -----------------------------------------

        x = self.embedding(
            ids
        )


        # Conv1d expects:
        #
        # [B, E, L]

        x = x.transpose(
            1,
            2,
        )


        L = ids.shape[1]


        mask = (

            torch.arange(
                L,
                device=ids.device,
            )[None, :]

            <

            lengths[
                :,
                None
            ].to(
                ids.device
            )
        )


        pooled = []


        for conv in self.convs:

            z = F.relu(
                conv(x)
            )


            # [B, C, L]

            z = z.masked_fill(

                ~mask[
                    :,
                    None,
                    :
                ],

                -1e9,
            )


            z = z.max(
                dim=2
            ).values


            pooled.append(
                z
            )


        h = torch.cat(
            pooled,
            dim=1,
        )


        h = self.projection(
            h
        )


        h = F.relu(
            h
        )


        h = self.norm(
            h
        )


        return h


# ============================================================
# CHANGE RELATION MODEL
#
# add → h_a
# del → h_d
#
# [
#   h_a,
#   h_d,
#   h_a-h_d,
#   h_a*h_d
# ]
#
# ============================================================

class ChangeModel(nn.Module):

    def __init__(
        self,
        vocab_size,
    ):

        super().__init__()


        self.encoder = CodeEncoder(
            vocab_size
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                REP_DIM * 4,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                1,
            ),
        )


    def score_from_repr(
        self,
        h_add,
        h_del,
    ):

        relation = torch.cat(

            [
                h_add,
                h_del,

                h_add
                - h_del,

                h_add
                * h_del,
            ],

            dim=1,
        )


        return (
            self.classifier(
                relation
            )
            .squeeze(1)
        )


    def forward(
        self,
        added_ids,
        added_len,
        deleted_ids,
        deleted_len,
    ):

        h_add = self.encoder(

            added_ids,

            added_len,
        )


        h_del = self.encoder(

            deleted_ids,

            deleted_len,
        )


        original = (
            self.score_from_repr(
                h_add,
                h_del,
            )
        )


        # -----------------------------------------
        # Counterfactual:
        #
        # Added ↔ Deleted
        #
        # encoder를 다시 돌릴 필요 없이
        # representation만 swap.
        # -----------------------------------------

        reverse = (
            self.score_from_repr(
                h_del,
                h_add,
            )
        )


        return (
            original,
            reverse,
        )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    train_rows,
    vocab_size,
    lam,
    pos_weight,
):

    set_seed(0)


    model = ChangeModel(
        vocab_size
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY,
    )


    loader = make_loader(

        train_rows,

        shuffle=True,

        seed=0,
    )


    print(
        f"\n--- TRAIN λ={lam} ---"
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()


        total_loss = 0.0
        total_bce = 0.0
        total_rank = 0.0


        for (
            added_ids,
            added_len,
            deleted_ids,
            deleted_len,
            labels,
        ) in loader:


            added_ids = (
                added_ids.to(
                    DEVICE
                )
            )

            added_len = (
                added_len.to(
                    DEVICE
                )
            )

            deleted_ids = (
                deleted_ids.to(
                    DEVICE
                )
            )

            deleted_len = (
                deleted_len.to(
                    DEVICE
                )
            )

            labels = (
                labels.to(
                    DEVICE
                )
            )


            orig_logit, rev_logit = (
                model(
                    added_ids,
                    added_len,
                    deleted_ids,
                    deleted_len,
                )
            )


            # =====================================
            # 1. Standard defect classification
            # =====================================

            bce = (
                F.binary_cross_entropy_with_logits(

                    orig_logit,

                    labels,

                    pos_weight=torch.tensor(
                        pos_weight,
                        device=DEVICE,
                    ),
                )
            )


            # =====================================
            # 2. Pairwise direction objective
            #
            # ONLY defective changes
            #
            # orig - reverse >= margin
            # =====================================

            defective = (
                labels == 1
            )


            if (
                lam > 0
                and
                defective.any()
            ):

                score_gap = (

                    orig_logit[
                        defective
                    ]

                    -

                    rev_logit[
                        defective
                    ]
                )


                rank_loss = (
                    F.relu(

                        PAIR_MARGIN

                        -

                        score_gap
                    )
                    .mean()
                )


            else:

                rank_loss = (
                    torch.tensor(
                        0.0,
                        device=DEVICE,
                    )
                )


            loss = (

                bce

                +

                lam
                * rank_loss
            )


            optimizer.zero_grad()


            loss.backward()


            nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0,
            )


            optimizer.step()


            total_loss += (
                loss.item()
            )

            total_bce += (
                bce.item()
            )

            total_rank += (
                rank_loss.item()
            )


        n_batches = len(
            loader
        )


        print(

            f"epoch={epoch} "

            f"loss="
            f"{total_loss/n_batches:.4f} "

            f"bce="
            f"{total_bce/n_batches:.4f} "

            f"rank="
            f"{total_rank/n_batches:.4f}"
        )


    return model


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    rows,
):

    loader = make_loader(

        rows,

        shuffle=False,
    )


    model.eval()


    all_y = []
    all_orig = []
    all_rev = []


    for (
        added_ids,
        added_len,
        deleted_ids,
        deleted_len,
        labels,
    ) in loader:


        added_ids = added_ids.to(
            DEVICE
        )

        added_len = added_len.to(
            DEVICE
        )

        deleted_ids = deleted_ids.to(
            DEVICE
        )

        deleted_len = deleted_len.to(
            DEVICE
        )


        orig_logit, rev_logit = (
            model(

                added_ids,
                added_len,

                deleted_ids,
                deleted_len,
            )
        )


        orig_prob = (
            torch.sigmoid(
                orig_logit
            )
            .cpu()
            .numpy()
        )


        rev_prob = (
            torch.sigmoid(
                rev_logit
            )
            .cpu()
            .numpy()
        )


        all_orig.extend(
            orig_prob
        )

        all_rev.extend(
            rev_prob
        )

        all_y.extend(
            labels.numpy()
        )


    y = np.array(
        all_y,
        dtype=int,
    )


    p = np.array(
        all_orig
    )


    p_rev = np.array(
        all_rev
    )


    pred = (
        p >= 0.5
    ).astype(int)


    pred_rev = (
        p_rev >= 0.5
    ).astype(int)


    defective = (
        y == 1
    )


    direction_consistency = np.mean(

        p[
            defective
        ]

        >

        p_rev[
            defective
        ]
    )


    direction_margin = np.mean(

        p[
            defective
        ]

        -

        p_rev[
            defective
        ]
    )


    return {

        "accuracy":
        accuracy_score(
            y,
            pred,
        ),

        "precision":
        precision_score(
            y,
            pred,
            zero_division=0,
        ),

        "recall":
        recall_score(
            y,
            pred,
            zero_division=0,
        ),

        "f1":
        f1_score(
            y,
            pred,
            zero_division=0,
        ),

        "direction_consistency":
        direction_consistency,

        "direction_margin":
        direction_margin,

        "prob_delta":
        np.mean(
            np.abs(
                p
                - p_rev
            )
        ),

        "prediction_change":
        np.mean(
            pred
            != pred_rev
        ),
    }


def print_metrics(
    lam,
    m,
):

    print(

        f"λ={lam:<4} "

        f"F1={m['f1']:.4f}  "

        f"DirCons="
        f"{m['direction_consistency']:.2%}  "

        f"Margin="
        f"{m['direction_margin']:+.4f}  "

        f"ΔP="
        f"{m['prob_delta']:.4f}  "

        f"PredChange="
        f"{m['prediction_change']:.2%}"
    )


# ============================================================
# MAIN
# ============================================================

set_seed(0)


print(
    "device:",
    DEVICE
)


print(
    "\n================================"
)

print(
    "LOAD"
)

print(
    "================================"
)


raw_train = load_jsonl(
    URLS["train"],
    LIMITS["train"],
)

raw_valid = load_jsonl(
    URLS["valid"],
    LIMITS["valid"],
)

raw_test = load_jsonl(
    URLS["test"],
    LIMITS["test"],
)


train_raw = prepare_raw(
    raw_train
)

valid_raw = prepare_raw(
    raw_valid
)

test_raw = prepare_raw(
    raw_test
)


print(
    "\ntrain:",
    len(train_raw)
)

print(
    "valid:",
    len(valid_raw)
)

print(
    "test:",
    len(test_raw)
)


# ============================================================
# VOCAB
# ============================================================

print(
    "\n================================"
)

print(
    "BUILD VOCAB"
)

print(
    "================================"
)


vocab = build_vocab(
    train_raw
)


print(
    "vocab size:",
    len(vocab)
)


# ============================================================
# ENCODE
# ============================================================

train = encode_rows(
    train_raw,
    vocab,
)

valid = encode_rows(
    valid_raw,
    vocab,
)

test = encode_rows(
    test_raw,
    vocab,
)


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

train_labels = np.array(

    [
        x["label"]
        for x in train
    ]
)


n_pos = np.sum(
    train_labels == 1
)

n_neg = np.sum(
    train_labels == 0
)


pos_weight = (
    n_neg
    / n_pos
)


print(
    "positive:",
    n_pos
)

print(
    "negative:",
    n_neg
)

print(
    "pos_weight:",
    round(
        pos_weight,
        3,
    )
)


# ============================================================
# λ SEARCH
# ============================================================

print(
    "\n================================"
)

print(
    "VALIDATION — λ SEARCH"
)

print(
    "================================"
)


results = []


for lam in LAMBDAS:

    model = train_model(

        train,

        len(vocab),

        lam,

        pos_weight,
    )


    metrics = evaluate(

        model,

        valid,
    )


    print(
        "\nVALID:"
    )

    print_metrics(
        lam,
        metrics,
    )


    results.append(
        (
            lam,
            model,
            metrics,
        )
    )


# ============================================================
# MODEL SELECTION
#
# best F1 - 0.01 이내에서
# direction consistency 최대
# ============================================================

best_f1 = max(

    metrics["f1"]

    for _, _, metrics
    in results
)


f1_floor = (
    best_f1
    - 0.01
)


eligible = [

    result

    for result
    in results

    if result[2]["f1"]
    >= f1_floor
]


selected = max(

    eligible,

    key=lambda x:
    (
        x[2][
            "direction_consistency"
        ],

        x[2][
            "direction_margin"
        ],
    )
)


selected_lambda, \
selected_model, \
selected_valid = selected


print(
    "\n================================"
)

print(
    "SELECT MODEL"
)

print(
    "================================"
)


print(
    "best valid F1:",
    f"{best_f1:.4f}"
)


print(
    "F1 floor:",
    f"{f1_floor:.4f}"
)


print(
    "selected λ:",
    selected_lambda
)


print_metrics(
    selected_lambda,
    selected_valid,
)


# ============================================================
# TEST
# ============================================================

print(
    "\n================================"
)

print(
    "FINAL TEST"
)

print(
    "================================"
)


selected_test = evaluate(

    selected_model,

    test,
)


print_metrics(

    selected_lambda,

    selected_test,
)


# ============================================================
# BASELINE λ=0
# ============================================================

baseline_model = next(

    model

    for lam, model, metrics
    in results

    if lam == 0.0
)


baseline_test = evaluate(

    baseline_model,

    test,
)


print(
    "\n================================"
)

print(
    "BASELINE vs SELECTED"
)

print(
    "================================"
)


print(
    "BASELINE"
)

print_metrics(
    0.0,
    baseline_test,
)


print(
    "\nSELECTED"
)

print_metrics(
    selected_lambda,
    selected_test,
)


print(
    "\nΔ F1:",
    f"{selected_test['f1'] - baseline_test['f1']:+.4f}"
)


print(
    "Δ DirCons:",
    f"{selected_test['direction_consistency'] - baseline_test['direction_consistency']:+.2%}"
)


print(
    "Δ Margin:",
    f"{selected_test['direction_margin'] - baseline_test['direction_margin']:+.4f}"
)