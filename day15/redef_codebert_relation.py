import json
import difflib
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
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

MODEL_NAME = "microsoft/codebert-base"

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

MAX_LEN = 160

BATCH_SIZE = 8

EPOCHS = 3

LR = 2e-5

WEIGHT_DECAY = 0.01

PAIR_MARGIN = 0.5

LAMBDAS = [
    0.0,
    0.1,
    0.25,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed=0):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD
# ============================================================

def load_jsonl(url, limit):

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
# EXTRACT CHANGE
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


def prepare(examples):

    rows = []

    for ex in examples:

        added, deleted = (
            extract_change(ex)
        )

        if (
            not added.strip()
            and
            not deleted.strip()
        ):
            continue

        # CodeBERT tokenizer에 빈 문자열도 넣을 수는 있지만,
        # 방향을 명시적으로 구분하기 위해 placeholder 사용.
        if not added.strip():
            added = "<EMPTY_CHANGE>"

        if not deleted.strip():
            deleted = "<EMPTY_CHANGE>"

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
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def collate_fn(batch):

    added_text = [
        x["added"]
        for x in batch
    ]

    deleted_text = [
        x["deleted"]
        for x in batch
    ]


    added = tokenizer(

        added_text,

        padding=True,

        truncation=True,

        max_length=MAX_LEN,

        return_tensors="pt",
    )


    deleted = tokenizer(

        deleted_text,

        padding=True,

        truncation=True,

        max_length=MAX_LEN,

        return_tensors="pt",
    )


    labels = torch.tensor(

        [
            x["label"]
            for x in batch
        ],

        dtype=torch.float32,
    )


    return (
        added,
        deleted,
        labels,
    )


def make_loader(
    rows,
    shuffle,
):

    return DataLoader(

        ChangeDataset(
            rows
        ),

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        collate_fn=collate_fn,
    )


# ============================================================
# MODEL
# ============================================================

class CodeBERTChangeModel(nn.Module):

    def __init__(self):

        super().__init__()


        self.encoder = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )


        hidden = (
            self.encoder
            .config
            .hidden_size
        )


        print(
            "CodeBERT hidden:",
            hidden
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden * 4,
                512,
            ),

            nn.GELU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                512,
                1,
            ),
        )


    def encode(
        self,
        batch,
    ):

        output = self.encoder(

            input_ids=batch[
                "input_ids"
            ],

            attention_mask=batch[
                "attention_mask"
            ],
        )


        # CodeBERT = RoBERTa 계열
        #
        # 첫 token <s> representation을
        # sequence representation으로 사용.
        h = (
            output
            .last_hidden_state[
                :,
                0,
                :
            ]
        )


        return h


    def score(
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
        added,
        deleted,
    ):

        h_add = self.encode(
            added
        )

        h_del = self.encode(
            deleted
        )


        original = self.score(
            h_add,
            h_del,
        )


        # Counterfactual:
        #
        # added ↔ deleted
        #
        # CodeBERT를 다시 실행하지 않고
        # 이미 얻은 representation만 swap.
        reverse = self.score(
            h_del,
            h_add,
        )


        return (
            original,
            reverse,
        )


# ============================================================
# MOVE TOKEN BATCH TO GPU
# ============================================================

def to_device(batch):

    return {

        key:
        value.to(
            DEVICE
        )

        for key, value
        in batch.items()
    }


# ============================================================
# TRAIN
# ============================================================

def train_model(
    train_rows,
    lam,
    pos_weight,
):

    set_seed(0)


    model = (
        CodeBERTChangeModel()
        .to(
            DEVICE
        )
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY,
    )


    loader = make_loader(

        train_rows,

        shuffle=True,
    )


    pos_weight_tensor = (
        torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )


    # AMP:
    # Colab GPU에서 훨씬 효율적.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type
            == "cuda"
        ),
    )


    print(
        f"\n--- TRAIN λ={lam} ---"
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()


        total = 0.0
        total_bce = 0.0
        total_rank = 0.0


        for (
            added,
            deleted,
            labels,
        ) in loader:


            added = to_device(
                added
            )

            deleted = to_device(
                deleted
            )

            labels = labels.to(
                DEVICE
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.amp.autocast(

                "cuda",

                enabled=(
                    DEVICE.type
                    == "cuda"
                ),
            ):


                orig_logit, rev_logit = (
                    model(
                        added,
                        deleted,
                    )
                )


                # --------------------------------
                # standard defect loss
                # --------------------------------

                bce = (
                    F.binary_cross_entropy_with_logits(

                        orig_logit,

                        labels,

                        pos_weight=
                        pos_weight_tensor,
                    )
                )


                # --------------------------------
                # pairwise semantic direction loss
                # --------------------------------

                defective = (
                    labels == 1
                )


                if (
                    lam > 0
                    and
                    defective.any()
                ):

                    gap = (

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

                            gap
                        )

                        .mean()
                    )


                else:

                    rank_loss = (
                        orig_logit.sum()
                        * 0.0
                    )


                loss = (

                    bce

                    +

                    lam
                    * rank_loss
                )


            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0,
            )


            scaler.step(
                optimizer
            )


            scaler.update()


            total += (
                loss.item()
            )

            total_bce += (
                bce.item()
            )

            total_rank += (
                rank_loss.item()
            )


        n = len(loader)


        print(

            f"epoch={epoch} "

            f"loss="
            f"{total/n:.4f} "

            f"bce="
            f"{total_bce/n:.4f} "

            f"rank="
            f"{total_rank/n:.4f}"
        )


    return model


# ============================================================
# EVALUATE
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


    ys = []
    originals = []
    reverses = []


    for (
        added,
        deleted,
        labels,
    ) in loader:


        added = to_device(
            added
        )

        deleted = to_device(
            deleted
        )


        with torch.amp.autocast(

            "cuda",

            enabled=(
                DEVICE.type
                == "cuda"
            ),
        ):


            orig_logit, rev_logit = (
                model(
                    added,
                    deleted,
                )
            )


        orig_prob = (
            torch.sigmoid(
                orig_logit
            )
            .float()
            .cpu()
            .numpy()
        )


        rev_prob = (
            torch.sigmoid(
                rev_logit
            )
            .float()
            .cpu()
            .numpy()
        )


        originals.extend(
            orig_prob
        )

        reverses.extend(
            rev_prob
        )

        ys.extend(
            labels.numpy()
        )


    y = np.array(
        ys,
        dtype=int,
    )

    p = np.array(
        originals
    )

    p_rev = np.array(
        reverses
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
        np.mean(

            p[
                defective
            ]

            >

            p_rev[
                defective
            ]
        ),

        "direction_margin":
        np.mean(

            p[
                defective
            ]

            -

            p_rev[
                defective
            ]
        ),

        "delta_p":
        np.mean(
            np.abs(
                p - p_rev
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
        f"{m['delta_p']:.4f}  "

        f"PredChange="
        f"{m['prediction_change']:.2%}"
    )


# ============================================================
# MAIN
# ============================================================

print(
    "device:",
    DEVICE
)

assert (
    DEVICE.type == "cuda"
), "Colab GPU runtime을 켜는 걸 추천"


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


train = prepare(
    raw_train
)

valid = prepare(
    raw_valid
)

test = prepare(
    raw_test
)


print(
    "\ntrain:",
    len(train)
)

print(
    "valid:",
    len(valid)
)

print(
    "test:",
    len(test)
)


labels = np.array(
    [
        x["label"]
        for x in train
    ]
)


n_pos = np.sum(
    labels == 1
)

n_neg = np.sum(
    labels == 0
)


pos_weight = (
    n_neg
    /
    n_pos
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

results = []


print(
    "\n================================"
)

print(
    "VALIDATION — CodeBERT"
)

print(
    "================================"
)


for lam in LAMBDAS:


    model = train_model(

        train,

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
# SELECT MODEL
# ============================================================

best_f1 = max(

    m["f1"]

    for _, _, m
    in results
)


f1_floor = (
    best_f1
    - 0.01
)


eligible = [

    x

    for x in results

    if x[2]["f1"]
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
    "SELECTED"
)

print(
    "================================"
)


print(
    "best F1:",
    round(
        best_f1,
        4,
    )
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

test_metrics = evaluate(

    selected_model,

    test,
)


baseline_model = next(

    model

    for lam, model, _
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
    "FINAL TEST"
)

print(
    "================================"
)


print(
    "BASELINE CodeBERT"
)

print_metrics(
    0.0,
    baseline_test,
)


print(
    "\nDIRECTION-AWARE CodeBERT"
)

print_metrics(
    selected_lambda,
    test_metrics,
)


print(
    "\nΔ F1:",
    f"{test_metrics['f1'] - baseline_test['f1']:+.4f}"
)


print(
    "Δ DirCons:",
    f"{test_metrics['direction_consistency'] - baseline_test['direction_consistency']:+.2%}"
)


print(
    "Δ Margin:",
    f"{test_metrics['direction_margin'] - baseline_test['direction_margin']:+.4f}"
)