import json
import difflib
import urllib.request

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. Official ReDef dataset
# ============================================================

TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "waroad/ReDef/main/dataset/1_train.jsonl"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "waroad/ReDef/main/dataset/1_test.jsonl"
)


TRAIN_N = 5000
TEST_N = 1500


# ============================================================
# 2. Load JSONL
# ============================================================

def load_jsonl_stream(
    url,
    limit,
):

    result = []

    print(
        f"loading {limit} examples:"
    )

    print(
        url
    )

    with urllib.request.urlopen(
        url
    ) as response:

        for i, raw_line in enumerate(
            response
        ):

            if i >= limit:
                break

            js = json.loads(
                raw_line.decode(
                    "utf-8"
                )
            )

            result.append(
                js
            )

    return result


# ============================================================
# 3. Diff encoding
#
# mode:
#
# original
#     <DEL> old
#     <ADD> new
#
# reversed
#     <ADD> old
#     <DEL> new
#
# blind
#     old
#     new
#
# ============================================================

def encode_diff(
    example,
    mode,
):

    before = example[
        "function_before"
    ].splitlines()

    after = example[
        "function_after"
    ].splitlines()


    matcher = difflib.SequenceMatcher(
        None,
        before,
        after,
    )


    result = []


    def emit_deleted(line):

        if mode == "original":

            result.append(
                f"<DEL> {line}"
            )

        elif mode == "reversed":

            result.append(
                f"<ADD> {line}"
            )

        elif mode == "blind":

            result.append(
                line
            )

        else:

            raise ValueError(
                mode
            )


    def emit_added(line):

        if mode == "original":

            result.append(
                f"<ADD> {line}"
            )

        elif mode == "reversed":

            result.append(
                f"<DEL> {line}"
            )

        elif mode == "blind":

            result.append(
                line
            )

        else:

            raise ValueError(
                mode
            )


    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():


        if tag == "delete":

            for line in before[
                i1:i2
            ]:

                emit_deleted(
                    line
                )


        elif tag == "insert":

            for line in after[
                j1:j2
            ]:

                emit_added(
                    line
                )


        elif tag == "replace":

            for line in before[
                i1:i2
            ]:

                emit_deleted(
                    line
                )


            for line in after[
                j1:j2
            ]:

                emit_added(
                    line
                )


    return "\n".join(
        result
    )


# ============================================================
# 4. Dataset
# ============================================================

def make_xy(
    examples,
    mode,
):

    X = []
    y = []


    for ex in examples:

        text = encode_diff(
            ex,
            mode,
        )


        if not text.strip():
            continue


        X.append(
            text
        )


        y.append(
            int(
                ex[
                    "defective_modification"
                ]
            )
        )


    return X, y


# ============================================================
# 5. Train model
# ============================================================

def train_model(
    X_train_raw,
    y_train,
):

    vectorizer = (
        TfidfVectorizer(

            tokenizer=str.split,

            preprocessor=None,

            token_pattern=None,

            lowercase=False,

            max_features=30000,

            ngram_range=(
                1,
                2,
            ),

            min_df=2,
        )
    )


    X_train = (
        vectorizer.fit_transform(
            X_train_raw
        )
    )


    model = LogisticRegression(

        max_iter=1000,

        class_weight="balanced",

        random_state=0,
    )


    model.fit(
        X_train,
        y_train,
    )


    return (
        vectorizer,
        model,
    )


# ============================================================
# 6. Evaluation
# ============================================================

def evaluate(
    name,
    vectorizer,
    model,
    X_raw,
    y,
):

    X = vectorizer.transform(
        X_raw
    )


    pred = model.predict(
        X
    )


    prob = model.predict_proba(
        X
    )[:, 1]


    metrics = {

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
    }


    print(
        "\n================================"
    )

    print(
        name
    )

    print(
        "================================"
    )


    for key, value in (
        metrics.items()
    ):

        print(
            f"{key:10s}: "
            f"{value:.4f}"
        )


    return (
        metrics,
        pred,
        prob,
    )


# ============================================================
# 7. Compare predictions
# ============================================================

def compare_predictions(
    name,
    pred_a,
    pred_b,
    prob_a,
    prob_b,
):

    changed = np.sum(
        pred_a
        != pred_b
    )


    total = len(
        pred_a
    )


    mean_prob_change = (
        np.mean(
            np.abs(
                prob_a
                - prob_b
            )
        )
    )


    print(
        "\n--------------------------------"
    )

    print(
        name
    )

    print(
        "--------------------------------"
    )


    print(
        "prediction changed:",
        f"{changed}/{total}"
    )


    print(
        "change rate:",
        f"{changed / total:.2%}"
    )


    print(
        "mean |Δ probability|:",
        f"{mean_prob_change:.4f}"
    )


# ============================================================
# 8. Inspect learned ADD / DEL features
#
# TF-IDF logistic regression이 실제로
# ADD/DEL marker를 어느 정도 사용했는지 확인.
# ============================================================

def inspect_direction_features(
    vectorizer,
    model,
    top_k=15,
):

    names = np.array(
        vectorizer
        .get_feature_names_out()
    )


    weights = (
        model.coef_[0]
    )


    mask = np.array([

        (
            "<ADD>" in name
            or
            "<DEL>" in name
        )

        for name
        in names
    ])


    indices = np.where(
        mask
    )[0]


    if len(indices) == 0:

        print(
            "No direction features found."
        )

        return


    direction_weights = (
        weights[
            indices
        ]
    )


    order = np.argsort(
        np.abs(
            direction_weights
        )
    )[::-1]


    print(
        "\n================================"
    )

    print(
        "STRONGEST ADD/DEL FEATURES"
    )

    print(
        "================================"
    )


    for pos in order[
        :top_k
    ]:

        idx = indices[
            pos
        ]

        print(
            f"{names[idx]:40s} "
            f"weight="
            f"{weights[idx]:+.4f}"
        )


# ============================================================
# MAIN
# ============================================================

print(
    "================================"
)

print(
    "LOAD ReDef"
)

print(
    "================================"
)


train_examples = (
    load_jsonl_stream(
        TRAIN_URL,
        TRAIN_N,
    )
)


test_examples = (
    load_jsonl_stream(
        TEST_URL,
        TEST_N,
    )
)


# ============================================================
# Build representations
# ============================================================

X_train_tagged, y_train = (
    make_xy(
        train_examples,
        "original",
    )
)


X_train_blind, y_train_blind = (
    make_xy(
        train_examples,
        "blind",
    )
)


X_test_original, y_test = (
    make_xy(
        test_examples,
        "original",
    )
)


X_test_reversed, y_test_rev = (
    make_xy(
        test_examples,
        "reversed",
    )
)


X_test_blind, y_test_blind = (
    make_xy(
        test_examples,
        "blind",
    )
)


assert (
    y_train
    == y_train_blind
)


assert (
    y_test
    == y_test_rev
    == y_test_blind
)


print(
    "\ntrain examples:",
    len(
        y_train
    )
)


print(
    "test examples:",
    len(
        y_test
    )
)


print(
    "train defective ratio:",
    f"{np.mean(y_train):.2%}"
)


print(
    "test defective ratio:",
    f"{np.mean(y_test):.2%}"
)


# ============================================================
# MODEL A
#
# Direction-aware model
# ============================================================

print(
    "\n================================"
)

print(
    "TRAIN DIRECTION-AWARE MODEL"
)

print(
    "================================"
)


tag_vectorizer, tag_model = (
    train_model(
        X_train_tagged,
        y_train,
    )
)


print(
    "vocab:",
    len(
        tag_vectorizer.vocabulary_
    )
)


# ------------------------------------------------------------
# A. Normal
# ------------------------------------------------------------

normal_metrics, \
normal_pred, \
normal_prob = evaluate(

    "A. ORIGINAL TAGS",

    tag_vectorizer,

    tag_model,

    X_test_original,

    y_test,
)


# ------------------------------------------------------------
# B. ADD / DEL reversed
# ------------------------------------------------------------

reverse_metrics, \
reverse_pred, \
reverse_prob = evaluate(

    "B. REVERSED TAGS",

    tag_vectorizer,

    tag_model,

    X_test_reversed,

    y_test,
)


# ------------------------------------------------------------
# C. Tags removed at test time
#
# SAME trained model.
#
# 즉:
# "방향 정보가 없어지면
# 이 모델의 판단이 얼마나 흔들리나?"
# ------------------------------------------------------------

removed_metrics, \
removed_pred, \
removed_prob = evaluate(

    "C. TAGS REMOVED AT TEST",

    tag_vectorizer,

    tag_model,

    X_test_blind,

    y_test,
)


# ============================================================
# MODEL B
#
# Completely direction-blind model
#
# train부터 ADD / DEL 없음.
# ============================================================

print(
    "\n================================"
)

print(
    "TRAIN DIRECTION-BLIND MODEL"
)

print(
    "================================"
)


blind_vectorizer, blind_model = (
    train_model(
        X_train_blind,
        y_train_blind,
    )
)


print(
    "vocab:",
    len(
        blind_vectorizer.vocabulary_
    )
)


blind_metrics, \
blind_pred, \
blind_prob = evaluate(

    "D. BLIND TRAIN + BLIND TEST",

    blind_vectorizer,

    blind_model,

    X_test_blind,

    y_test,
)


# ============================================================
# Prediction sensitivity
# ============================================================

print(
    "\n\n================================"
)

print(
    "PREDICTION SENSITIVITY"
)

print(
    "================================"
)


compare_predictions(

    "A vs B: reverse ADD / DEL",

    normal_pred,

    reverse_pred,

    normal_prob,

    reverse_prob,
)


compare_predictions(

    "A vs C: remove direction tags",

    normal_pred,

    removed_pred,

    normal_prob,

    removed_prob,
)


# ============================================================
# Summary
# ============================================================

print(
    "\n================================"
)

print(
    "SUMMARY"
)

print(
    "================================"
)


print(
    f"A original       F1 = "
    f"{normal_metrics['f1']:.4f}"
)


print(
    f"B reversed       F1 = "
    f"{reverse_metrics['f1']:.4f}"
)


print(
    f"C tags removed   F1 = "
    f"{removed_metrics['f1']:.4f}"
)


print(
    f"D blind model    F1 = "
    f"{blind_metrics['f1']:.4f}"
)


print(
    "\nF1 drops from A:"
)


print(
    "reverse:",
    f"{reverse_metrics['f1'] - normal_metrics['f1']:+.4f}"
)


print(
    "remove tags:",
    f"{removed_metrics['f1'] - normal_metrics['f1']:+.4f}"
)


print(
    "blind train:",
    f"{blind_metrics['f1'] - normal_metrics['f1']:+.4f}"
)


# ============================================================
# Inspect learned direction-sensitive features
# ============================================================

inspect_direction_features(
    tag_vectorizer,
    tag_model,
)