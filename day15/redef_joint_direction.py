import json
import difflib
import urllib.request

import numpy as np

from scipy.sparse import (
    hstack,
    vstack,
    csr_matrix,
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.linear_model import (
    SGDClassifier,
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# ============================================================
# 1. ReDef data
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


# ============================================================
# 2. Load
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
# 3. Extract modification
#
# Returns:
#
# tagged:
#   <DEL> old
#   <ADD> new
#
# added:
#   only added code
#
# deleted:
#   only deleted code
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


    tagged = []
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

            for line in before[
                i1:i2
            ]:

                tagged.append(
                    "<DEL> " + line
                )

                deleted.append(
                    line
                )


        elif tag == "insert":

            for line in after[
                j1:j2
            ]:

                tagged.append(
                    "<ADD> " + line
                )

                added.append(
                    line
                )


        elif tag == "replace":

            for line in before[
                i1:i2
            ]:

                tagged.append(
                    "<DEL> " + line
                )

                deleted.append(
                    line
                )


            for line in after[
                j1:j2
            ]:

                tagged.append(
                    "<ADD> " + line
                )

                added.append(
                    line
                )


    return (
        "\n".join(tagged),
        "\n".join(added),
        "\n".join(deleted),
    )


# ============================================================
# 4. Reverse tagged diff
#
# Same code,
# ADD <-> DEL
# ============================================================

def reverse_tagged(
    tagged,
):

    lines = []


    for line in (
        tagged.splitlines()
    ):

        if line.startswith(
            "<ADD> "
        ):

            lines.append(
                "<DEL> "
                + line[len("<ADD> "):]
            )


        elif line.startswith(
            "<DEL> "
        ):

            lines.append(
                "<ADD> "
                + line[len("<DEL> "):]
            )


        else:

            lines.append(
                line
            )


    return "\n".join(
        lines
    )


# ============================================================
# 5. Prepare raw dataset
# ============================================================

def prepare(examples):

    tagged = []
    reverse_tagged_list = []

    added = []
    deleted = []

    labels = []


    for ex in examples:

        t, a, d = extract_change(
            ex
        )


        if not t.strip():
            continue


        tagged.append(
            t
        )

        reverse_tagged_list.append(
            reverse_tagged(t)
        )

        added.append(
            a
        )

        deleted.append(
            d
        )

        labels.append(
            int(
                ex[
                    "defective_modification"
                ]
            )
        )


    return {
        "tagged":
        tagged,

        "reverse_tagged":
        reverse_tagged_list,

        "added":
        added,

        "deleted":
        deleted,

        "y":
        np.array(
            labels,
            dtype=int,
        ),
    }


# ============================================================
# 6. Fit vectorizers
#
# Two vocabularies:
#
# T:
# tagged diff vocabulary
#
# C:
# code vocabulary shared by A and D
# ============================================================

def fit_vectorizers(
    train,
):

    tagged_vectorizer = (
        TfidfVectorizer(

            tokenizer=str.split,

            preprocessor=None,

            token_pattern=None,

            lowercase=False,

            max_features=15000,

            ngram_range=(1, 2),

            min_df=2,
        )
    )


    change_vectorizer = (
        TfidfVectorizer(

            tokenizer=str.split,

            preprocessor=None,

            token_pattern=None,

            lowercase=False,

            max_features=15000,

            ngram_range=(1, 2),

            min_df=2,
        )
    )


    tagged_vectorizer.fit(
        train[
            "tagged"
        ]
    )


    change_vectorizer.fit(

        train[
            "added"
        ]

        +

        train[
            "deleted"
        ]
    )


    return (
        tagged_vectorizer,
        change_vectorizer,
    )


# ============================================================
# 7. Build combined features
#
# Original:
#
# [T, A, D, A-D]
#
#
# Reverse:
#
# [T_rev, D, A, D-A]
# ============================================================

def transform(
    dataset,
    tagged_vectorizer,
    change_vectorizer,
):

    T = tagged_vectorizer.transform(
        dataset[
            "tagged"
        ]
    )


    T_rev = (
        tagged_vectorizer.transform(
            dataset[
                "reverse_tagged"
            ]
        )
    )


    A = change_vectorizer.transform(
        dataset[
            "added"
        ]
    )


    D = change_vectorizer.transform(
        dataset[
            "deleted"
        ]
    )


    X = hstack(

        [
            T,
            A,
            D,
            A - D,
        ],

        format="csr",
    )


    X_rev = hstack(

        [
            T_rev,
            D,
            A,
            D - A,
        ],

        format="csr",
    )


    return (
        X,
        X_rev,
    )


# ============================================================
# 8. Add explicit bias feature
#
# Why?
#
# Original classification:
#
#   score = w*x + b
#
# Pair comparison:
#
#   score(orig)-score(rev)
#       = w*(orig-rev)
#
# bias must cancel.
#
#
# So:
#
# original rows:
#   [x, 1]
#
# pair rows:
#   [orig-rev, 0]
# ============================================================

def add_bias_column(
    X,
    value,
):

    column = csr_matrix(

        np.full(
            (
                X.shape[0],
                1,
            ),
            value,
            dtype=np.float64,
        )
    )


    return hstack(

        [
            X,
            column,
        ],

        format="csr",
    )


# ============================================================
# 9. Class balancing weights
# ============================================================

def original_sample_weights(
    y,
):

    n = len(y)

    n_pos = np.sum(
        y == 1
    )

    n_neg = np.sum(
        y == 0
    )


    pos_weight = (
        n
        /
        (
            2.0
            * n_pos
        )
    )


    neg_weight = (
        n
        /
        (
            2.0
            * n_neg
        )
    )


    return np.where(

        y == 1,

        pos_weight,

        neg_weight,
    )


# ============================================================
# 10. Joint training
#
# λ = 0:
#
# defect classification only
#
#
# λ > 0:
#
# defect classification
# +
# pairwise direction training
# ============================================================

def train_joint(
    X,
    X_rev,
    y,
    lam,
):

    # --------------------------------------------------------
    # Original classification rows
    # --------------------------------------------------------

    X_orig = add_bias_column(
        X,
        1.0,
    )


    y_orig = y.copy()


    weights_orig = (
        original_sample_weights(
            y
        )
    )


    # --------------------------------------------------------
    # No pairwise loss
    # --------------------------------------------------------

    if lam == 0:

        X_train = X_orig

        y_train = y_orig

        sample_weight = (
            weights_orig
        )


    else:

        # ----------------------------------------------------
        # Only defective modifications have known direction:
        #
        # orig = defect introducing
        # reverse = undo
        # ----------------------------------------------------

        defective_idx = np.where(
            y == 1
        )[0]


        pair_diff = (

            X[
                defective_idx
            ]

            -

            X_rev[
                defective_idx
            ]
        )


        # ----------------------------------------------------
        # positive pair:
        #
        # orig - rev > 0
        # ----------------------------------------------------

        pair_positive = (
            add_bias_column(
                pair_diff,
                0.0,
            )
        )


        y_pair_positive = (
            np.ones(
                len(
                    defective_idx
                ),
                dtype=int,
            )
        )


        # ----------------------------------------------------
        # negative symmetric pair:
        #
        # rev - orig < 0
        # ----------------------------------------------------

        pair_negative = (
            add_bias_column(
                -pair_diff,
                0.0,
            )
        )


        y_pair_negative = (
            np.zeros(
                len(
                    defective_idx
                ),
                dtype=int,
            )
        )


        X_train = vstack(

            [
                X_orig,
                pair_positive,
                pair_negative,
            ],

            format="csr",
        )


        y_train = np.concatenate(

            [
                y_orig,
                y_pair_positive,
                y_pair_negative,
            ]
        )


        # each pair direction gets λ/2
        pair_weight = (
            lam
            / 2.0
        )


        sample_weight = (
            np.concatenate(

                [
                    weights_orig,

                    np.full(
                        len(
                            defective_idx
                        ),
                        pair_weight,
                    ),

                    np.full(
                        len(
                            defective_idx
                        ),
                        pair_weight,
                    ),
                ]
            )
        )


    # --------------------------------------------------------
    # Logistic linear model
    #
    # fit_intercept=False because
    # explicit bias column is used.
    # --------------------------------------------------------

    model = SGDClassifier(

        loss="log_loss",

        penalty="l2",

        alpha=1e-4,

        max_iter=3000,

        tol=1e-5,

        fit_intercept=False,

        random_state=0,

        average=True,
    )


    model.fit(

        X_train,

        y_train,

        sample_weight=sample_weight,
    )


    return model


# ============================================================
# 11. Score
# ============================================================

def model_scores(
    model,
    X,
):

    Xb = add_bias_column(
        X,
        1.0,
    )


    return model.predict_proba(
        Xb
    )[:, 1]


# ============================================================
# 12. Evaluation
# ============================================================

def evaluate(
    model,
    X,
    X_rev,
    y,
):

    p = model_scores(
        model,
        X,
    )


    p_rev = model_scores(
        model,
        X_rev,
    )


    pred = (
        p >= 0.5
    ).astype(int)


    f1 = f1_score(
        y,
        pred,
        zero_division=0,
    )


    precision = precision_score(
        y,
        pred,
        zero_division=0,
    )


    recall = recall_score(
        y,
        pred,
        zero_division=0,
    )


    accuracy = accuracy_score(
        y,
        pred,
    )


    defective = (
        y == 1
    )


    direction_consistency = (
        np.mean(

            p[
                defective
            ]

            >

            p_rev[
                defective
            ]
        )
    )


    direction_margin = (
        np.mean(

            p[
                defective
            ]

            -

            p_rev[
                defective
            ]
        )
    )


    reversal_delta = np.mean(
        np.abs(
            p - p_rev
        )
    )


    pred_rev = (
        p_rev >= 0.5
    ).astype(int)


    prediction_change = (
        np.mean(
            pred
            !=
            pred_rev
        )
    )


    return {

        "accuracy":
        accuracy,

        "precision":
        precision,

        "recall":
        recall,

        "f1":
        f1,

        "direction_consistency":
        direction_consistency,

        "direction_margin":
        direction_margin,

        "reversal_delta":
        reversal_delta,

        "prediction_change":
        prediction_change,
    }


# ============================================================
# 13. Print
# ============================================================

def print_metrics(
    lam,
    metrics,
):

    print(
        f"λ={lam:<4} "
        f"F1={metrics['f1']:.4f}  "
        f"DirCons="
        f"{metrics['direction_consistency']:.2%}  "
        f"Margin="
        f"{metrics['direction_margin']:+.4f}  "
        f"ΔP="
        f"{metrics['reversal_delta']:.4f}  "
        f"PredChange="
        f"{metrics['prediction_change']:.2%}"
    )


# ============================================================
# MAIN
# ============================================================

print(
    "================================"
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
    len(train["y"])
)

print(
    "valid:",
    len(valid["y"])
)

print(
    "test:",
    len(test["y"])
)


# ============================================================
# Vectorizers fit ONLY on train
# ============================================================

print(
    "\n================================"
)

print(
    "FIT VECTORIZERS"
)

print(
    "================================"
)


tag_vec, change_vec = (
    fit_vectorizers(
        train
    )
)


print(
    "tag vocab:",
    len(
        tag_vec.vocabulary_
    )
)


print(
    "change vocab:",
    len(
        change_vec.vocabulary_
    )
)


# ============================================================
# Features
# ============================================================

X_train, X_train_rev = (
    transform(
        train,
        tag_vec,
        change_vec,
    )
)


X_valid, X_valid_rev = (
    transform(
        valid,
        tag_vec,
        change_vec,
    )
)


X_test, X_test_rev = (
    transform(
        test,
        tag_vec,
        change_vec,
    )
)


print(
    "feature dim:",
    X_train.shape[1]
)


# ============================================================
# 14. Tune lambda on validation
# ============================================================

LAMBDAS = [
    0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]


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

    model = train_joint(
        X_train,
        X_train_rev,
        train["y"],
        lam,
    )


    metrics = evaluate(
        model,
        X_valid,
        X_valid_rev,
        valid["y"],
    )


    results.append(
        (
            lam,
            model,
            metrics,
        )
    )


    print_metrics(
        lam,
        metrics,
    )


# ============================================================
# 15. Model selection
#
# Scientific constraint:
#
# keep F1 within 0.01 of best validation F1,
# then maximize direction consistency.
# ============================================================

best_valid_f1 = max(

    m["f1"]

    for _, _, m
    in results
)


f1_floor = (
    best_valid_f1
    - 0.01
)


eligible = [

    item

    for item
    in results

    if item[2]["f1"]
    >= f1_floor
]


selected = max(

    eligible,

    key=lambda item:
    (
        item[2][
            "direction_consistency"
        ],

        item[2][
            "direction_margin"
        ],
    )
)


selected_lambda, \
selected_model, \
selected_valid_metrics = (
    selected
)


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
    "best validation F1:",
    f"{best_valid_f1:.4f}"
)


print(
    "allowed F1 floor:",
    f"{f1_floor:.4f}"
)


print(
    "selected λ:",
    selected_lambda
)


print(
    "selected valid:"
)


print_metrics(
    selected_lambda,
    selected_valid_metrics,
)


# ============================================================
# 16. Final TEST
#
# test was never used for λ selection.
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


test_metrics = evaluate(

    selected_model,

    X_test,

    X_test_rev,

    test["y"],
)


print_metrics(

    selected_lambda,

    test_metrics,
)


# ============================================================
# Baseline λ=0 test for direct comparison
# ============================================================

baseline_model = [

    model

    for lam, model, metrics
    in results

    if lam == 0

][0]


baseline_test = evaluate(

    baseline_model,

    X_test,

    X_test_rev,

    test["y"],
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
    "BASELINE λ=0"
)

print_metrics(
    0,
    baseline_test,
)


print(
    "\nSELECTED"
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
    "Δ direction consistency:",
    f"{test_metrics['direction_consistency'] - baseline_test['direction_consistency']:+.2%}"
)


print(
    "Δ direction margin:",
    f"{test_metrics['direction_margin'] - baseline_test['direction_margin']:+.4f}"
)