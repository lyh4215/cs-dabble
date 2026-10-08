import json
import difflib
import urllib.request

import numpy as np

from scipy.sparse import (
    hstack,
    vstack,
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. ReDef dataset
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
# 2. Load
# ============================================================

def load_jsonl(
    url,
    limit,
):

    result = []

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

            result.append(
                json.loads(
                    raw.decode("utf-8")
                )
            )

    return result


# ============================================================
# 3. Extract ADD / DELETE separately
#
# 핵심:
#
# 기존:
#
#   "<ADD> foo <DEL> bar"
#
# 이번:
#
#   added   = "foo"
#   deleted = "bar"
#
# ============================================================

def extract_change(example):

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
# 4. Build dataset
# ============================================================

def make_dataset(examples):

    added = []
    deleted = []
    labels = []


    for ex in examples:

        a, d = extract_change(
            ex
        )


        if (
            not a.strip()
            and not d.strip()
        ):
            continue


        added.append(a)
        deleted.append(d)

        labels.append(
            int(
                ex[
                    "defective_modification"
                ]
            )
        )


    return (
        added,
        deleted,
        np.array(labels),
    )


# ============================================================
# 5. Directional representation
#
# X =
#
# [
#   A,
#   D,
#   A-D
# ]
#
#
# reverse:
#
# [
#   D,
#   A,
#   D-A
# ]
# ============================================================

def directional_features(
    A,
    D,
):

    return hstack(
        [
            A,
            D,
            A - D,
        ],
        format="csr",
    )


# ============================================================
# 6. Metrics
# ============================================================

def evaluate(
    name,
    model,
    X,
    y,
):

    pred = model.predict(
        X
    )

    prob = model.predict_proba(
        X
    )[:, 1]


    print(
        "\n================================"
    )

    print(name)

    print(
        "================================"
    )


    print(
        "accuracy :",
        f"{accuracy_score(y, pred):.4f}"
    )

    print(
        "precision:",
        f"{precision_score(y, pred, zero_division=0):.4f}"
    )

    print(
        "recall   :",
        f"{recall_score(y, pred, zero_division=0):.4f}"
    )

    print(
        "f1       :",
        f"{f1_score(y, pred, zero_division=0):.4f}"
    )


    return pred, prob


# ============================================================
# 7. Direction probe
#
# defective examples에 대해:
#
# score(original)
#       >
# score(reverse)
#
# 이길 기대한다.
#
# 왜?
#
# original:
# defect-introducing modification
#
# reverse:
# 그 modification을 undo
# ============================================================

def direction_probe(
    name,
    model,
    X_original,
    X_reverse,
    y,
):

    p_original = (
        model.predict_proba(
            X_original
        )[:, 1]
    )

    p_reverse = (
        model.predict_proba(
            X_reverse
        )[:, 1]
    )


    defective = (
        y == 1
    )


    orig_def = (
        p_original[
            defective
        ]
    )

    rev_def = (
        p_reverse[
            defective
        ]
    )


    consistency = np.mean(
        orig_def
        >
        rev_def
    )


    mean_margin = np.mean(
        orig_def
        -
        rev_def
    )


    print(
        "\n================================"
    )

    print(
        name
    )

    print(
        "================================"
    )


    print(
        "defective examples:",
        len(orig_def)
    )


    print(
        "P(original) > P(reverse):",
        f"{consistency:.2%}"
    )


    print(
        "mean directional margin:",
        f"{mean_margin:+.4f}"
    )


    print(
        "mean P(original):",
        f"{np.mean(orig_def):.4f}"
    )


    print(
        "mean P(reverse): ",
        f"{np.mean(rev_def):.4f}"
    )


    return (
        consistency,
        mean_margin,
    )


# ============================================================
# MAIN
# ============================================================

print(
    "================================"
)

print(
    "LOAD DATA"
)

print(
    "================================"
)


train_examples = load_jsonl(
    TRAIN_URL,
    TRAIN_N,
)

test_examples = load_jsonl(
    TEST_URL,
    TEST_N,
)


train_added, \
train_deleted, \
y_train = make_dataset(
    train_examples
)


test_added, \
test_deleted, \
y_test = make_dataset(
    test_examples
)


print(
    "\ntrain:",
    len(y_train)
)

print(
    "test:",
    len(y_test)
)

print(
    "train defective:",
    f"{np.mean(y_train):.2%}"
)

print(
    "test defective:",
    f"{np.mean(y_test):.2%}"
)


# ============================================================
# 8. TF-IDF vocabulary
#
# ADD / DEL에 동일 vocabulary 사용
#
# 그래야 A-D 계산이 가능.
# ============================================================

vectorizer = TfidfVectorizer(

    tokenizer=str.split,

    preprocessor=None,

    token_pattern=None,

    lowercase=False,

    max_features=15000,

    ngram_range=(
        1,
        2,
    ),

    min_df=2,
)


print(
    "\n================================"
)

print(
    "FIT TF-IDF"
)

print(
    "================================"
)


vectorizer.fit(
    train_added
    +
    train_deleted
)


print(
    "vocab:",
    len(
        vectorizer.vocabulary_
    )
)


# ============================================================
# 9. Vectorize
# ============================================================

A_train = vectorizer.transform(
    train_added
)

D_train = vectorizer.transform(
    train_deleted
)


A_test = vectorizer.transform(
    test_added
)

D_test = vectorizer.transform(
    test_deleted
)


# ============================================================
# Original direction
# ============================================================

X_train = directional_features(
    A_train,
    D_train,
)


X_test = directional_features(
    A_test,
    D_test,
)


# ============================================================
# Reversed direction
#
# Added ↔ Deleted
# ============================================================

X_test_reverse = (
    directional_features(
        D_test,
        A_test,
    )
)


# ============================================================
# 10. MODEL 1
#
# Directional representation만 사용.
#
# counterfactual supervision 없음.
# ============================================================

baseline = LogisticRegression(

    max_iter=1500,

    class_weight="balanced",

    random_state=0,
)


baseline.fit(
    X_train,
    y_train,
)


baseline_pred, \
baseline_prob = evaluate(

    "MODEL 1 — DIRECTIONAL REPRESENTATION",

    baseline,

    X_test,

    y_test,
)


base_consistency, \
base_margin = direction_probe(

    "MODEL 1 — DIRECTION PROBE",

    baseline,

    X_test,

    X_test_reverse,

    y_test,
)


# ============================================================
# 11. MODEL 2
#
# Counterfactual supervision
#
# defective training sample:
#
#       A,D        label=1
#
# reversed:
#
#       D,A        label=0
#
#
# IMPORTANT:
#
# clean modification의 reverse는
# defective라고 단정할 수 없으므로
# clean samples에는 synthetic reverse label을 만들지 않는다.
# ============================================================

defective_idx = np.where(
    y_train == 1
)[0]


A_def = A_train[
    defective_idx
]

D_def = D_train[
    defective_idx
]


X_reverse_def = (
    directional_features(
        D_def,
        A_def,
    )
)


y_reverse_def = np.zeros(
    len(defective_idx),
    dtype=int,
)


X_augmented = vstack(

    [
        X_train,
        X_reverse_def,
    ],

    format="csr",
)


y_augmented = np.concatenate(

    [
        y_train,
        y_reverse_def,
    ]
)


print(
    "\n================================"
)

print(
    "COUNTERFACTUAL AUGMENTATION"
)

print(
    "================================"
)


print(
    "original train:",
    len(y_train)
)

print(
    "reversed defective negatives:",
    len(
        y_reverse_def
    )
)

print(
    "augmented train:",
    len(
        y_augmented
    )
)


cf_model = LogisticRegression(

    max_iter=1500,

    class_weight="balanced",

    random_state=0,
)


cf_model.fit(
    X_augmented,
    y_augmented,
)


cf_pred, \
cf_prob = evaluate(

    "MODEL 2 — COUNTERFACTUAL TRAINING",

    cf_model,

    X_test,

    y_test,
)


cf_consistency, \
cf_margin = direction_probe(

    "MODEL 2 — DIRECTION PROBE",

    cf_model,

    X_test,

    X_test_reverse,

    y_test,
)


# ============================================================
# 12. How much does reversal change prediction?
# ============================================================

def reversal_sensitivity(
    name,
    model,
):

    pred_original = model.predict(
        X_test
    )

    pred_reverse = model.predict(
        X_test_reverse
    )


    prob_original = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prob_reverse = (
        model.predict_proba(
            X_test_reverse
        )[:, 1]
    )


    changed = np.mean(
        pred_original
        !=
        pred_reverse
    )


    delta = np.mean(
        np.abs(
            prob_original
            -
            prob_reverse
        )
    )


    print(
        "\n",
        name
    )


    print(
        "prediction change rate:",
        f"{changed:.2%}"
    )


    print(
        "mean |Δ probability|:",
        f"{delta:.4f}"
    )


print(
    "\n================================"
)

print(
    "REVERSAL SENSITIVITY"
)

print(
    "================================"
)


reversal_sensitivity(
    "MODEL 1",
    baseline,
)


reversal_sensitivity(
    "MODEL 2",
    cf_model,
)


# ============================================================
# 13. Final comparison
# ============================================================

base_f1 = f1_score(
    y_test,
    baseline_pred,
)


cf_f1 = f1_score(
    y_test,
    cf_pred,
)


print(
    "\n================================"
)

print(
    "FINAL COMPARISON"
)

print(
    "================================"
)


print(
    "standard F1:"
)

print(
    "  Model 1:",
    f"{base_f1:.4f}"
)

print(
    "  Model 2:",
    f"{cf_f1:.4f}"
)


print(
    "\ndirectional consistency:"
)

print(
    "  Model 1:",
    f"{base_consistency:.2%}"
)

print(
    "  Model 2:",
    f"{cf_consistency:.2%}"
)


print(
    "\ndirectional margin:"
)

print(
    "  Model 1:",
    f"{base_margin:+.4f}"
)

print(
    "  Model 2:",
    f"{cf_margin:+.4f}"
)