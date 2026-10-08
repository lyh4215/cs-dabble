import json
import difflib
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Official ReDef dataset
#
# 전체 파일을 저장하지 않고 GitHub raw stream에서
# 앞쪽 N개 example만 읽는다.
# ============================================================

TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "waroad/ReDef/main/dataset/1_train.jsonl"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "waroad/ReDef/main/dataset/1_test.jsonl"
)


# Codespaces CPU에서도 금방 돌게 축소
TRAIN_N = 5000
TEST_N = 1500


# ============================================================
# Read only first N JSONL records
# ============================================================

def load_jsonl_stream(url, limit):

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

            line = raw_line.decode(
                "utf-8"
            )

            js = json.loads(
                line
            )

            result.append(
                js
            )

    return result


# ============================================================
# Diff encoding
#
# Official ReDef의 Diff_with_tags 아이디어와 동일:
#
# deleted:
#
#     <DEL> old code
#
# added:
#
#     <ADD> new code
#
# replace:
#
#     <DEL> old
#     <ADD> new
# ============================================================

def encode_diff(
    example,
    reverse_tags=False,
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


    lines = []


    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():


        # ----------------------------------------------------
        # helper:
        #
        # counterfactual에서는 ADD / DEL 의미만 뒤집는다.
        # code text는 그대로 둔다.
        # ----------------------------------------------------

        def del_tag():

            if reverse_tags:
                return "<ADD>"

            return "<DEL>"


        def add_tag():

            if reverse_tags:
                return "<DEL>"

            return "<ADD>"


        # ----------------------------------------------------

        if tag == "delete":

            for line in before[
                i1:i2
            ]:

                lines.append(
                    f"{del_tag()} {line}"
                )


        elif tag == "insert":

            for line in after[
                j1:j2
            ]:

                lines.append(
                    f"{add_tag()} {line}"
                )


        elif tag == "replace":

            for line in before[
                i1:i2
            ]:

                lines.append(
                    f"{del_tag()} {line}"
                )


            for line in after[
                j1:j2
            ]:

                lines.append(
                    f"{add_tag()} {line}"
                )


    return "\n".join(
        lines
    )


# ============================================================
# Build dataset
# ============================================================

def make_xy(
    examples,
    reverse_tags=False,
):

    X = []

    y = []


    for ex in examples:

        text = encode_diff(
            ex,
            reverse_tags=reverse_tags,
        )

        # empty diff는 제외
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
# Metrics
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


    return metrics, pred


# ============================================================
# MAIN
# ============================================================

print(
    "================================"
)

print(
    "LOAD REAL ReDef DATA"
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


print(
    "\ntrain:",
    len(train_examples)
)

print(
    "test:",
    len(test_examples)
)


# ============================================================
# Normal training data
# ============================================================

X_train_raw, y_train = (
    make_xy(
        train_examples,
        reverse_tags=False,
    )
)


# ============================================================
# Test A:
#
# normal diff
# ============================================================

X_test_normal_raw, y_test = (
    make_xy(
        test_examples,
        reverse_tags=False,
    )
)


# ============================================================
# Test B:
#
# SAME examples,
# but ADD / DEL tags reversed.
#
# labels remain unchanged.
#
# 중요:
#
# model은 counterfactual version으로 재학습하지 않는다.
# original training 그대로 사용한다.
# ============================================================

X_test_reverse_raw, y_reverse = (
    make_xy(
        test_examples,
        reverse_tags=True,
    )
)


assert (
    y_test
    == y_reverse
)


# ============================================================
# TF-IDF
#
# char n-gram을 쓰지 않고 token-based representation.
#
# <ADD>, <DEL>도 token feature에 들어가게
# custom tokenizer는 쓰지 않고 간단히 whitespace split.
# ============================================================

vectorizer = TfidfVectorizer(

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


print(
    "\n================================"
)

print(
    "FIT TF-IDF"
)

print(
    "================================"
)


X_train = (
    vectorizer.fit_transform(
        X_train_raw
    )
)


X_test_normal = (
    vectorizer.transform(
        X_test_normal_raw
    )
)


X_test_reverse = (
    vectorizer.transform(
        X_test_reverse_raw
    )
)


print(
    "vocab size:",
    len(
        vectorizer.vocabulary_
    )
)


# ============================================================
# Logistic Regression
#
# class imbalance가 있으므로 balanced 사용.
# ============================================================

model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    random_state=0,
)


print(
    "\n================================"
)

print(
    "TRAIN CLASSIFIER"
)

print(
    "================================"
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# Original test
# ============================================================

normal_metrics, normal_pred = (
    evaluate(

        "ORIGINAL DIFF",

        model,

        X_test_normal,

        y_test,
    )
)


# ============================================================
# Counterfactual test
# ============================================================

reverse_metrics, reverse_pred = (
    evaluate(

        "REVERSED ADD/DEL TAGS",

        model,

        X_test_reverse,

        y_test,
    )
)


# ============================================================
# Compare
# ============================================================

print(
    "\n================================"
)

print(
    "COUNTERFACTUAL EFFECT"
)

print(
    "================================"
)


delta_f1 = (

    reverse_metrics["f1"]

    -

    normal_metrics["f1"]
)


changed_predictions = sum(

    a != b

    for a, b

    in zip(
        normal_pred,
        reverse_pred,
    )
)


change_rate = (

    changed_predictions

    / len(normal_pred)
)


print(
    "original F1:",
    round(
        normal_metrics["f1"],
        4,
    )
)


print(
    "reversed F1:",
    round(
        reverse_metrics["f1"],
        4,
    )
)


print(
    "delta F1:",
    round(
        delta_f1,
        4,
    )
)


print(
    "prediction changed:",
    f"{changed_predictions}"
    f"/{len(normal_pred)}"
)


print(
    "prediction change rate:",
    f"{change_rate:.2%}"
)


# ============================================================
# Inspect individual examples
#
# 원본/뒤집은 diff에서 prediction이 같은 예시 몇 개 출력.
# ============================================================

same_indices = [

    i

    for i, (
        a,
        b,
    )

    in enumerate(
        zip(
            normal_pred,
            reverse_pred,
        )
    )

    if a == b
]


print(
    "\n================================"
)

print(
    "SAME-PREDICTION EXAMPLES"
)

print(
    "================================"
)


for idx in same_indices[:3]:

    print(
        f"\n--- example {idx} ---"
    )

    print(
        "label:",
        y_test[idx]
    )

    print(
        "prediction:",
        normal_pred[idx]
    )


    print(
        "\nORIGINAL:"
    )

    print(
        X_test_normal_raw[idx][
            :1200
        ]
    )


    print(
        "\nREVERSED TAGS:"
    )

    print(
        X_test_reverse_raw[idx][
            :1200
        ]
    )