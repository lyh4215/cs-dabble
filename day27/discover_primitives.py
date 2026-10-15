import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

np.random.seed(0)


# ============================================
# hidden world
#
# 모델은 이걸 모른다.
#
# state:
# [x, y, paint, rotation]
# ============================================

TRUE_PRIMITIVES = {
    "L": np.array([
        -1.0, 0.0, 0.0, 0.0
    ]),

    "D": np.array([
        0.0, -1.0, 0.0, 0.0
    ]),

    "P": np.array([
        0.0, 0.0, +1.0, 0.0
    ]),

    "R": np.array([
        0.0, 0.0, 0.0, +1.0
    ]),
}


primitive_names = list(
    TRUE_PRIMITIVES.keys()
)


# ============================================
# generate observations
#
# model sees:
# before, after
#
# hidden_labels는 평가할 때만 사용
# ============================================

transitions = []
hidden_labels = []


for _ in range(400):

    # random starting state
    before = np.random.uniform(
        -5,
        5,
        size=4,
    )


    # hidden primitive chosen by world
    name = np.random.choice(
        primitive_names
    )

    transformation = (
        TRUE_PRIMITIVES[name]
    )


    # noisy observation
    noise = np.random.normal(
        0,
        0.06,
        size=4,
    )


    after = (
        before
        + transformation
        + noise
    )


    transitions.append(
        (
            before,
            after,
        )
    )

    hidden_labels.append(
        name
    )


# ============================================
# model only computes observed change
# ============================================

deltas = np.array([
    after - before
    for before, after
    in transitions
])


# ============================================
# primitive discovery
#
# "there seem to be 4 recurring
# types of transformation"
#
# primitive names are NOT given
# ============================================

model = KMeans(
    n_clusters=4,
    random_state=0,
    n_init=20,
)

clusters = model.fit_predict(
    deltas
)


print(
    "=== DISCOVERED PRIMITIVES ==="
)


for i, center in enumerate(
    model.cluster_centers_
):

    print(
        f"cluster {i}:",
        np.round(
            center,
            3,
        ),
    )


# ============================================
# inspect what each cluster contains
#
# IMPORTANT:
# this is only evaluation.
# KMeans did NOT receive these names.
# ============================================

print(
    "\n=== CLUSTER CONTENTS ==="
)


cluster_to_name = {}


for cluster_id in range(4):

    idx = np.where(
        clusters == cluster_id
    )[0]


    names = [
        hidden_labels[i]
        for i in idx
    ]


    counts = {
        name: names.count(name)
        for name in primitive_names
    }


    majority = max(
        counts,
        key=counts.get,
    )


    cluster_to_name[
        cluster_id
    ] = majority


    purity = (
        counts[majority]
        / len(names)
    )


    print(
        f"cluster {cluster_id}:",
        counts,
        f"→ {majority}",
        f"purity={purity:.3f}",
    )


# ============================================
# label-independent clustering score
#
# true names are converted to IDs
# only for evaluation
# ============================================

name_to_id = {
    name: i
    for i, name
    in enumerate(
        primitive_names
    )
}


true_ids = np.array([
    name_to_id[x]
    for x in hidden_labels
])


ari = adjusted_rand_score(
    true_ids,
    clusters,
)


print(
    "\nARI:",
    ari,
)


# ============================================
# NEW observation
#
# Can learned codebook identify
# which primitive happened?
# ============================================

before = np.array([
    2.3,
    -1.7,
    0.5,
    3.2,
])


# world secretly applies DOWN
after = (
    before
    + TRUE_PRIMITIVES["D"]
    + np.random.normal(
        0,
        0.06,
        size=4,
    )
)


observed_delta = (
    after - before
)


cluster = model.predict(
    observed_delta.reshape(
        1,
        -1,
    )
)[0]


print(
    "\n=== NEW OBSERVATION ==="
)

print(
    "observed delta:",
    np.round(
        observed_delta,
        3,
    ),
)

print(
    "assigned cluster:",
    cluster,
)

print(
    "interpreted as:",
    cluster_to_name[
        cluster
    ],
)