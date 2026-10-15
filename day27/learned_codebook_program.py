import numpy as np

from sklearn.cluster import KMeans
from itertools import product

np.random.seed(0)


# ==================================================
# Hidden world primitives
#
# learner는 아래의 이름을 모른다고 생각한다.
#
# homogeneous affine matrix
# ==================================================

TRUE = {

    "T": np.array([
        [1., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]),

    "U": np.array([
        [1., 0., 0.],
        [0., 1., 1.],
        [0., 0., 1.],
    ]),

    "R": np.array([
        [0., -1., 0.],
        [1.,  0., 0.],
        [0.,  0., 1.],
    ]),
}


# ==================================================
# apply affine transformation to a scene
#
# scene shape = [N, 2]
# ==================================================

def apply_transform(
    scene,
    H,
):

    ones = np.ones(
        (len(scene), 1)
    )

    X = np.concatenate(
        [scene, ones],
        axis=1,
    )

    Y = (
        H
        @ X.T
    ).T

    return Y[:, :2]


# ==================================================
# execute a program
# ==================================================

def execute_true(
    scene,
    program,
):

    for name in program:

        scene = apply_transform(
            scene,
            TRUE[name],
        )

    return scene


# ==================================================
# infer affine matrix from before/after
#
# learner only gets point correspondences.
# ==================================================

def estimate_affine(
    before,
    after,
):

    ones = np.ones(
        (len(before), 1)
    )

    X = np.concatenate(
        [before, ones],
        axis=1,
    )

    # X @ B ≈ after
    B, _, _, _ = np.linalg.lstsq(
        X,
        after,
        rcond=None,
    )

    H = np.array([
        [
            B[0, 0],
            B[1, 0],
            B[2, 0],
        ],

        [
            B[0, 1],
            B[1, 1],
            B[2, 1],
        ],

        [
            0.,
            0.,
            1.,
        ],
    ])

    return H


# ==================================================
# 1. Generate raw one-step observations
#
# labels are hidden from learner.
# ==================================================

estimated_H = []
hidden_labels = []

names = list(
    TRUE.keys()
)


for _ in range(300):

    before = np.random.uniform(
        -3,
        3,
        size=(10, 2),
    )

    hidden = np.random.choice(
        names
    )

    after = apply_transform(
        before,
        TRUE[hidden],
    )

    # noisy observation
    after += np.random.normal(
        0,
        0.03,
        size=after.shape,
    )

    H_hat = estimate_affine(
        before,
        after,
    )

    estimated_H.append(
        H_hat
    )

    # ONLY FOR EVALUATION
    hidden_labels.append(
        hidden
    )


estimated_H = np.array(
    estimated_H
)


# ==================================================
# 2. Discover primitive codebook
#
# learner sees flattened matrices only.
# ==================================================

features = estimated_H[
    :, :2, :
].reshape(
    len(estimated_H),
    -1,
)


kmeans = KMeans(
    n_clusters=3,
    n_init=20,
    random_state=0,
)

cluster_ids = kmeans.fit_predict(
    features
)


# ==================================================
# convert cluster centers back to affine matrices
# ==================================================

codebook = []


for center in kmeans.cluster_centers_:

    A = center.reshape(
        2,
        3,
    )

    H = np.vstack([
        A,
        [0., 0., 1.],
    ])

    codebook.append(
        H
    )


print(
    "=== LEARNED CODEBOOK ==="
)


for i, H in enumerate(
    codebook
):

    print(
        f"\nprimitive #{i}"
    )

    print(
        np.round(
            H,
            3,
        )
    )


# ==================================================
# 3. Interpret clusters
#
# ONLY evaluation:
# learner itself does not know T/U/R.
# ==================================================

cluster_to_name = {}


print(
    "\n=== CLUSTER CONTENTS ==="
)


for c in range(3):

    indices = np.where(
        cluster_ids == c
    )[0]

    labels = [
        hidden_labels[i]
        for i in indices
    ]

    counts = {
        name: labels.count(name)
        for name in names
    }

    majority = max(
        counts,
        key=counts.get,
    )

    cluster_to_name[c] = (
        majority
    )

    print(
        f"primitive #{c}:",
        counts,
        "=>",
        majority,
    )


# ==================================================
# execute LEARNED primitives
#
# program = sequence of cluster IDs
# ==================================================

def execute_learned(
    scene,
    program,
):

    for primitive_id in program:

        scene = apply_transform(
            scene,
            codebook[
                primitive_id
            ],
        )

    return scene


# ==================================================
# reconstruction error
# ==================================================

def mse(
    a,
    b,
):

    return np.mean(
        (a - b) ** 2
    )


# ==================================================
# 4. Infer latent program
#
# before/after만 보고
# codebook primitive sequence를 search.
# ==================================================

def infer_program(
    before,
    after,
    max_len=3,
):

    best_program = None
    best_error = float(
        "inf"
    )


    for length in range(
        1,
        max_len + 1,
    ):

        for program in product(
            range(len(codebook)),
            repeat=length,
        ):

            predicted = execute_learned(
                before,
                program,
            )

            error = mse(
                predicted,
                after,
            )


            if error < best_error:

                best_error = error
                best_program = program


    return (
        best_program,
        best_error,
    )


# ==================================================
# 5. Unseen compositions
#
# learner NEVER observed these compositions
# during primitive discovery.
# ==================================================

tests = [

    ("R", "T"),

    ("T", "R"),

    ("U", "R", "T"),

]


print(
    "\n=== UNSEEN COMPOSITION TEST ==="
)


for hidden_program in tests:

    before = np.random.uniform(
        -2,
        2,
        size=(12, 2),
    )


    after = execute_true(
        before,
        hidden_program,
    )


    # small observation noise
    after += np.random.normal(
        0,
        0.01,
        size=after.shape,
    )


    inferred_ids, error = (
        infer_program(
            before,
            after,
            max_len=3,
        )
    )


    inferred_names = tuple(
        cluster_to_name[i]
        for i in inferred_ids
    )


    print(
        "\nhidden:",
        " -> ".join(
            hidden_program
        ),
    )

    print(
        "inferred IDs:",
        inferred_ids,
    )

    print(
        "interpreted:",
        " -> ".join(
            inferred_names
        ),
    )

    print(
        "reconstruction error:",
        error,
    )


# ==================================================
# 6. Show that order matters
# ==================================================

scene = np.array([
    [1., 0.],
    [0., 1.],
])


RT = execute_true(
    scene,
    ("R", "T"),
)

TR = execute_true(
    scene,
    ("T", "R"),
)


print(
    "\n=== ORDER MATTERS ==="
)

print(
    "R -> T:"
)

print(
    np.round(
        RT,
        2,
    )
)


print(
    "\nT -> R:"
)

print(
    np.round(
        TR,
        2,
    )
)


print(
    "\nsame?",
    np.allclose(
        RT,
        TR,
    ),
)