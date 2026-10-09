import random


HEIGHT = 2
NUM_LEAVES = 2 ** HEIGHT
BUCKET_SIZE = 2

random.seed(4)


# binary heap indexing
#
#          1
#       /     \
#      2       3
#    /  \     /  \
#   4    5   6    7
#
# leaves 0,1,2,3
# correspond to nodes 4,5,6,7


def path_to_leaf(leaf):
    node = (2 ** HEIGHT) + leaf

    path = []

    while node >= 1:
        path.append(node)
        node //= 2

    return list(reversed(path))


def deepest_common_level(leaf_a, leaf_b):
    """
    두 leaf 경로가 어디까지 같은지.
    block assigned leaf_a가
    accessed path leaf_b의 어느 깊이까지 내려갈 수 있는지.
    """

    pa = path_to_leaf(leaf_a)
    pb = path_to_leaf(leaf_b)

    level = -1

    for i, (a, b) in enumerate(zip(pa, pb)):
        if a != b:
            break

        level = i

    return level


tree = {
    node: []
    for node in range(1, 2 ** (HEIGHT + 1))
}


position_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


# 처음에는 각 block을 자기 leaf bucket에 둠
tree[4] = ["A"]
tree[5] = ["B"]
tree[6] = ["C"]
tree[7] = ["D"]

stash = []


def print_state(title):
    print(f"\n=== {title} ===")

    print("position map:")
    for block, leaf in position_map.items():
        print(f"  {block} -> leaf {leaf}")

    print("tree:")
    for node in sorted(tree):
        print(f"  node {node}: {tree[node]}")

    print("stash:", stash)


def access(block):
    old_leaf = position_map[block]

    print(f"\nACCESS {block}")
    print("old leaf:", old_leaf)

    path = path_to_leaf(old_leaf)

    print("read path:", path)

    # 1. path 전체를 읽어 stash로 이동
    for node in path:
        stash.extend(tree[node])
        tree[node] = []

    print("stash after read:", stash)

    # 2. target block을 새로운 random leaf에 할당
    new_leaf = random.randrange(NUM_LEAVES)

    position_map[block] = new_leaf

    print(
        f"remap {block}:",
        old_leaf,
        "->",
        new_leaf,
    )

    # 3. 읽었던 path에 다시 block들을 가능한 깊게 밀어 넣음
    #
    # leaf 쪽부터 root 방향으로 처리
    for level in reversed(range(HEIGHT + 1)):
        node = path[level]

        candidates = []

        for b in stash:
            assigned_leaf = position_map[b]

            deepest = deepest_common_level(
                assigned_leaf,
                old_leaf,
            )

            if deepest >= level:
                candidates.append(b)

        while candidates and len(tree[node]) < BUCKET_SIZE:
            b = candidates.pop(0)

            tree[node].append(b)
            stash.remove(b)

    print("stash after eviction:", stash)


print_state("INITIAL")

access("A")

print_state("AFTER ACCESS A")