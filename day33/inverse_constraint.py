
from collections import deque

H, W = 7, 9

START = (3, 0)
GOAL = (3, 8)
KEY = (1, 1)
DOOR = (3, 4)

WALLS = {
    (r, 4)
    for r in range(H)
    if r != 3
}

RULES = (
    "free",
    "key_before_door",
    "key_before_goal",
    "never_door",
)


# ==========================================
# 1. Temporal constraint checks
# ==========================================

def allowed(position, has_key, rule):

    if rule == "key_before_door":
        if position == DOOR and not has_key:
            return False

    if rule == "key_before_goal":
        if position == GOAL and not has_key:
            return False

    if rule == "never_door":
        if position == DOOR:
            return False

    return True


# ==========================================
# 2. BFS with augmented state
#
# Normal BFS:
#   state = position
#
# Here:
#   state = (position, has_key)
#
# The same position can require different
# actions depending on history.
# ==========================================

def plan(rule):

    initial = (START, START == KEY)

    queue = deque([initial])
    parent = {initial: None}

    end = None

    while queue:
        state = queue.popleft()

        position, has_key = state

        if position == GOAL:
            end = state
            break

        r, c = position

        for dr, dc in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if not (
                0 <= nr < H
                and 0 <= nc < W
            ):
                continue

            if nxt in WALLS:
                continue

            new_has_key = (
                has_key or nxt == KEY
            )

            if not allowed(
                nxt,
                new_has_key,
                rule,
            ):
                continue

            new_state = (
                nxt,
                new_has_key,
            )

            if new_state not in parent:
                parent[new_state] = state
                queue.append(new_state)

    if end is None:
        return None

    path = []

    while end is not None:
        path.append(end[0])
        end = parent[end]

    return path[::-1]


# ==========================================
# 3. Check temporal properties
# ==========================================

def properties(path):

    if path is None:
        return {}

    has_key = False
    touched_door = False
    violated_door = False

    for p in path:
        if p == KEY:
            has_key = True

        if p == DOOR:
            touched_door = True

            if not has_key:
                violated_door = True

    return {
        "key_before_door": not violated_door,
        "key_before_goal": KEY in path,
        "never_door": not touched_door,
    }


# ==========================================
# 4. Visualize paths
# ==========================================

def display(path):

    if path is None:
        print("No feasible path")
        return

    grid = [
        ["." for _ in range(W)]
        for _ in range(H)
    ]

    for r, c in WALLS:
        grid[r][c] = "#"

    for r, c in path:
        grid[r][c] = "*"

    for position, char in (
        (START, "S"),
        (GOAL, "G"),
        (KEY, "K"),
        (DOOR, "D"),
    ):
        grid[position[0]][position[1]] = char

    for row in grid:
        print(" ".join(row))


# ==========================================
# 5. Demonstration from an expert
#
# Hidden rule:
#   key_before_door
# ==========================================

expert = plan("key_before_door")

print("=== EXPERT DEMONSTRATION ===")
display(expert)

print("Steps:", len(expert) - 1)
print("Properties:", properties(expert))


# ==========================================
# 6. Baseline without learned constraint
# ==========================================

baseline = plan("free")

print("\n=== UNCONSTRAINED PLANNER ===")
display(baseline)

print("Steps:", len(baseline) - 1)
print("Properties:", properties(baseline))


# ==========================================
# 7. Candidate rule discovery
#
# A candidate is consistent with a demo
# if every demonstrated trajectory
# satisfies that candidate.
#
# This is a simplified consistency test,
# not the paper's GA-TL-Mining algorithm.
# ==========================================

print("\n=== CANDIDATE RULES ===")

expert_props = properties(expert)

candidates = []

for rule in RULES[1:]:

    consistent = expert_props[rule]

    print(
        f"{rule:20s} "
        f"consistent={consistent}"
    )

    if consistent:
        candidates.append(rule)


# ==========================================
# 8. Challenge each rule
#
# If a learned rule is insufficient,
# the shortest path satisfying it may
# still violate another property.
# ==========================================

print("\n=== CANDIDATE POLICY TEST ===")

for rule in candidates:

    path = plan(rule)

    if path is None:
        print(rule, "No feasible path")
        continue

    props = properties(path)

    print(
        f"{rule:20s}"
        f"steps={len(path)-1:2d} "
        f"key_before_door="
        f"{props['key_before_door']}"
    )


print("\nDone.")
