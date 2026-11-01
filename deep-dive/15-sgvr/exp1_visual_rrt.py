
import math
import random
from dataclasses import dataclass

import numpy as np

START = np.array([0.10, 0.15])
GOAL = np.array([0.88, 0.85])  # Used only to construct the goal image

OBSTACLES = [
    (0.40, 0.48, 0.00, 0.40),
    (0.40, 0.48, 0.60, 1.00),
]

STEP = 0.045
BUDGET = 3000
RES = 32
SIGMA = 0.06

axis = np.linspace(0, 1, RES)
X, Y = np.meshgrid(axis, axis)


def render(pos):
    """A tiny differentiable-style Gaussian point renderer."""
    return np.exp(
        -((X - pos[0]) ** 2 + (Y - pos[1]) ** 2)
        / (2 * SIGMA ** 2)
    )


TARGET_IMAGE = render(GOAL)


def visual_loss(pos):
    image = render(pos)
    return np.mean((image - TARGET_IMAGE) ** 2)


def image_gradient(pos, h=0.005):
    grad = np.zeros(2)
    for d in range(2):
        delta = np.zeros(2)
        delta[d] = h
        grad[d] = (
            visual_loss(pos + delta)
            - visual_loss(pos - delta)
        ) / (2 * h)
    return grad


def collision(pos):
    x, y = pos

    if not (0 <= x <= 1 and 0 <= y <= 1):
        return True

    for xmin, xmax, ymin, ymax in OBSTACLES:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True

    return False


def edge_valid(a, b):
    distance = np.linalg.norm(b - a)
    n = max(2, int(math.ceil(distance / 0.005)))

    for t in np.linspace(0, 1, n + 1):
        if collision(a * (1 - t) + b * t):
            return False

    return True


@dataclass
class Node:
    pos: np.ndarray
    parent: int


def plan(strategy, seed):
    rng = random.Random(seed)
    nodes = [Node(START.copy(), -1)]

    best_loss = visual_loss(START)
    goal_index = None

    for iteration in range(BUDGET):
        use_guidance = (
            strategy in ("visual", "goal")
            and rng.random() < 0.55
        )

        # A tree must also explore away from the goal
        # to find its way around obstacles.
        if use_guidance:
            pool = [rng.randrange(len(nodes)) for _ in range(5)]
            parent_id = min(
                pool,
                key=lambda i: visual_loss(nodes[i].pos),
            )
            current = nodes[parent_id].pos
            if strategy == "visual":
                direction = -image_gradient(current)
            else:
                # Oracle goal-coordinate baseline.
                # Not a visual-only planning method.
                direction = GOAL - current
            
            if np.linalg.norm(direction) < 1e-10:
                direction = np.array([
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                ])
        else:
            target = np.array([
                rng.random(),
                rng.random(),
            ])
            parent_id = min(
                range(len(nodes)),
                key=lambda i: np.sum(
                    (nodes[i].pos - target) ** 2
                ),
            )
            current = nodes[parent_id].pos
            direction = target - current

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue

        candidate = current + STEP * direction / norm

        if not edge_valid(current, candidate):
            continue

        nodes.append(Node(candidate, parent_id))
        index = len(nodes) - 1

        loss = visual_loss(candidate)
        best_loss = min(best_loss, loss)

        # Success defined in visual space.
        # Threshold determined against target image.
        if loss < 0.00002:
            goal_index = index
            break

    path = []
    if goal_index is not None:
        index = goal_index
        while index != -1:
            path.append(nodes[index].pos)
            index = nodes[index].parent
        path.reverse()

    return {
        "success": goal_index is not None,
        "iterations": iteration + 1,
        "nodes": len(nodes),
        "best_loss": best_loss,
        "path": np.array(path),
    }


if __name__ == "__main__":
    for strategy in ("random", "goal", "visual"):
        results = [plan(strategy, s) for s in range(20)]
        successful = [r for r in results if r["success"]]

        print(f"\n=== {strategy.upper()} ===")
        print("Success:", len(successful), "/ 20")
        print("Mean nodes:", np.mean([r["nodes"] for r in results]))
        print("Mean best visual loss:",
              np.mean([r["best_loss"] for r in results]))

        if successful:
            print(
                "Mean iterations to success:",
                np.mean([r["iterations"] for r in successful])
            )
