
import numpy as np
import exp1_visual_rrt as r

SIGMAS = [0.06, 0.12, 0.24, 0.40]
DISTANCES = [1.0, 0.75, 0.5, 0.25, 0.1]

for sigma in SIGMAS:
    r.SIGMA = sigma
    r.TARGET_IMAGE = r.render(r.GOAL)

    print(f"\n=== SIGMA {sigma} ===")
    print(f"{'distance':>10} {'grad_norm':>14} {'cosine':>10}")

    for ratio in DISTANCES:
        pos = r.GOAL + ratio * (r.START - r.GOAL)
        grad = r.image_gradient(pos)
        direction = -grad
        to_goal = r.GOAL - pos

        norm = np.linalg.norm(direction)
        goal_norm = np.linalg.norm(to_goal)

        cosine = (
            np.dot(direction, to_goal) / (norm * goal_norm)
            if norm > 1e-12
            else float("nan")
        )

        print(
            f"{goal_norm:10.4f} "
            f"{norm:14.6e} "
            f"{cosine:10.4f}"
        )
