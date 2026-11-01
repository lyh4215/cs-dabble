
import numpy as np
from exp1_visual_rrt import (
    START, GOAL, visual_loss, image_gradient
)

print(
    f"{'distance':>10} "
    f"{'loss':>12} "
    f"{'grad_norm':>14} "
    f"{'cosine_to_goal':>16}"
)

for alpha in np.linspace(0, 0.99, 20):
    pos = START * (1 - alpha) + GOAL * alpha

    grad = image_gradient(pos)
    direction = -grad
    target_direction = GOAL - pos

    norm = np.linalg.norm(direction)
    target_norm = np.linalg.norm(target_direction)

    if norm > 1e-15 and target_norm > 0:
        cosine = np.dot(
            direction, target_direction
        ) / (norm * target_norm)
    else:
        cosine = float("nan")

    print(
        f"{target_norm:10.5f} "
        f"{visual_loss(pos):12.8f} "
        f"{norm:14.8e} "
        f"{cosine:16.5f}"
    )
