import argparse
import hashlib
import json
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from urllib.request import (
    Request,
    urlopen,
)


BASE = "http://127.0.0.1:8088"


def get_json(path):
    with urlopen(
        BASE + path,
        timeout=10,
    ) as response:
        return json.load(response)


def post_json(
    path,
    data,
):
    body = json.dumps(
        data
    ).encode()

    request = Request(
        BASE + path,
        data=body,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=20,
    ) as response:
        return json.load(response)


def has_leading_zero_bits(
    digest,
    bits,
):
    value = int.from_bytes(
        digest,
        "big",
    )

    return (
        value >> (256 - bits)
    ) == 0


def solve(
    challenge,
    difficulty,
):
    prefix = (
        challenge.encode()
        + b":"
    )

    nonce = 0

    while True:
        digest = hashlib.sha256(
            prefix
            + str(nonce).encode()
        ).digest()

        if has_leading_zero_bits(
            digest,
            difficulty,
        ):
            return nonce

        nonce += 1


def prepare_solution():
    challenge_data = get_json(
        "/challenge"
    )

    challenge = (
        challenge_data["challenge"]
    )

    difficulty = (
        challenge_data["difficulty"]
    )

    nonce = solve(
        challenge,
        difficulty,
    )

    return {
        "challenge": challenge,
        "nonce": nonce,
        "difficulty": difficulty,
    }


def submit(solution):
    return post_json(
        "/submit",
        {
            "challenge":
                solution["challenge"],
            "nonce":
                solution["nonce"],
        },
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--burst",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--trigger",
        type=float,
        default=0.8,
        help=(
            "submit when this many "
            "seconds remain in round"
        ),
    )

    args = parser.parse_args()

    print(
        f"Preparing "
        f"{args.burst} puzzle solutions..."
    )

    solutions = []

    start = time.monotonic()

    for i in range(args.burst):
        solution = (
            prepare_solution()
        )

        solutions.append(solution)

        if (
            (i + 1) % 8 == 0
            or i + 1 == args.burst
        ):
            print(
                f"prepared "
                f"{i + 1}/{args.burst}"
            )

    elapsed = (
        time.monotonic()
        - start
    )

    difficulties = sorted(
        set(
            x["difficulty"]
            for x in solutions
        )
    )

    print()
    print(
        f"prepared in "
        f"{elapsed:.3f}s"
    )

    print(
        f"puzzle difficulties: "
        f"{difficulties}"
    )

    print()
    print(
        "Waiting for end of round..."
    )

    while True:
        stats = get_json(
            "/stats"
        )

        remaining = stats[
            "round_ends_in"
        ]

        print(
            f"\rround={stats['round']} "
            f"difficulty="
            f"{stats['difficulty']} "
            f"ends_in="
            f"{remaining:.2f}s",
            end="",
            flush=True,
        )

        if remaining <= (
            args.trigger
        ):
            break

        time.sleep(0.05)

    print()
    print()
    print(
        "Submitting local burst..."
    )

    burst_start = (
        time.monotonic()
    )

    success = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:

        futures = [
            pool.submit(
                submit,
                solution,
            )
            for solution
            in solutions
        ]

        for future in (
            as_completed(futures)
        ):
            try:
                future.result()
                success += 1

            except Exception:
                failed += 1

    burst_elapsed = (
        time.monotonic()
        - burst_start
    )

    print(
        f"success={success} "
        f"failed={failed}"
    )

    print(
        f"burst completed in "
        f"{burst_elapsed:.3f}s"
    )

    print()
    print(
        "Current service state:"
    )

    print(
        json.dumps(
            get_json("/stats"),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()