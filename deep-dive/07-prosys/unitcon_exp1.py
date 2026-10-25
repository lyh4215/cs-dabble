
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile


ACTIONS = (
    "CREATE", "RESET", "REVOKE", "RESTORE",
    "DISABLE", "PING", "AUTHORIZE",
    "ENABLE", "NULLIFY",
)


@dataclass(frozen=True)
class State:
    exists: bool = False
    authorized: bool = False
    enabled: bool = False
    payload_null: bool = False


def step(s: State, action: str) -> State:
    if action == "CREATE":
        return State(exists=True)

    if not s.exists:
        return s

    if action == "RESET":
        return State(exists=True)

    if action == "REVOKE":
        return State(
            True, False, False, s.payload_null
        )

    if action == "RESTORE":
        return State(
            True, s.authorized, s.enabled, False
        )

    if action == "DISABLE":
        return State(
            True, s.authorized, False, s.payload_null
        )

    if action == "AUTHORIZE":
        return State(
            True, True, s.enabled, s.payload_null
        )

    if action == "ENABLE" and s.authorized:
        return State(
            True, True, True, s.payload_null
        )

    if action == "NULLIFY" and s.enabled:
        return State(True, s.authorized, True, True)

    return s


def is_target_npe(s: State) -> bool:
    return (
        s.exists
        and s.enabled
        and s.payload_null
    )


def synthesize(max_depth: int):
    queue = deque([((), State())])
    tested = 0
    by_depth = Counter()

    while queue:
        sequence, state = queue.popleft()

        tested += 1
        by_depth[len(sequence)] += 1

        if is_target_npe(state):
            return sequence, tested, by_depth

        if len(sequence) >= max_depth:
            continue

        # Baseline: pruning 없이 모든 호출 확장
        for action in ACTIONS:
            new_state = step(state, action)

            queue.append((
                sequence + (action,),
                new_state,
            ))

    return None, tested, by_depth


JAVA = r"""
public class UnitConFixture {
    static class Service {
        boolean authorized = false;
        boolean enabled = false;
        String payload = "ready";

        void reset() {
            authorized = false;
            enabled = false;
            payload = "ready";
        }

        void revoke() {
            authorized = false;
            enabled = false;
        }

        void restore() {
            payload = "ready";
        }

        void disable() {
            enabled = false;
        }

        void ping() {}

        void authorize() {
            authorized = true;
        }

        void enable() {
            if (authorized) enabled = true;
        }

        void nullify() {
            if (enabled) payload = null;
        }

        int target() {
            if (!enabled) return -1;
            return payload.length(); // TARGET_NPE
        }
    }

    public static void main(String[] args) {
        Service s = null;

        try {
            for (String op : args) {
                if (op.equals("CREATE")) {
                    s = new Service();
                    continue;
                }

                if (s == null) continue;

                switch (op) {
                    case "RESET": s.reset(); break;
                    case "REVOKE": s.revoke(); break;
                    case "RESTORE": s.restore(); break;
                    case "DISABLE": s.disable(); break;
                    case "PING": s.ping(); break;
                    case "AUTHORIZE": s.authorize(); break;
                    case "ENABLE": s.enable(); break;
                    case "NULLIFY": s.nullify(); break;
                    default:
                        throw new IllegalArgumentException(op);
                }
            }

            if (s != null) s.target();
            System.out.println("NO_TARGET_NPE");

        } catch (NullPointerException e) {
            if (
                e.getStackTrace().length > 0
                && e.getStackTrace()[0]
                    .getMethodName().equals("target")
            ) {
                System.out.println("TARGET_NPE");
            } else {
                System.out.println("WRONG_NPE");
            }
        }
    }
}
"""


def verify_java(sequence):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        (root / "UnitConFixture.java").write_text(
            JAVA, encoding="utf-8"
        )

        subprocess.run(
            ["javac", "UnitConFixture.java"],
            cwd=root, check=True,
            capture_output=True, text=True,
        )

        result = subprocess.run(
            [
                "java", "-cp", str(root),
                "UnitConFixture", *sequence,
            ],
            check=True,
            capture_output=True, text=True,
        )

        return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--skip-java", action="store_true"
    )
    args = parser.parse_args()

    sequence, tested, depths = synthesize(
        args.depth
    )

    print("Tested candidates:", tested)
    print("By depth:", dict(depths))

    if sequence is None:
        print("NO WITNESS")
        return

    print("WITNESS:", " -> ".join(sequence))

    if not args.skip_java:
        result = verify_java(sequence)
        print("JVM result:", result)
        assert result == "TARGET_NPE"

        # Negative control
        without_nullify = tuple(
            a for a in sequence
            if a != "NULLIFY"
        )

        control = verify_java(without_nullify)
        print("Control:", control)
        assert control == "NO_TARGET_NPE"


if __name__ == "__main__":
    main()
