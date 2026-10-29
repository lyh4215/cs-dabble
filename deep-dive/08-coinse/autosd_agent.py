
#!/usr/bin/env python3
"""COINSE Exp 6 v3: Groq-powered debugging ablation on historic Black bug.

Install: pip install openai python-dotenv
Place GROQ_API_KEY and GROQ_MODEL in .env beside this script.
Runs historic repository tests; use a disposable Codespaces/workspace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

TEST = "tests.test_black.BlackTestCase.test_works_in_mono_process_only_environment"
ALLOWED_FILES = ("black.py", "tests/test_black.py")

TRACE_SCRIPT = r'''
import json
import sys
import unittest
from pathlib import Path

root = Path.cwd().resolve()
observations = []

def trace(frame, event, arg):
    if event == "call":
        path = Path(frame.f_code.co_filename).resolve()
        return trace if path == root / "black.py" else None

    if event == "exception" and len(observations) < 12:
        typ, exc, _ = arg

        if issubclass(
            typ, (OSError, RuntimeError, ValueError, TypeError)
        ):
            names = {
                "worker_count", "sources", "write_back",
                "mode", "loop", "executor"
            }

            vals = {}
            for key, val in frame.f_locals.items():
                if key in names:
                    try:
                        vals[key] = repr(val)[:160]
                    except Exception:
                        vals[key] = "<repr failed>"

            observations.append({
                "function": frame.f_code.co_name,
                "line": frame.f_lineno,
                "exception": typ.__name__,
                "message": str(exc)[:200],
                "locals": vals,
            })

    return trace


sys.settrace(trace)

try:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_black.BlackTestCase."
        "test_works_in_mono_process_only_environment"
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
finally:
    sys.settrace(None)

print("=== DEBUGGER OBSERVATIONS ===")

for observation in observations:
    print(
        "TRACE",
        json.dumps(observation, ensure_ascii=False)
    )

sys.exit(0 if result.wasSuccessful() else 1)
'''


def tool_spec(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


READ_TOOL = tool_spec(
    "read_file",
    (
        "Read source by numbered lines. "
        "Use EXACT keys path, line_start, line_end "
        "(NOT start/end)."
    ),
    {
        "path": {
            "type": "string",
            "enum": list(ALLOWED_FILES),
        },
        "line_start": {
            "type": "integer",
            "description": "First line, 1-indexed",
        },
        "line_end": {
            "type": "integer",
            "description": (
                "Last line inclusive; at most 140 lines"
            ),
        },
    },
    ["path", "line_start", "line_end"],
)

TEST_TOOL = tool_spec(
    "run_test",
    "Run the original regression test.",
    {},
    [],
)

TRACE_TOOL = tool_spec(
    "trace_test",
    (
        "Observe exception events and locals "
        "in black.py using sys.settrace."
    ),
    {},
    [],
)

PATCH_TOOL = tool_spec(
    "replace_text",
    (
        "Replace ONE exact source substring in black.py. "
        "old/new are raw Python source, NOT diff lines. "
        "Syntax checked before saving; retry on error."
    ),
    {
        "old": {"type": "string"},
        "new": {"type": "string"},
    },
    ["old", "new"],
)


def process(
    argv: list[str],
    cwd: Path,
    timeout=90,
) -> dict:
    try:
        p = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        return {
            "exit_code": p.returncode,
            "stdout": p.stdout[-4500:],
            "stderr": p.stderr[-8500:],
        }

    except subprocess.TimeoutExpired:
        return {
            "error": f"timeout after {timeout}s"
        }


class Sandbox:
    def __init__(
        self,
        path: Path,
        mode: str,
        python: str,
    ):
        self.path = path
        self.mode = mode
        self.python = python

        self.calls = 0
        self.test_runs = 0
        self.traces = 0
        self.patch_attempts = 0
        self.patch_accepted = 0

        self.log: list[dict] = []

    def call(self, name: str, args: dict) -> dict:
        self.calls += 1
        result: dict

        try:
            if name == "read_file":
                path = args["path"]

                if path not in ALLOWED_FILES:
                    result = {
                        "error": "path not allowed"
                    }

                else:
                    start = int(
                        args.get(
                            "line_start",
                            args.get("start", -1),
                        )
                    )

                    end = int(
                        args.get(
                            "line_end",
                            args.get("end", -1),
                        )
                    )

                    if not (
                        1 <= start <= end < start + 140
                    ):
                        result = {
                            "error": (
                                "invalid range: "
                                "maximum 140 lines"
                            )
                        }

                    else:
                        lines = (
                            (self.path / path)
                            .read_text()
                            .splitlines()
                        )

                        result = {
                            "total_lines": len(lines),
                            "lines": "\n".join(
                                f"{i}: {lines[i - 1]}"
                                for i in range(
                                    start,
                                    min(end, len(lines)) + 1,
                                )
                            ),
                        }

            elif name == "run_test":
                self.test_runs += 1

                result = process(
                    [
                        self.python,
                        "-m",
                        "unittest",
                        "-q",
                        TEST,
                    ],
                    self.path,
                )

            elif name == "trace_test":
                if self.mode != "debugger":
                    result = {
                        "error": (
                            "trace not available in baseline"
                        )
                    }

                else:
                    self.traces += 1

                    result = process(
                        [
                            self.python,
                            "-c",
                            TRACE_SCRIPT,
                        ],
                        self.path,
                    )

            elif name == "replace_text":
                self.patch_attempts += 1

                if (
                    self.mode == "debugger"
                    and self.traces == 0
                ):
                    result = {
                        "error": (
                            "Must collect debugger "
                            "evidence first"
                        )
                    }

                else:
                    old = args["old"]
                    new = args["new"]

                    filename = self.path / "black.py"
                    source = filename.read_text()

                    if (
                        not old
                        or old == new
                        or len(old) > 4500
                        or len(new) > 6500
                    ):
                        result = {
                            "error": (
                                "empty/oversized/no-op "
                                "replacement"
                            )
                        }

                    elif source.count(old) != 1:
                        result = {
                            "error": (
                                "Expected exactly one "
                                "old match, found "
                                f"{source.count(old)}"
                            )
                        }

                    else:
                        patched = source.replace(
                            old,
                            new,
                            1,
                        )

                        # Validate before modifying the file.
                        try:
                            compile(
                                patched,
                                str(filename),
                                "exec",
                            )

                        except (
                            SyntaxError,
                            ValueError,
                        ) as exc:
                            result = {
                                "error": (
                                    "invalid Python source; "
                                    "file unchanged"
                                ),
                                "exception": (
                                    type(exc).__name__
                                ),
                                "line": getattr(
                                    exc, "lineno", None
                                ),
                                "message": str(exc),
                                "text": getattr(
                                    exc, "text", None
                                ),
                            }

                        else:
                            filename.write_text(patched)

                            self.patch_accepted += 1

                            result = {
                                "ok": True,
                                "note": (
                                    "Syntax OK; run "
                                    "regression test to "
                                    "verify behavior"
                                ),
                            }

            else:
                result = {
                    "error": f"Unknown tool: {name}"
                }

        except Exception as exc:
            result = {
                "error": (
                    f"{type(exc).__name__}: {exc}"
                )
            }

        safe_args = (
            args
            if name != "replace_text"
            else {
                "old_chars": len(
                    args.get("old", "")
                ),
                "new_chars": len(
                    args.get("new", "")
                ),
            }
        )

        self.log.append({
            "tool": name,
            "args": safe_args,
            "result": result,
        })

        return result


def checkout(
    original: Path,
    destination: Path,
) -> None:
    if destination.exists():
        raise RuntimeError(
            "Destination already exists: "
            f"{destination}. Choose a new --out."
        )

    res = process(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(destination),
            "HEAD",
        ],
        original,
    )

    if res.get("exit_code") != 0:
        raise RuntimeError(
            f"worktree failed: {res}"
        )

    # Copy only the regression test.
    # Do not copy a fixed implementation.
    src = original / "tests/test_black.py"

    (destination / "tests/test_black.py").write_bytes(
        src.read_bytes()
    )

    (
        destination / "tests/__init__.py"
    ).touch(exist_ok=True)


def agent(
    client: OpenAI,
    model: str,
    sandbox: Sandbox,
    max_steps: int,
) -> str:
    tools = [
        READ_TOOL,
        TEST_TOOL,
        PATCH_TOOL,
    ]

    if sandbox.mode == "debugger":
        tools.append(TRACE_TOOL)

    prompt = f"""
You are an automated program repair agent for Black.

Target regression:
{TEST}

Only edit black.py using replace_text.
Never edit tests or use the historic fixed commit.

Analyze the failure, read source, form competing
hypotheses, propose a minimal patch, then run_test
and revise if the test fails.

Return a final diagnosis and evidence.

TOOLS:
read_file requires path, line_start, line_end.
Do not use start/end as parameter names.

PATCH FORMAT:
replace_text old/new must be raw Python source
WITHOUT leading diff +/- markers.

If replace_text returns a syntax error,
correct the patch before testing.

You have a limited tool budget.
Never claim success without a passing run_test.

Mode: {sandbox.mode}.
"""

    # Groq: keep conversation history on client side.
    history: list[dict] = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    if sandbox.mode == "debugger":
        # Controlled experimental intervention.
        # The debugger group always receives a trace
        # before the LLM starts patching.
        evidence = sandbox.call(
            "trace_test",
            {},
        )

        print(
            "[debugger] mandatory trace_test: "
            f"{str(evidence)[-650:]}",
            flush=True,
        )

        history.append({
            "role": "user",
            "content": (
                "Runtime debugger evidence was "
                "collected before patching. "
                "Use this evidence in your diagnosis:\n"
                + json.dumps(evidence)
            ),
        })

    report = ""

    for turn in range(max_steps):
        response = None

        for attempt in range(3):
            try:
                response = client.responses.create(
                    model=model,
                    input=history,
                    tools=tools,
                    max_output_tokens=2400,
                )

                break

            except Exception as exc:
                msg = str(exc)

                if (
                    "tool call validation failed"
                    in msg.lower()
                ):
                    print(
                        f"[{sandbox.mode}] "
                        "schema validation failed "
                        f"({attempt + 1}/3): "
                        f"{msg[:260]}",
                        flush=True,
                    )

                    history.append({
                        "role": "user",
                        "content": (
                            "Your last attempted tool call "
                            "was REJECTED by the API "
                            "before execution. "
                            "Retry using the EXACT "
                            "declared JSON parameter names. "
                            "For read_file use path, "
                            "line_start, line_end only. "
                            "Do not invent or rename "
                            "parameters."
                        ),
                    })

                else:
                    raise

        if response is None:
            return (
                report
                + "\n[INCOMPLETE: repeated "
                "provider-side schema errors]"
            )

        # No previous_response_id: resend history.
        history.extend(
            item.model_dump(exclude_none=True)
            for item in response.output
        )

        calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not calls:
            report = (
                response.output_text
                or "(no final explanation)"
            )

            print(
                f"[{sandbox.mode}] FINAL: "
                f"{report[:2200]}",
                flush=True,
            )

            return report

        for call in calls:
            try:
                args = json.loads(call.arguments)
                result = sandbox.call(
                    call.name,
                    args,
                )

            except Exception as exc:
                result = {
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    )
                }

            print(
                f"[{sandbox.mode}] {call.name}: "
                f"{str(result)[:330]}",
                flush=True,
            )

            history.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            })

    return (
        report
        + "\n[INCOMPLETE: agent step limit reached]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("coinse_exp5/black-bug1"),
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=Path("coinse_exp6_groq_v3"),
    )

    ap.add_argument(
        "--model",
        default=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
    )

    ap.add_argument(
        "--python",
        default=sys.executable,
    )

    ap.add_argument(
        "--steps",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--mode",
        choices=[
            "baseline",
            "debugger",
            "both",
        ],
        default="both",
    )

    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        ap.error(
            "Set GROQ_API_KEY in the .env "
            "beside this script"
        )

    root = args.repo.resolve()
    out = args.out.resolve()

    if not (root / "black.py").is_file():
        ap.error(
            f"Black source missing at {root}"
        )

    if TEST.split(".")[-1] not in (
        root / "tests/test_black.py"
    ).read_text():
        ap.error(
            "Regression test missing "
            "from tests/test_black.py"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=(
            "https://api.groq.com/openai/v1"
        ),
    )

    modes = (
        ["baseline", "debugger"]
        if args.mode == "both"
        else [args.mode]
    )

    for mode in modes:
        destination = out / mode

        checkout(root, destination)

        sandbox = Sandbox(
            destination,
            mode,
            args.python,
        )

        start = time.monotonic()

        initial = sandbox.call(
            "run_test",
            {},
        )

        print(
            f"[{mode}] initial exit_code: "
            f"{initial.get('exit_code')}",
            flush=True,
        )

        try:
            report = agent(
                client,
                args.model,
                sandbox,
                args.steps,
            )

            error = None

        except Exception as exc:
            traceback.print_exc()

            report = ""
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        final = sandbox.call(
            "run_test",
            {},
        )

        diff = process(
            [
                "git",
                "diff",
                "--",
                "black.py",
            ],
            destination,
        )

        metrics = {
            "mode": mode,
            "model": args.model,
            "initial_exit": initial.get(
                "exit_code"
            ),
            "final_exit": final.get(
                "exit_code"
            ),
            "seconds": round(
                time.monotonic() - start,
                2,
            ),
            "tool_calls": sandbox.calls,
            "test_runs": sandbox.test_runs,
            "traces": sandbox.traces,
            "patch_attempts": (
                sandbox.patch_attempts
            ),
            "patch_accepted": (
                sandbox.patch_accepted
            ),
            "final_report": report,
            "error": error,
            "final_test": final,
            "patch_diff": diff.get(
                "stdout", ""
            ),
            "tool_log": sandbox.log,
        }

        (
            out / f"{mode}.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            f"[{mode}] "
            f"final={metrics['final_exit']} "
            f"tests={sandbox.test_runs} "
            f"traces={sandbox.traces} "
            "accepted_patches="
            f"{sandbox.patch_accepted}",
            flush=True,
        )

    print("Results:", out)


if __name__ == "__main__":
    main()
