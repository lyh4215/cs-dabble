
import json
import itertools
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from playwright.sync_api import sync_playwright


POLICIES = {
    "deny": "default-src 'none'; frame-src 'self'",
    "self": "default-src 'none'; frame-src 'self'; script-src 'self'",
    "inline": (
        "default-src 'none'; frame-src 'self'; "
        "script-src 'unsafe-inline'"
    ),
    "both": (
        "default-src 'none'; frame-src 'self'; "
        "script-src 'self' 'unsafe-inline'"
    ),
    "elem_only": (
        "default-src 'none'; frame-src 'self'; "
        "script-src 'none'; "
        "script-src-elem 'self' 'unsafe-inline'"
    ),
}

CONTEXTS = ("top", "srcdoc", "iframe_url")
METHODS = (
    "inline",
    "external",
    "handler",
    "javascript_url",
    "dynamic_inline",
)


@dataclass(frozen=True)
class Case:
    policy: str
    context: str
    method: str


CASES = [
    Case(*parts)
    for parts in itertools.product(
        POLICIES, CONTEXTS, METHODS
    )
]


def make_content(method):
    marker = f"window.__hits.push('{method}')"

    if method == "inline":
        return f"<script>{marker};</script>"

    if method == "external":
        return '<script src="/marker.js"></script>'

    if method == "handler":
        return (
            '<button id="trigger" '
            f'onclick="{marker}">Run</button>'
        )

    if method == "javascript_url":
        return (
            '<a id="trigger" '
            f'href="javascript:void({marker})">Run</a>'
        )

    if method == "dynamic_inline":
        # Insertion is triggered by our test driver
        # after the page has loaded.
        return '<div id="injection-point"></div>'

    raise ValueError(method)


def page_html(case):
    content = make_content(case.method)

    if case.context == "top":
        body = content

    elif case.context == "srcdoc":
        # The child document inherits the parent's CSP.
        child = (
            "<!doctype html><html><body>"
            f"{content}</body></html>"
        )
        import html
        body = (
            '<iframe name="child" '
            f'srcdoc="{html.escape(child, quote=True)}">'
            "</iframe>"
        )

    elif case.context == "iframe_url":
        query = urlencode({
            "policy": case.policy,
            "method": case.method,
        })
        body = (
            '<iframe name="child" '
            f'src="/child?{query}"></iframe>'
        )

    else:
        raise ValueError(case.context)

    return (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if url.path == "/marker.js":
            body = b"window.__hits.push('external');"
            csp = None
            kind = "application/javascript"

        elif url.path in ("/test", "/child"):
            q = parse_qs(url.query)
            pid = q.get("policy", [""])[0]
            method = q.get("method", [""])[0]
            context = q.get("context", ["top"])[0]

            if pid not in POLICIES or method not in METHODS:
                self.send_error(400)
                return

            if url.path == "/child":
                # A URL iframe receives its own CSP header.
                body = (
                    "<!doctype html><html><body>"
                    f"{make_content(method)}"
                    "</body></html>"
                ).encode()
            else:
                case = Case(pid, context, method)
                body = page_html(case).encode()

            csp = POLICIES[pid]
            kind = "text/html; charset=utf-8"

        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))

        if csp:
            self.send_header("Content-Security-Policy", csp)

        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


INIT_SCRIPT = """
window.__hits = [];
window.__violations = [];

document.addEventListener(
    "securitypolicyviolation",
    e => window.__violations.push({
        directive: e.effectiveDirective,
        blockedURI: e.blockedURI
    })
);
"""


def execute(browser, base_url, case):
    page = browser.new_page()
    page.add_init_script(INIT_SCRIPT)

    try:
        query = urlencode({
            "policy": case.policy,
            "context": case.context,
            "method": case.method,
        })
        page.goto(
            f"{base_url}/test?{query}",
            wait_until="load",
        )

        if case.context == "top":
            frame = page.main_frame
        else:
            frame = page.frame(name="child")
            if frame is None:
                raise RuntimeError("Child frame missing")

        if case.method in ("handler", "javascript_url"):
            frame.locator("#trigger").click(timeout=3000)

        elif case.method == "dynamic_inline":
            # Drive a DOM insertion in the document.
            # The inserted script is still subject to CSP.
            frame.evaluate("""() => {
                const s = document.createElement('script');
                s.textContent =
                    "window.__hits.push('dynamic_inline')";
                document.getElementById(
                    'injection-point'
                ).appendChild(s);
            }""")

        # Wait for the observation, not merely the click.
        # A blocked script will simply time out.
        try:
            frame.wait_for_function(
                "method => window.__hits.includes(method)",
                arg=case.method,
                timeout=350,
            )
        except Exception:
            pass

        hits = frame.evaluate("window.__hits")
        violations = frame.evaluate(
            "window.__violations"
        )

        return {
            "executed": case.method in hits,
            "hits": hits,
            "violations": violations,
        }

    except Exception as exc:
        return {"error": str(exc)}

    finally:
        page.close()


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), Handler
    )
    base_url = (
        f"http://127.0.0.1:{server.server_address[1]}"
    )

    worker = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    worker.start()

    records = []

    try:
        with sync_playwright() as p:
            browsers = {
                "chromium": p.chromium.launch(headless=True),
                "firefox": p.firefox.launch(headless=True),
            }

            try:
                for name, browser in browsers.items():
                    print(name, browser.version)

                print(
                    f"\n{'POLICY':<12} {'CONTEXT':<12}"
                    f"{'METHOD':<18} {'CHROME':<8}"
                    f"{'FIREFOX':<8} STATUS"
                )

                for case in CASES:
                    outputs = {
                        name: execute(browser, base_url, case)
                        for name, browser in browsers.items()
                    }

                    c = outputs["chromium"]
                    f = outputs["firefox"]

                    if "error" in c or "error" in f:
                        status = "ERROR"
                    elif c["executed"] != f["executed"]:
                        # Re-execute both browsers to filter
                        # one-off inconsistencies.
                        retry = {
                            name: execute(
                                browser, base_url, case
                            )
                            for name, browser in browsers.items()
                        }

                        if (
                            all(
                                "executed" in r
                                for r in retry.values()
                            )
                            and retry["chromium"]["executed"]
                            == c["executed"]
                            and retry["firefox"]["executed"]
                            == f["executed"]
                        ):
                            status = "REPRODUCED"
                        else:
                            status = "FLAKY"

                        outputs["retry"] = retry
                    else:
                        status = "MATCH"

                    records.append({
                        "case": vars(case),
                        "outputs": outputs,
                        "status": status,
                    })

                    print(
                        f"{case.policy:<12} "
                        f"{case.context:<12}"
                        f"{case.method:<18} "
                        f"{str(c.get('executed')):<8}"
                        f"{str(f.get('executed')):<8}"
                        f"{status}"
                    )

            finally:
                for browser in browsers.values():
                    browser.close()

    finally:
        server.shutdown()
        server.server_close()

    with open("exp5_results.json", "w") as file:
        json.dump(records, file, indent=2)

    print("\n=== SUMMARY ===")
    for status in (
        "MATCH", "REPRODUCED", "FLAKY", "ERROR"
    ):
        n = sum(r["status"] == status for r in records)
        print(f"{status}: {n}")

    print("Total:", len(records))
    print("Saved: exp5_results.json")


if __name__ == "__main__":
    main()
