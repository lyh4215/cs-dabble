
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright


POLICIES = {
    "deny_all": "default-src 'none'; script-src 'none'",
    "self": "default-src 'none'; script-src 'self'",
    "inline": "default-src 'none'; script-src 'unsafe-inline'",
    "self_inline": (
        "default-src 'none'; "
        "script-src 'self' 'unsafe-inline'"
    ),
    "fallback": "default-src 'none'",
}

METHODS = [
    "inline_script",
    "external_script",
    "event_handler",
    "javascript_url",
]


def make_html(method):
    if method == "inline_script":
        content = """
        <script>
          window.__hits.push("inline_script");
        </script>
        """

    elif method == "external_script":
        content = '<script src="/external.js"></script>'

    elif method == "event_handler":
        content = """
        <button id="trigger"
          onclick="window.__hits.push('event_handler')">
          Run
        </button>
        """

    elif method == "javascript_url":
        content = """
        <a id="trigger"
          href="javascript:void(window.__hits.push('javascript_url'))">
          Run
        </a>
        """

    else:
        raise ValueError(method)

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body>{content}</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/external.js":
            body = (
                b'window.__hits.push("external_script");'
            )
            policy = None

        elif url.path == "/test":
            args = parse_qs(url.query)
            policy_id = args["policy"][0]
            method = args["method"][0]

            if policy_id not in POLICIES or method not in METHODS:
                self.send_error(400)
                return

            policy = POLICIES[policy_id]
            body = make_html(method).encode()

        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", (
            "application/javascript"
            if url.path == "/external.js"
            else "text/html; charset=utf-8"
        ))
        self.send_header("Content-Length", str(len(body)))

        if policy:
            self.send_header("Content-Security-Policy", policy)

        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass



def run_case(browser, base_url, policy_id, method):
    page = browser.new_page()
    console_messages = []
    violations = []
    page_errors = []

    page.on(
        "console",
        lambda msg: console_messages.append(
            {"type": msg.type, "text": msg.text}
        )
    )
    page.on(
        "pageerror",
        lambda error: page_errors.append(str(error))
    )

    page.add_init_script("""
        window.__hits = [];
        window.__violations = [];

        document.addEventListener(
            "securitypolicyviolation",
            event => {
                window.__violations.push({
                    directive: event.effectiveDirective,
                    blockedURI: event.blockedURI,
                    disposition: event.disposition
                });
            }
        );
    """)

    try:
        url = (
            f"{base_url}/test?"
            f"policy={policy_id}&method={method}"
        )
        response = page.goto(url, wait_until="load")

        if method in ("event_handler", "javascript_url"):
            page.locator("#trigger").click(timeout=3000)

        page.wait_for_timeout(200)

        hits = page.evaluate("window.__hits")
        violations = page.evaluate(
            "window.__violations"
        )

        if method == "javascript_url":
            print(
                "\n[DIAGNOSTIC]",
                browser.browser_type.name,
                policy_id
            )
            print("HTTP:", response.status)
            print("CSP:", response.headers.get(
                "content-security-policy"
            ))
            print("Hits:", hits)
            print("Violations:", violations)
            print("Console:", console_messages)
            print("Page errors:", page_errors)
            print("Current URL:", page.url)

        return method in hits

    finally:
        page.close()



def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]

    thread = threading.Thread(
        target=server.serve_forever, daemon=True
    )
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    records = []

    try:
        with sync_playwright() as p:
            browsers = {
                "chromium": p.chromium.launch(headless=True),
                "firefox": p.firefox.launch(headless=True),
            }

            try:
                print(
                    f"{'POLICY':<14} {'METHOD':<18}"
                    f"{'CHROMIUM':<12} {'FIREFOX':<12} RESULT"
                )
                print("-" * 68)

                for policy_id in POLICIES:
                    for method in METHODS:
                        outputs = {}

                        for name, browser in browsers.items():
                            try:
                                outputs[name] = run_case(
                                    browser, base_url, policy_id, method
                                )
                            except Exception as exc:
                                outputs[name] = None
                                print(
                                    f"ERROR [{name}, {policy_id}, "
                                    f"{method}]: {exc}"
                                )

                        a = outputs["chromium"]
                        b = outputs["firefox"]

                        if a is None or b is None:
                            status = "ERROR"
                        elif a != b:
                            status = "DIFFERENCE"
                        else:
                            status = "MATCH"

                        record = {
                            "policy": policy_id,
                            "method": method,
                            "chromium": a,
                            "firefox": b,
                            "status": status,
                        }
                        records.append(record)

                        print(
                            f"{policy_id:<14} {method:<18}"
                            f"{str(a):<12} {str(b):<12} {status}"
                        )
            finally:
                for browser in browsers.values():
                    browser.close()

    finally:
        server.shutdown()
        server.server_close()

    differences = [
        r for r in records if r["status"] == "DIFFERENCE"
    ]
    errors = [
        r for r in records if r["status"] == "ERROR"
    ]

    with open("diffcsp_results.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\n=== SUMMARY ===")
    print("Test cases:", len(records))
    print("Differences:", len(differences))
    print("Errors:", len(errors))
    print("Saved: diffcsp_results.json")


if __name__ == "__main__":
    main()
