
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

NONCE = "demoNonce123"

POLICIES = {
    "deny": (
        "default-src 'none'; script-src 'none'",
    ),
    "self": (
        "default-src 'none'; script-src 'self'",
    ),
    "inline": (
        "default-src 'none'; script-src 'unsafe-inline'",
    ),
    "elem_only": (
        "default-src 'none'; "
        "script-src 'none'; "
        "script-src-elem 'unsafe-inline' 'self'",
    ),
    "attr_only": (
        "default-src 'none'; "
        "script-src 'none'; "
        "script-src-attr 'unsafe-inline'",
    ),
    "elem_deny": (
        "default-src 'none'; "
        "script-src 'unsafe-inline' 'self'; "
        "script-src-elem 'none'",
    ),
    "attr_deny": (
        "default-src 'none'; "
        "script-src 'unsafe-inline' 'self'; "
        "script-src-attr 'none'",
    ),
    "nonce": (
        "default-src 'none'; "
        f"script-src 'nonce-{NONCE}'",
    ),
    "nonce_inline": (
        "default-src 'none'; "
        f"script-src 'nonce-{NONCE}' 'unsafe-inline'",
    ),
    "multi_header": (
        "default-src 'none'; "
        "script-src 'self' 'unsafe-inline'",
        "script-src 'none'",
    ),
}

METHODS = [
    "inline",
    "external",
    "handler",
    "nonce_valid",
    "nonce_invalid",
]

CASES = list(itertools.product(POLICIES, METHODS))


def make_html(method):
    if method == "inline":
        content = """
        <script>
          window.__hits.push("inline");
        </script>
        """
    elif method == "external":
        content = '<script src="/external.js"></script>'
    elif method == "handler":
        content = """
        <button id="trigger"
          onclick="window.__hits.push('handler')">
          Run
        </button>
        """
    elif method in ("nonce_valid", "nonce_invalid"):
        token = NONCE if method == "nonce_valid" else "wrongNonce"
        content = f"""
        <script nonce="{token}">
          window.__hits.push("{method}");
        </script>
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

        if url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if url.path == "/external.js":
            body = b'window.__hits.push("external");'
            policies = ()
            content_type = "application/javascript"

        elif url.path == "/test":
            query = parse_qs(url.query)
            policy_id = query.get("policy", [""])[0]
            method = query.get("method", [""])[0]

            if policy_id not in POLICIES or method not in METHODS:
                self.send_error(400)
                return

            body = make_html(method).encode()
            policies = POLICIES[policy_id]
            content_type = "text/html; charset=utf-8"

        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))

        for policy in policies:
            self.send_header("Content-Security-Policy", policy)

        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def execute(browser, base_url, policy_id, method):
    page = browser.new_page()
    violations = []

    page.on(
        "console",
        lambda msg: None,
    )

    page.add_init_script("""
        window.__hits = [];
        window.__violations = [];

        document.addEventListener(
            "securitypolicyviolation",
            e => window.__violations.push({
                directive: e.effectiveDirective,
                blockedURI: e.blockedURI
            })
        );
    """)

    try:
        url = (
            f"{base_url}/test?"
            f"policy={policy_id}&method={method}"
        )
        page.goto(url, wait_until="load")

        if method == "handler":
            page.locator("#trigger").click()

        # Allow CSP violation events and short follow-up
        # browser tasks to be observed.
        page.wait_for_timeout(150)

        hits = page.evaluate("window.__hits")
        violations = page.evaluate("window.__violations")

        return {
            "executed": method in hits,
            "hits": hits,
            "violations": violations,
        }
    except Exception as exc:
        return {
            "error": str(exc),
        }
    finally:
        page.close()


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), Handler
    )
    base_url = (
        f"http://127.0.0.1:{server.server_address[1]}"
    )
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    records = []

    try:
        with sync_playwright() as p:
            browsers = {
                "chromium": p.chromium.launch(),
                "firefox": p.firefox.launch(),
            }

            try:
                print("Versions:")
                for name, browser in browsers.items():
                    print(f"  {name}: {browser.version}")

                for policy_id, method in CASES:
                    results = {}

                    for name, browser in browsers.items():
                        results[name] = execute(
                            browser, base_url, policy_id, method
                        )

                    a = results["chromium"]
                    b = results["firefox"]

                    if "error" in a or "error" in b:
                        status = "ERROR"
                    elif a["executed"] == b["executed"]:
                        status = "MATCH"
                    else:
                        status = "DIFFERENCE"

                    # Independently retry discrepancies.
                    if status == "DIFFERENCE":
                        retry = {
                            name: execute(
                                browser, base_url,
                                policy_id, method
                            )
                            for name, browser in browsers.items()
                        }

                        if (
                            all("executed" in r for r in retry.values())
                            and retry["chromium"]["executed"]
                            != retry["firefox"]["executed"]
                            and retry["chromium"]["executed"]
                            == a["executed"]
                            and retry["firefox"]["executed"]
                            == b["executed"]
                        ):
                            status = "REPRODUCED"
                        else:
                            status = "FLAKY"

                        results["retry"] = retry

                    records.append({
                        "policy": policy_id,
                        "method": method,
                        "policy_headers": POLICIES[policy_id],
                        "status": status,
                        "results": results,
                    })

                    print(
                        f"{policy_id:<15} "
                        f"{method:<15} "
                        f"{str(a.get('executed')):<8} "
                        f"{str(b.get('executed')):<8} "
                        f"{status}"
                    )

            finally:
                for browser in browsers.values():
                    browser.close()
    finally:
        server.shutdown()
        server.server_close()

    with open("exp3_results.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\n=== SUMMARY ===")
    for status in ("MATCH", "REPRODUCED", "FLAKY", "ERROR"):
        count = sum(r["status"] == status for r in records)
        print(f"{status}: {count}")

    print("Saved: exp3_results.json")


if __name__ == "__main__":
    main()
