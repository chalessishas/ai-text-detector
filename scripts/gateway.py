#!/usr/bin/env python3
"""Lightweight HTTP gateway that routes /detect → :5001 and /humanize → :5002.

Railway only exposes one port, so this gateway merges both services.
"""

import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

DETECT_URL = os.environ.get("DETECT_URL", "http://127.0.0.1:5001")
HUMANIZE_URL = os.environ.get("HUMANIZE_URL", "http://127.0.0.1:5002")


class GatewayHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/detect":
            self._proxy(DETECT_URL)
        elif self.path == "/humanize":
            self._proxy(HUMANIZE_URL)
        elif self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": f"Unknown path: {self.path}"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "POST to /detect or /humanize"})

    def _proxy(self, target_url):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""

            req = Request(
                target_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
        except URLError as e:
            self._respond(502, {"error": f"Backend unavailable: {e.reason}"})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, status, obj):
        import json
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[gateway] {args[0]}", file=sys.stderr)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), GatewayHandler)
    print(f"Gateway running on :{port}  /detect → {DETECT_URL}  /humanize → {HUMANIZE_URL}", file=sys.stderr)
    server.serve_forever()
