import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from install import models as model_artifacts
from install import router

# A server of one model, as `splash serve` behind the router runs one: it
# listens on the socket, says ready in its record, names its model in every
# answer, and stops on SIGINT. A model named fail/... records a failed start.
FAKE_SERVER = r"""
import json, os, signal, socket, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

model, path, record = sys.argv[1:4]

def write(phase, **more):
    Path = record + ".tmp"
    with open(Path, "w") as file:
        json.dump({"pid": os.getpid(), "phase": phase, **more}, file)
    os.replace(Path, record)

write("loading")
if model.startswith("fail/"):
    write("failed", failure={"kind": "model", "message": "cannot load " + model})
    sys.exit(1)

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def answer(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") == "Bearer wrong":
            return self.answer(401, {"error": "unauthorized"})
        if self.path == "/v1/models":
            return self.answer(200, {"object": "list", "data": [
                {"id": model, "object": "model", "context_length": 4096}]})
        if self.path == "/status":
            return self.answer(200, {"model": model})
        self.answer(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("slow"):
            time.sleep(body["slow"])
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for index in range(3):
                self.wfile.write(f"data: {index}\n\n".encode())
                self.wfile.flush()
            return
        self.answer(200, {"served_by": model, "asked": body.get("model"),
                          "host": self.headers.get("Host")})

class Server(HTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)

    def get_request(self):
        connection, _ = self.socket.accept()
        return connection, ("unix", 0)

server = Server(path, Handler)
signal.signal(signal.SIGINT, lambda *_: os._exit(0))
write("warming", warmup={"steps_done": 1, "steps": 2})
write("ready")
server.serve_forever()
"""


class Fakes:
    """Model servers the router starts, each counted as it starts."""

    def __init__(self):
        self.started = []
        self.lock = threading.Lock()

    def spawn(self, installed, socket_path, record):
        with self.lock:
            self.started.append(installed.model)
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                FAKE_SERVER,
                installed.model,
                socket_path,
                str(record),
            ]
        )


def installed(*names):
    return [router.Installed(name, Path("/models") / name) for name in names]


class RouterTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.catalog = installed("a/one", "b/two", "fail/three")
        self.fakes = Fakes()
        self.own_record = self.directory / "serve.json"
        self.models = router.Models(
            self.fakes.spawn,
            lambda: self.catalog,
            self.directory / "model.sock",
            self.directory / "model.json",
            self.own_record,
        )
        self.addCleanup(self.models.close)
        self.models.start(self.catalog[0])
        self.server = router.RouterServer(("127.0.0.1", 0), self.models)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=30
        )
        data = None if body is None else json.dumps(body).encode()
        connection.request(
            method,
            path,
            body=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, raw

    def post(self, body, path="/v1/chat/completions"):
        status, raw = self.request("POST", path, body)
        return status, json.loads(raw)

    def test_lists_every_installed_model_the_loaded_one_first(self):
        status, raw = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        listed = json.loads(raw)
        self.assertEqual(
            [(item["id"], item["loaded"]) for item in listed["data"]],
            [("a/one", True), ("b/two", False), ("fail/three", False)],
        )
        # The loaded server's own entry, as it gives it.
        self.assertEqual(listed["data"][0]["context_length"], 4096)
        self.assertEqual(
            [item["name"] for item in listed["models"]],
            ["a/one", "b/two", "fail/three"],
        )
        status, raw = self.request("GET", "/v1/models/b/two")
        self.assertEqual((status, json.loads(raw)["loaded"]), (200, False))
        status, _ = self.request("GET", "/v1/models/c/none")
        self.assertEqual(status, 404)

    def test_a_listing_the_loaded_server_refuses_is_refused(self):
        status, _ = self.request(
            "GET", "/v1/models", headers={"Authorization": "Bearer wrong"}
        )
        self.assertEqual(status, 401)

    def test_a_request_for_another_installed_model_switches_to_it(self):
        self.assertEqual(self.post({"model": "a/one"})[1]["served_by"], "a/one")
        self.assertEqual(self.post({})[1]["served_by"], "a/one")
        status, answer = self.post({"model": "b/two"})
        self.assertEqual((status, answer["served_by"]), (200, "b/two"))
        # Requests naming none go on to the model loaded now.
        self.assertEqual(self.post({})[1]["served_by"], "b/two")
        self.assertEqual(self.fakes.started, ["a/one", "b/two"])
        listed = json.loads(self.request("GET", "/v1/models")[1])["data"]
        self.assertEqual(listed[0]["id"], "b/two")
        self.assertEqual(
            json.loads(self.request("GET", "/status")[1]), {"model": "b/two"}
        )
        # The router's record says ready again once the switch ended.
        self.assertEqual(json.loads(self.own_record.read_text())["phase"], "ready")

    def test_a_name_no_installed_model_has_goes_to_the_loaded_server(self):
        status, answer = self.post({"model": "gpt-4o"})
        self.assertEqual((status, answer["served_by"]), (200, "a/one"))
        self.assertEqual(self.fakes.started, ["a/one"])

    def test_a_switch_waits_for_the_requests_in_flight(self):
        results = {}

        def slow():
            results["slow"] = self.post({"slow": 1.0})[1]["served_by"]

        thread = threading.Thread(target=slow)
        thread.start()
        time.sleep(0.3)
        self.assertEqual(self.post({"model": "b/two"})[1]["served_by"], "b/two")
        thread.join()
        self.assertEqual(results["slow"], "a/one")

    def test_concurrent_requests_for_one_model_switch_once(self):
        answers = []

        def ask():
            answers.append(self.post({"model": "b/two"})[1]["served_by"])

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(answers, ["b/two"] * 4)
        self.assertEqual(self.fakes.started, ["a/one", "b/two"])

    def test_a_failed_switch_says_why_and_starts_the_model_before_again(self):
        status, answer = self.post({"model": "fail/three"})
        self.assertEqual(status, 503)
        self.assertIn("cannot load fail/three", answer["error"]["message"])
        self.assertEqual(self.post({})[1]["served_by"], "a/one")
        self.assertEqual(self.fakes.started, ["a/one", "fail/three", "a/one"])

    def test_a_stream_passes_through_as_it_comes(self):
        status, raw = self.request("POST", "/stream", {})
        self.assertEqual(status, 200)
        self.assertEqual(raw, b"data: 0\n\ndata: 1\n\ndata: 2\n\n")

    def test_a_loopback_host_reaches_the_server_as_it_is(self):
        port = self.server.server_address[1]
        self.assertEqual(self.post({})[1]["host"], f"127.0.0.1:{port}")


class InstalledModelsTest(unittest.TestCase):
    def test_lists_named_and_text_only_installations_with_their_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            weights = Path(directory) / "weights.gguf"
            weights.write_bytes(b"x")

            def install(link, model, files):
                target = Path(directory) / "assemblies" / link.name
                target.mkdir(parents=True)
                (target / "model.json").write_text(
                    json.dumps({"model": model, "files": files})
                )
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target)

            present = {"model.gguf": {"path": str(weights), "bytes": 1}}
            install(root / "owner" / "named:Q4", "owner/named:Q4", present)
            install(
                model_artifacts.selection_link(root, "owner/text", language_only=True),
                "owner/text",
                present,
            )
            install(
                root / "owner" / "gone",
                "owner/gone",
                {"model.gguf": {"path": str(Path(directory) / "missing"), "bytes": 1}},
            )
            found = router.installed_models(root)
        self.assertEqual(
            [(item.model, item.language_only) for item in found],
            [("owner/named:Q4", False), ("owner/text", True)],
        )


class LauncherTest(unittest.TestCase):
    def test_switch_models_runs_the_router_with_the_serve_command_line(self):
        from install import launcher

        argv = ["serve", "--model", "owner/repo", "--switch-models", "--port", "8123"]
        args = launcher.parse_args(argv)
        self.assertTrue(args.switch_models)
        with mock.patch.dict(os.environ, {}, clear=False):
            command, environment = launcher._router(args, Path("/prepared"))
        self.assertEqual(command[-len(argv) - 1 :], ["install.router", *argv])
        self.assertEqual(environment["SPLASH_ROUTER_MODEL_DIR"], "/prepared")
        routed = launcher.parse_args(command[command.index("install.router") + 1 :])
        self.assertEqual((routed.model, routed.port), ("owner/repo", 8123))


if __name__ == "__main__":
    unittest.main()
