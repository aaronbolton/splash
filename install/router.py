"""`splash serve --switch-models`: one address for every installed model.

The router listens where `splash serve` would and runs the server of one
model at a time behind it, on a private UNIX socket. /v1/models lists every
model installed here, the loaded one first; a request that names another
installed model waits for the requests in flight to end, the loaded model is
stopped, the named one started, and the request goes on to it. Two models
are never loaded at once: on a Mac sized for one, two would exhaust memory.

A request that names no model, the loaded one, or a name the loaded server
answers to (--served-model-name) goes to the loaded server as it is; one
that names neither an installed model nor a name of the loaded one gets the
loaded server's own refusal.

The router keeps the launcher's pid, locks and serve record: the record says
loading, warming and ready for each model the router starts, so the app
follows a switch the API asked for as it follows one of its own. A switch
that fails answers its request with the failure and starts the model that
served before; a start of that one that fails too ends the router with the
failure, as a failed start ends `splash serve`.

Standard library and the launcher's modules alone, as the launcher.
"""

import argparse
import copy
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from server import diagnostics, serve_record

from . import launcher, paths
from . import models as model_artifacts

# How often a start's record is read while the server it started loads.
POLL_SECONDS = 0.2
# How long a server that was asked to stop has before it is killed: its
# engine flushes the SSD cache as it stops.
STOP_SECONDS = 60
# The headers a proxy answers for itself rather than passing on (RFC 9110,
# 7.6.1), and the length, which the router sets for the body it read.
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


@dataclass(frozen=True)
class Installed:
    """A model the router can start: its ID, as requests name it, and the
    selection link `splash serve --model` would serve it from."""

    model: str
    link: Path
    language_only: bool = False


def _present(link):
    """The model ID the assembly at `link` records, while every file it names
    is on this Mac (as `present` in install/cached.py has it); else None."""
    try:
        record = json.loads((link / "model.json").read_text())
        model = record["model"]
        files = record["files"].values()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not isinstance(model, str) or not all(
        isinstance(item, dict) and Path(str(item.get("path"))).exists()
        for item in files
    ):
        return None
    return model


def installed_models(models_root):
    """Every model installed under `models_root` as the app lists them
    (Runtime.installedModels): each OWNER/REPO[:VARIANT] link, and a model
    installed with --language-only alone under the link of that option,
    where it has no link of its own. Sorted by ID."""
    models_root = Path(models_root)
    found = {}
    for link in model_artifacts.selection_links(models_root):
        if link.parent.name.startswith("."):
            continue
        model = _present(link)
        if model is not None:
            found[model] = Installed(model, link)
    selections = models_root / ".selections"
    if selections.is_dir():
        for link in selections.iterdir():
            model = _present(link)
            if model is None or model in found or model.startswith("/"):
                continue
            try:
                expected = model_artifacts.selection_link(
                    models_root, model, language_only=True
                )
            except (ValueError, argparse.ArgumentTypeError):
                continue
            if expected.name == link.name:
                found[model] = Installed(model, link, language_only=True)
    return [found[model] for model in sorted(found)]


class SwitchFailed(Exception):
    """A model the router could not start: the failure its server recorded."""

    def __init__(self, failure):
        super().__init__(failure["message"])
        self.failure = failure


class Models:
    """The server of the one model loaded, which requests hold while they
    run and a switch waits for.

    `spawn(installed, socket_path, record_path)` starts the server of a
    model, listening on `socket_path` and writing its serve record at
    `record_path`, and returns its process."""

    def __init__(self, spawn, catalog, socket_path, record, own_record):
        self._spawn = spawn
        self._catalog = catalog
        self.socket_path = str(socket_path)
        self.record = Path(record)
        # The router's own, which the app follows.
        self._own_record = own_record
        self._condition = threading.Condition()
        self._process = None
        # The model loaded, or loading; None before the first start.
        self.loaded = None
        # The model a switch under way starts; None while none is.
        self.target = None
        self.ready = False
        self._active = 0
        self.closed = False

    def installed(self):
        return self._catalog()

    def find(self, name):
        """The installed model `name` names, or None."""
        if not isinstance(name, str) or name == "":
            return None
        return next((item for item in self.installed() if item.model == name), None)

    def start(self, installed):
        """Start `installed` as the first model: SwitchFailed where it cannot
        be."""
        with self._condition:
            self.target = installed
        self._change(installed, fallback=None)

    def acquire(self, name):
        """Hold the loaded server for a request that names `name`, switching
        to the model it names first where that is another installed one.
        SwitchFailed where the switch fails."""
        wanted = self.find(name)
        with self._condition:
            while True:
                if self.closed:
                    raise SwitchFailed(_stopping())
                if self.target is None:
                    if self.ready and wanted in (None, self.loaded):
                        self._active += 1
                        return
                    # Another model, or the loaded one again where its
                    # server ended.
                    goal = wanted or self.loaded
                    if goal is None:
                        raise SwitchFailed(
                            serve_record.failure(
                                serve_record.STOPPED, "no model is loaded"
                            )
                        )
                    self.target = goal
                    # Said at once, as /status says it: the app follows a
                    # switch from here, the wait for the requests in flight
                    # included.
                    serve_record.entered(self._own_record, "loading")
                    break
                # Another request's switch: this one waits for its end.
                self._condition.wait()
            # The switch is this request's: it waits for those in flight,
            # while those that come wait for it.
            while self._active:
                self._condition.wait()
            previous = self.loaded
        self._change(goal, fallback=previous, hold=True)

    def release(self):
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def wait_ready(self):
        """Wait for a switch under way to end; whether a server serves."""
        with self._condition:
            while self.target is not None and not self.closed:
                self._condition.wait()
            return self.ready

    def _change(self, installed, *, fallback, hold=False):
        """Stop the loaded server and start `installed`'s, held for the
        request that asked for it with `hold`; on failure start `fallback`'s,
        where there is another, and raise the failure."""
        started = False
        try:
            self._stop_process()
            try:
                self._launch(installed)
                started = True
                return
            except SwitchFailed as error:
                failure = error
            if fallback is None or fallback == installed:
                raise failure
            diagnostics.print_line(
                f"Switch to {installed.model} failed: {failure}; "
                f"starting {fallback.model} again",
                error=True,
            )
            self._stop_process()
            self._launch(fallback)
            raise failure
        finally:
            with self._condition:
                self.target = None
                # Held before any request waiting for the switch's end can
                # start another.
                if started and hold:
                    self._active += 1
                self._condition.notify_all()

    def _launch(self, installed):
        with self._condition:
            self.loaded = installed
            self.ready = False
        diagnostics.print_line(f"Loading · {installed.model}")
        serve_record.entered(self._own_record, "loading")
        # What a server of an earlier start left.
        for path in (self.record, Path(self.socket_path)):
            path.unlink(missing_ok=True)
        process = self._spawn(installed, self.socket_path, self.record)
        with self._condition:
            self._process = process
        warmup = None
        while True:
            record = _read_record(self.record)
            if record is not None and record.get("pid") == process.pid:
                phase = record.get("phase")
                if phase == "ready":
                    break
                if phase == "failed":
                    self._stop_process()
                    raise SwitchFailed(record.get("failure") or _ended(installed))
                if phase == "warming" and record.get("warmup") != warmup:
                    warmup = record.get("warmup") or {}
                    serve_record.warming(
                        self._own_record,
                        warmup.get("steps_done", 0),
                        warmup.get("steps", 0),
                    )
            if process.poll() is not None:
                record = _read_record(self.record)
                failure = (
                    (record or {}).get("failure")
                    if (record or {}).get("pid") == process.pid
                    else None
                )
                with self._condition:
                    self._process = None
                raise SwitchFailed(failure or _ended(installed))
            with self._condition:
                if self.closed:
                    raise SwitchFailed(_stopping())
            time.sleep(POLL_SECONDS)
        with self._condition:
            self.ready = True
        serve_record.ready(self._own_record)
        diagnostics.print_line(f"Ready · {installed.model}")

    def exited(self):
        """Whether the loaded server ended unasked."""
        with self._condition:
            process = self._process
            return self.ready and process is not None and process.poll() is not None

    def _stop_process(self):
        with self._condition:
            process, self._process = self._process, None
            self.ready = False
        if process is None or process.poll() is not None:
            return
        # As the app stops `splash serve`: SIGINT, on which the server stops
        # its engine, its flush included.
        process.send_signal(signal.SIGINT)
        try:
            process.wait(STOP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()
        self._stop_process()


def _stopping():
    return serve_record.failure(serve_record.STOPPED, "the server is stopping")


def _ended(installed):
    return serve_record.failure(
        serve_record.INTERNAL, f"the server of {installed.model} ended as it started"
    )


def _read_record(path):
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, models):
        super().__init__(address, RouterHandler)
        self.models = models


class RouterHandler(BaseHTTPRequestHandler):
    server_version = "Splash"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        # The server behind logs every request.
        pass

    @property
    def models(self):
        return self.server.models

    def do_GET(self):
        self._route("GET")

    def do_HEAD(self):
        self._route("HEAD")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def do_PATCH(self):
        self._route("PATCH")

    def do_DELETE(self):
        self._route("DELETE")

    def do_OPTIONS(self):
        self._route("OPTIONS")

    def _route(self, method):
        self.close_connection = True
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except ValueError as error:
            return self._error(400, str(error), "invalid_request_error")
        if method == "GET" and path == "/health":
            return self._json(200, {"status": "ok"})
        if method == "GET" and path == "/status" and self.models.target is not None:
            # While a switch is under way, which the app follows.
            target = self.models.target
            return self._json(
                503,
                {"status": "switching", "model": target and target.model},
            )
        if method == "GET" and path == "/v1/models":
            return self._list()
        if method == "GET" and path.startswith("/v1/models/"):
            return self._list(path.removeprefix("/v1/models/"))
        if method == "POST":
            return self._forward_held(method, body, self._requested(body))
        if not self.models.wait_ready():
            return self._error(503, "no model is loaded", "server_unavailable")
        self._forward(method, body)

    def _requested(self, body):
        """The model a request's JSON body names, or None."""
        if not body or "json" not in (self.headers.get("Content-Type") or "json"):
            return None
        try:
            value = json.loads(body)
        except ValueError:
            return None
        return value.get("model") if isinstance(value, dict) else None

    def _body(self):
        length = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            return self._chunked()
        if length is None:
            return b""
        try:
            count = int(length)
        except ValueError:
            raise ValueError("invalid Content-Length") from None
        if count < 0:
            raise ValueError("invalid Content-Length")
        return self.rfile.read(count)

    def _chunked(self):
        parts = []
        while True:
            line = self.rfile.readline(65537)
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                raise ValueError("invalid chunked body") from None
            if size == 0:
                while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            parts.append(self.rfile.read(size))
            self.rfile.readline(65537)

    def _forward_held(self, method, body, requested):
        try:
            self.models.acquire(requested)
        except SwitchFailed as error:
            return self._error(
                503,
                f"cannot load {requested or 'the model'}: {error}",
                "model_load_failed",
            )
        try:
            self._forward(method, body)
        finally:
            self.models.release()

    def _list(self, name=None):
        """/v1/models: the loaded server's entries, first, then every other
        installed model, which a request may name to load it. The loaded
        server's answer is passed on where it refuses, as for an API key it
        does not take."""
        if not self.models.wait_ready():
            loaded = []
        else:
            status, headers, raw = self._ask("GET", "/v1/models")
            if status != 200:
                return self._send(status, headers, raw)
            try:
                loaded = json.loads(raw)["data"]
            except (ValueError, KeyError, TypeError):
                loaded = []
        for item in loaded:
            item["loaded"] = True
        names = {item.get("id") for item in loaded}
        others = [
            {
                "id": installed.model,
                "object": "model",
                "created": 0,
                "owned_by": "splash",
                "loaded": False,
                **({"input_modalities": ["text"]} if installed.language_only else {}),
            }
            for installed in self.models.installed()
            if installed.model not in names
        ]
        data = loaded + others
        if name is not None:
            item = next((item for item in data if item.get("id") == name), None)
            if item is None:
                return self._error(404, "model not found", "model_not_found")
            return self._json(200, item)
        self._json(
            200,
            {
                "object": "list",
                "data": data,
                # As the server lists them, for TypeSafe's SDK.
                "models": [
                    {
                        "name": item["id"],
                        "description": "Splash resident model"
                        if item["loaded"]
                        else "Splash installed model",
                        "release_date": "",
                    }
                    for item in data
                ],
            },
        )

    def _headers_out(self):
        """The request's headers for the server behind: those of the
        connection dropped, and a Host of the router's own address named as
        loopback, which the server serves on its socket; a page of that
        address is the server's own, so its Origin is dropped as well."""
        own = {self.server.server_address[0].lower(), self._local_host()}
        lines = []
        host = self.headers.get("Host", "")
        name = host.rsplit(":", 1)[0].strip("[]").lower() if host else ""
        rewrite = name in own and name not in ("127.0.0.1", "localhost", "::1")
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_HEADERS:
                continue
            if rewrite and lower == "host":
                value = "127.0.0.1"
            if rewrite and lower == "origin" and name in value.lower():
                continue
            lines.append(f"{key}: {value}")
        return lines

    def _local_host(self):
        try:
            return self.connection.getsockname()[0].lower()
        except (OSError, IndexError, TypeError):
            return ""

    def _connect(self):
        child = socket.socket(socket.AF_UNIX)
        child.connect(self.models.socket_path)
        return child

    def _request_bytes(self, method, path, body, headers):
        head = [f"{method} {path} HTTP/1.1", *headers, "Connection: close"]
        if body or method in ("POST", "PUT", "PATCH"):
            head.append(f"Content-Length: {len(body)}")
        return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body

    def _ask(self, method, path):
        """A whole answer of the server behind: status, headers and body."""
        try:
            with self._connect() as child:
                child.sendall(
                    self._request_bytes(method, path, b"", self._headers_out())
                )
                raw = b""
                while chunk := child.recv(65536):
                    raw += chunk
        except OSError:
            return 503, [], b""
        head, _, body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        try:
            status = int(lines[0].split(" ", 2)[1])
        except (IndexError, ValueError):
            return 502, [], b""
        headers = [line.split(":", 1) for line in lines[1:] if ":" in line]
        if any(k.strip().lower() == "transfer-encoding" for k, _ in headers):
            body = _dechunk(body)
        return status, [(k.strip(), v.strip()) for k, v in headers], body

    def _forward(self, method, body):
        """Pass the request to the server behind and its answer back as it
        comes, a stream's events as they are sent: the server closes the
        connection once it has answered."""
        try:
            child = self._connect()
        except OSError:
            return self._error(503, "no model is loaded", "server_unavailable")
        with child:
            try:
                child.sendall(
                    self._request_bytes(method, self.path, body, self._headers_out())
                )
            except OSError:
                return self._error(503, "the model stopped", "server_unavailable")
            while True:
                try:
                    chunk = child.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    # The client left: closing the connection tells the
                    # server, which ends the request.
                    return

    def _send(self, status, headers, body):
        self.send_response(status)
        for key, value in headers:
            if key.lower() not in HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, value):
        self._send(
            status,
            [("Content-Type", "application/json")],
            json.dumps(value).encode(),
        )

    def _error(self, status, message, code):
        self._json(
            status,
            {"error": {"message": message, "type": code, "code": code}},
        )


def _dechunk(data):
    parts = []
    while data:
        line, _, rest = data.partition(b"\r\n")
        try:
            size = int(line.split(b";", 1)[0], 16)
        except ValueError:
            break
        if size == 0:
            break
        parts.append(rest[:size])
        data = rest[size + 2 :]
    return b"".join(parts)


def _spawner(args):
    """Start the server of a model as `splash serve --model` runs it, its
    options `args`', listening on a UNIX socket."""

    def spawn(installed, socket_path, record):
        served = copy.copy(args)
        served.model = installed.model
        served.language_only = installed.language_only
        served.host = f"unix:{socket_path}"
        command, environment = launcher._server(served, installed.link, record)
        # The server stops once the router has ended, a crash too.
        environment["SPLASH_APP_PID"] = str(os.getpid())
        return subprocess.Popen(command, env=environment, cwd=paths.ROOT)

    return spawn


def _watch(pid):
    """Stop the router once process `pid` (the app) has ended, as the server
    stops with it."""
    if not hasattr(select, "kqueue"):
        return

    def watch():
        queue = select.kqueue()
        exited = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        queue.control([exited], 1, None)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="stop-with-app", daemon=True).start()


class _Stop(Exception):
    pass


def _stop(*_):
    # One stop: a second signal, as a stopping app's and the watch's, waits
    # for the first's cleanup rather than breaking it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise _Stop


def route(args, first_link, record, *, spawn=None, catalog=None):
    """Serve `args.host`:`args.port` for every installed model, starting
    with `args.model` from `first_link`. Returns the exit status."""
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, _stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
    if app := os.environ.get("SPLASH_APP_PID"):
        _watch(int(app))
    models_root = paths.MODELS

    def every():
        found = installed_models(models_root) if catalog is None else catalog()
        # The first model, as `splash serve` was given it, stays one a
        # request may name: model files on this Mac have no link of their own.
        if not any(item.model == first.model for item in found):
            found = [first, *found]
        return found

    first = Installed(args.model, Path(first_link), bool(args.language_only))
    runtime = paths.RUNTIME
    models = Models(
        spawn or _spawner(args),
        every,
        runtime / f"serve-{args.port}-model.sock",
        runtime / f"serve-{args.port}-model.json",
        record,
    )
    server = None
    try:
        server = RouterServer((args.host, args.port), models)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            models.start(first)
        except SwitchFailed as error:
            serve_record.failed(record, error.failure)
            diagnostics.print_line(f"Error · {error}", error=True)
            return 1
        diagnostics.print_line(
            f"Switching models on request · http://{args.host}:{args.port}"
        )
        while True:
            time.sleep(1)
            if models.exited() and models.target is None:
                # Its server stopped serving: the router does too, as the
                # server would have.
                record_failure = _read_record(models.record) or {}
                failure = record_failure.get("failure") or serve_record.failure(
                    serve_record.STOPPED, f"{models.loaded.model} stopped serving"
                )
                serve_record.failed(record, failure)
                diagnostics.print_line(f"Error · {failure['message']}", error=True)
                return 1
    except _Stop:
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None:
            server.shutdown()
            server.server_close()
        models.close()


def main(argv=None):
    """Run as `splash serve --switch-models` executes it: the serve command
    line, the first model's prepared directory in SPLASH_ROUTER_MODEL_DIR."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = launcher.parse_args(argv)
    first_link = os.environ.pop("SPLASH_ROUTER_MODEL_DIR")
    return route(args, first_link, launcher._serve_record(args.port))


if __name__ == "__main__":
    raise SystemExit(main())
