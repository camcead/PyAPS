"""A static image export can never block a science product (hung PPXF/EMI/RR/FR/RVS jobs of 2026-10-08..12).

Kaleido 1.x waits for its headless Chrome without a deadline while the browser starts, navigates and closes. A render that
never came back left the job idle until the SLURM wall limit, and the night it belonged to stayed PROCESSING.
``PyAPS.apsPlot.export.write_image`` renders in a separate worker process with a deadline per image, kills the worker and
its browser when the deadline passes, skips that picture and carries on, and switches the pictures off after repeated
failures. These tests use a fake worker that speaks the same line protocol (no Chrome, no Kaleido, no network).

Run with:
    cd <PYAPS_DIR> && pytest tests/test_plot_export.py -v
"""
from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import time

import pytest

from PyAPS.apsPlot import export

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="process groups and pipes are POSIX")

FAKE_WORKER = textwrap.dedent('''
    import json, os, subprocess, sys, time
    state = os.environ["FAKE_STATE"]
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        what = req["fig"]
        with open(state, "a") as fh:
            fh.write(f"{os.getpid()} {what}\\n")
        if what == "hang":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])   # the 'browser'
            with open(state + ".child", "w") as fh:
                fh.write(str(child.pid))
            time.sleep(600)
        elif what == "die":
            os._exit(3)
        elif what == "error":
            reply = {"id": req["id"], "ok": False, "error": "ValueError: bad figure"}
        elif what == "fatal":
            reply = {"id": req["id"], "ok": False, "error": "no browser", "fatal": True}
        elif what == "noise":
            sys.stdout.write("not json at all\\n")
            sys.stdout.flush()
            reply = {"id": req["id"], "ok": True}
        else:
            with open(req["path"], "w") as fh:
                fh.write("png " + json.dumps(req["kwargs"], sort_keys=True))
            reply = {"id": req["id"], "ok": True}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
''')


class Fig:
    """Stands in for a Plotly figure: the export only needs to_json() (and write_image() in the in-process mode)."""
    def __init__(self, what="ok"):
        self.what = what
        self.direct = []

    def to_json(self):
        return self.what

    def write_image(self, path, **kw):
        self.direct.append((path, kw))
        if self.what == "error":
            raise ValueError("bad figure")
        with open(path, "w") as fh:
            fh.write("direct")


@pytest.fixture
def worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    state = tmp_path / "state.txt"
    monkeypatch.setenv("FAKE_STATE", str(state))
    monkeypatch.setattr(export, "_worker_command", lambda: [sys.executable, str(script)])
    for var in ("PYAPS_PLOT_EXPORT", "PYAPS_PLOT_TIMEOUT", "PYAPS_PLOT_START_TIMEOUT", "PYAPS_PLOT_MAX_FAILURES",
                "PYAPS_PLOT_RECYCLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYAPS_PLOT_TIMEOUT", "2")
    monkeypatch.setenv("PYAPS_PLOT_START_TIMEOUT", "2")
    export.reset()
    yield state
    export.reset()


def pids(state):
    return [int(l.split()[0]) for l in state.read_text().splitlines()] if state.exists() else []


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_a_normal_image_is_written_with_the_keyword_arguments(worker, tmp_path):
    out = tmp_path / "a.png"
    assert export.write_image(Fig(), out, scale=2) is True
    assert json.loads(out.read_text().split(" ", 1)[1]) == {"scale": 2}
    assert export.status()["written"] == 1


def test_no_scale_is_passed_when_none_is_given(worker, tmp_path):
    out = tmp_path / "a.pdf"
    assert export.write_image(Fig(), out) is True
    assert json.loads(out.read_text().split(" ", 1)[1]) == {}


def test_one_worker_serves_many_images(worker, tmp_path):
    for i in range(5):
        assert export.write_image(Fig(), tmp_path / f"{i}.png", scale=2)
    assert len(set(pids(worker))) == 1


def test_a_hung_render_is_skipped_within_the_deadline_and_the_browser_is_killed(worker, tmp_path, caplog):
    t0 = time.monotonic()
    out = tmp_path / "hang.png"
    assert export.write_image(Fig("hang"), out, scale=2, timeout=1) is False
    assert time.monotonic() - t0 < 6
    assert not out.exists()
    worker_pid = pids(worker)[0]
    child = int((worker.parent / "state.txt.child").read_text())
    time.sleep(0.5)
    assert not alive(worker_pid) and not alive(child)                  # the whole process group is gone
    assert "gave no answer" in caplog.text and "killed" in caplog.text
    assert export.status()["timeouts"] == 1


def test_the_next_image_after_a_hang_gets_a_new_worker(worker, tmp_path):
    assert export.write_image(Fig("hang"), tmp_path / "h.png", timeout=1) is False
    assert export.write_image(Fig(), tmp_path / "ok.png", scale=2) is True
    first, second = pids(worker)[0], pids(worker)[-1]
    assert first != second and (tmp_path / "ok.png").exists()


def test_a_worker_that_dies_is_replaced(worker, tmp_path):
    assert export.write_image(Fig("die"), tmp_path / "d.png") is False
    assert export.write_image(Fig(), tmp_path / "ok.png") is True


def test_three_hangs_in_a_row_switch_the_pictures_off(worker, tmp_path, caplog):
    for i in range(3):
        assert export.write_image(Fig("hang"), tmp_path / f"{i}.png", timeout=0.5) is False
    assert export.status()["disabled"] is True
    n = len(pids(worker))
    t0 = time.monotonic()
    assert export.write_image(Fig(), tmp_path / "later.png") is False   # not even tried
    assert time.monotonic() - t0 < 0.2 and len(pids(worker)) == n
    assert "switched off" in caplog.text


def test_a_success_between_hangs_resets_the_count(worker, tmp_path):
    for i in range(4):
        export.write_image(Fig("hang"), tmp_path / f"h{i}.png", timeout=0.5)
        assert export.write_image(Fig(), tmp_path / f"ok{i}.png") is True
    assert export.status()["disabled"] is False


def test_an_ordinary_error_skips_the_picture_and_does_not_count_towards_the_switch_off(worker, tmp_path, caplog):
    for i in range(5):
        assert export.write_image(Fig("error"), tmp_path / f"e{i}.png") is False
    assert export.status()["disabled"] is False
    assert export.write_image(Fig(), tmp_path / "ok.png") is True
    assert "bad figure" in caplog.text


def test_no_browser_at_all_switches_the_pictures_off_at_once(worker, tmp_path):
    assert export.write_image(Fig("fatal"), tmp_path / "f.png") is False
    assert export.status()["disabled"] is True


def test_stray_output_of_a_library_on_stdout_is_ignored(worker, tmp_path):
    assert export.write_image(Fig("noise"), tmp_path / "n.png") is True


def test_the_worker_is_replaced_after_the_recycle_count(worker, tmp_path, monkeypatch):
    monkeypatch.setenv("PYAPS_PLOT_RECYCLE", "3")
    for i in range(7):
        assert export.write_image(Fig(), tmp_path / f"{i}.png")
    assert len(set(pids(worker))) == 3                                  # 3 + 3 + 1


def test_a_worker_inherited_through_fork_is_never_used(worker, tmp_path):
    assert export.write_image(Fig(), tmp_path / "a.png")
    parent_pid = pids(worker)[0]
    pid = os.fork()
    if pid == 0:                                                        # the child of a multiprocessing pool
        try:
            ok = export.write_image(Fig(), tmp_path / "child.png")
            code = 0 if ok and pids(worker)[-1] != parent_pid else 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    assert alive(parent_pid)                                            # the child did not kill the parent's worker


def test_mode_off_writes_nothing(worker, tmp_path, monkeypatch):
    monkeypatch.setenv("PYAPS_PLOT_EXPORT", "off")
    assert export.write_image(Fig(), tmp_path / "o.png") is False
    assert not (tmp_path / "o.png").exists() and not pids(worker)


def test_mode_inprocess_uses_the_figure_and_swallows_errors(worker, tmp_path, monkeypatch):
    monkeypatch.setenv("PYAPS_PLOT_EXPORT", "inprocess")
    fig = Fig()
    assert export.write_image(fig, tmp_path / "i.png", scale=2) is True
    assert fig.direct == [(str(tmp_path / "i.png"), {"scale": 2})] and not pids(worker)
    assert export.write_image(Fig("error"), tmp_path / "e.png") is False


def test_a_figure_that_cannot_be_serialised_is_skipped(worker, tmp_path):
    class Broken(Fig):
        def to_json(self):
            raise TypeError("not serialisable")
    assert export.write_image(Broken(), tmp_path / "b.png") is False
    assert export.write_image(Fig(), tmp_path / "ok.png") is True


def test_a_worker_that_cannot_be_started_skips_the_picture(worker, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_worker_command", lambda: ["/nonexistent/python-for-the-test"])
    assert export.write_image(Fig(), tmp_path / "x.png") is False


def test_shutdown_ends_the_worker(worker, tmp_path):
    assert export.write_image(Fig(), tmp_path / "a.png")
    pid = pids(worker)[0]
    export.shutdown()
    time.sleep(0.3)
    assert not alive(pid)


def test_every_static_image_call_of_the_pipeline_goes_through_the_guarded_export():
    """No module may call Figure.write_image directly (it can wait for the browser for ever)."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "py" / "PyAPS"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in ("export.py", "_export_worker.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("write_image", "write_images", "to_image")):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


@pytest.mark.skipif(not os.environ.get("PYAPS_TEST_KALEIDO"), reason="needs Kaleido and Chrome (set PYAPS_TEST_KALEIDO=1)")
def test_real_worker_renders_png_and_pdf(tmp_path, monkeypatch):
    """The real worker with the real Kaleido and Chrome: many images from one browser, atomic file, then a clean exit."""
    go = pytest.importorskip("plotly.graph_objects")
    monkeypatch.delenv("PYAPS_PLOT_EXPORT", raising=False)
    monkeypatch.setenv("PYAPS_PLOT_START_TIMEOUT", "120")
    export.reset()
    try:
        fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2]))
        for i in range(3):
            assert export.write_image(fig, tmp_path / f"{i}.png", scale=2) is True
        assert export.write_image(fig, tmp_path / "f.pdf") is True
        assert (tmp_path / "0.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert (tmp_path / "f.pdf").read_bytes()[:4] == b"%PDF"
        assert not list(tmp_path.glob("*.part.*"))
    finally:
        export.reset()


def test_a_real_worker_ends_with_its_process_group_when_the_parent_is_gone(tmp_path):
    """The real worker script (no Kaleido needed to start): killed parent -> the worker's watchdog ends worker and group."""
    import subprocess

    pidfile = tmp_path / "worker.pid"
    launcher = tmp_path / "launcher.py"
    launcher.write_text(textwrap.dedent(f'''
        import subprocess, sys
        w = subprocess.Popen([sys.executable, {str(export._WORKER)!r}], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        open({str(pidfile)!r}, "w").write(str(w.pid))
        import time; time.sleep(600)
    '''))
    parent = subprocess.Popen([sys.executable, str(launcher)])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    worker_pid = int(pidfile.read_text())
    assert alive(worker_pid)
    parent.kill()
    parent.wait()
    for _ in range(100):                                                # the watchdog looks every 5 s
        if not alive(worker_pid):
            break
        time.sleep(0.1)
    assert not alive(worker_pid)
