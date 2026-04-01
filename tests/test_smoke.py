"""
Playwright smoke tests for the NORT dashboard.

These tests verify basic dashboard functionality: page loading, canvas
rendering, frame rate, WebSocket connectivity, keyboard shortcuts, and
the health API endpoint.

Requirements:
    pip install playwright pytest-playwright
    python -m playwright install chromium

The serve.py server must be running on http://localhost:8000/ before
executing these tests.  A module-scoped fixture starts the server
automatically if it is not already reachable.

Run:
    pytest tests/test_smoke.py -v
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("NORT_TEST_URL", "http://localhost:8000")
HEALTH_URL = f"{BASE_URL}/api/health"
SERVER_STARTUP_TIMEOUT = 15  # seconds
PAGE_LOAD_TIMEOUT = 10_000  # ms (Playwright uses ms)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.smoke

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _server_is_ready() -> bool:
    """Return True if the health endpoint responds with 200."""
    import urllib.request
    import urllib.error
    try:
        resp = urllib.request.urlopen(HEALTH_URL, timeout=2)
        return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(scope="module")
def server():
    """Ensure the NORT server is running for the duration of this module.

    If the server is already reachable, the fixture is a no-op.
    Otherwise it starts serve.py as a subprocess and tears it down
    after all tests in this module have finished.
    """
    if _server_is_ready():
        yield None
        return

    env = os.environ.copy()
    env["NORT_PORT"] = "8000"
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "serve.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _server_is_ready():
            break
        time.sleep(0.3)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail(
            f"serve.py did not become ready within {SERVER_STARTUP_TIMEOUT}s"
        )

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser_context(server):
    """Launch a headless Chromium browser and yield a browser context."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        yield context
        context.close()
        browser.close()


@pytest.fixture()
def page(browser_context):
    """Create a fresh page for each test."""
    pg = browser_context.new_page()
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDashboardSmoke:
    """Smoke tests for the NORT dashboard."""

    def test_page_loads_without_errors(self, page):
        """Page loads successfully with no console errors."""
        console_errors = []

        def _on_console(msg):
            if msg.type == "error":
                text = msg.text
                # Filter known benign messages (e.g. favicon 404, WebSocket
                # reconnect noise when no orchestrator is pushing data).
                benign = (
                    "favicon",
                    "404",
                    "ERR_CONNECTION_REFUSED",
                    "WebSocket",
                    "net::ERR",
                )
                if not any(b in text for b in benign):
                    console_errors.append(text)

        page.on("console", _on_console)
        page.goto(BASE_URL, wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT)

        # Give the page a moment to settle (async scripts, WS connect, etc.)
        page.wait_for_timeout(1000)

        assert page.title() == "NORT", f"Unexpected page title: {page.title()}"
        body_text = page.text_content("body")
        assert body_text is not None and len(body_text) > 0

        assert console_errors == [], (
            f"Unexpected console errors: {console_errors}"
        )

    def test_canvas_renders(self, page):
        """A <canvas> element exists with non-zero dimensions."""
        page.goto(BASE_URL, wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT)

        canvas = page.locator("canvas#canvas")
        canvas.wait_for(state="attached", timeout=5000)

        # The canvas gets its size from JS (initCanvas → resize), so give
        # the render loop a tick to set dimensions.
        page.wait_for_timeout(500)

        dims = page.evaluate("""() => {
            const c = document.getElementById('canvas');
            return { width: c.width, height: c.height,
                     styleW: c.style.width, styleH: c.style.height };
        }""")

        assert dims["width"] > 0, "Canvas width is zero"
        assert dims["height"] > 0, "Canvas height is zero"

    def test_fps_above_threshold(self, page):
        """Render loop runs above 30 FPS."""
        page.goto(BASE_URL, wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT)

        # Wait for the render loop to start and stabilise.
        page.wait_for_timeout(2000)

        # Measure FPS by counting requestAnimationFrame callbacks over 1s.
        fps = page.evaluate("""() => new Promise(resolve => {
            let count = 0;
            const start = performance.now();
            function tick() {
                count++;
                if (performance.now() - start < 1000) {
                    requestAnimationFrame(tick);
                } else {
                    resolve(count);
                }
            }
            requestAnimationFrame(tick);
        })""")

        assert fps > 30, f"FPS ({fps}) is below the 30 FPS threshold"

    def test_websocket_connects(self, page):
        """WebSocket connection is established within a few seconds."""
        page.goto(BASE_URL, wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT)

        # The global `wsConnected` is set to true in websocket.js on ws.onopen.
        # Poll for up to 5 seconds.
        connected = False
        for _ in range(25):
            connected = page.evaluate("() => typeof wsConnected !== 'undefined' && wsConnected === true")
            if connected:
                break
            page.wait_for_timeout(200)

        assert connected, "WebSocket did not connect within 5 seconds"

    def test_keyboard_shortcuts(self, page):
        """Pressing 'Q' toggles the queue panel visibility."""
        page.goto(BASE_URL, wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT)
        page.wait_for_timeout(500)

        # Queue panel starts hidden (has class 'hidden').
        is_hidden = page.evaluate(
            "() => document.getElementById('queuePanel').classList.contains('hidden')"
        )
        assert is_hidden, "Queue panel should start hidden"

        # Press 'Q' to open the queue panel.
        page.keyboard.press("q")
        page.wait_for_timeout(300)

        is_hidden_after_open = page.evaluate(
            "() => document.getElementById('queuePanel').classList.contains('hidden')"
        )
        assert not is_hidden_after_open, "Queue panel should be visible after pressing Q"

        # Press 'Q' again to close it.
        page.keyboard.press("q")
        page.wait_for_timeout(300)

        is_hidden_after_close = page.evaluate(
            "() => document.getElementById('queuePanel').classList.contains('hidden')"
        )
        assert is_hidden_after_close, "Queue panel should be hidden after pressing Q again"

    def test_api_health_endpoint(self, page):
        """GET /api/health returns 200 with a status field."""
        resp = page.request.get(HEALTH_URL)

        assert resp.status == 200, f"Health endpoint returned {resp.status}"

        body = resp.json()
        assert "status" in body, f"Response missing 'status' key: {body}"
        assert body["status"] in ("idle", "running", "stuck"), (
            f"Unexpected status value: {body['status']}"
        )
