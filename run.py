from __future__ import annotations

"""
AuraNet Master Launcher
=======================
Single-command startup for the full AuraNet stack:

  python run.py

What it does
------------
  1. Pre-flight: verifies all model artifacts exist
  2. Starts uvicorn (FastAPI) in the background, logs to auranet_api.log
  3. Polls /health until the API is ready (≤ 15 s)
  4. Prints the 'System Online' banner
  5. Launches Streamlit in the foreground
  6. On Ctrl+C (or Streamlit exit): terminates both processes cleanly
"""

import atexit
import subprocess
import sys
import time
import os
import requests
from pathlib import Path

# ── Unicode + ANSI setup (Windows) ───────────────────────────────────────────
if sys.platform == "win32":
    os.system("")                                    # enable VT/ANSI in Win Console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

C_RESET  = "\033[0m"
C_CYAN   = "\033[96m"
C_GREEN  = "\033[92m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_DIM    = "\033[2m"
C_BOLD   = "\033[1m"

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
API_URL    = "http://localhost:8000"
LOG_PATH   = BASE_DIR / "auranet_api.log"

REQUIRED_ARTIFACTS = [
    "models/xgb_tuned.joblib",
    "models/scaler.joblib",
    "models/label_encoders.joblib",
    "models/feature_meta.joblib",
]

# Process handles kept at module scope so atexit can reach them
_api_proc : subprocess.Popen | None = None
_ui_proc  : subprocess.Popen | None = None
_log_fh   = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print(color: str, msg: str) -> None:
    print(f"{color}{msg}{C_RESET}", flush=True)


def _banner() -> None:
    lines = [
        "",
        f"  {'─' * 52}",
        f"  {'':>4}{C_BOLD}{C_CYAN}A  U  R  A  N  E  T{C_RESET}{C_CYAN}{'':>17}v 1.0{C_RESET}",
        f"  {C_DIM}{'':>4}Cyber Defense Command System{C_RESET}",
        f"  {C_DIM}{'':>4}NSL-KDD · XGBoost · Optuna · FastAPI · Streamlit{C_RESET}",
        f"  {'─' * 52}",
        "",
    ]
    for l in lines:
        print(l)


def _preflight() -> bool:
    _print(C_CYAN, "  [◈] Pre-flight checks ...")
    all_ok = True
    for rel in REQUIRED_ARTIFACTS:
        path = BASE_DIR / rel
        if path.exists():
            _print(C_DIM, f"      ✔  {rel}")
        else:
            _print(C_RED, f"      ✘  MISSING: {rel}")
            all_ok = False
    if not all_ok:
        _print(C_RED, "\n  [✘] Artifacts missing. Run the training pipeline first:")
        _print(C_YELLOW, "      python src/features.py")
        _print(C_YELLOW, "      python src/tune.py")
    return all_ok


def _wait_for_api(max_wait: float = 15.0, poll: float = 0.4) -> bool:
    deadline = time.monotonic() + max_wait
    dots     = 0
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            if r.status_code == 200 and r.json().get("model_loaded"):
                print()   # newline after the dot animation
                return True
        except Exception:
            pass
        symbol = ["·", "·", "·", "●"][dots % 4]
        print(f"\r  {C_DIM}[◌] API starting  {symbol}{C_RESET}   ", end="", flush=True)
        dots += 1
        time.sleep(poll)
    print()
    return False


def _online_banner() -> None:
    lines = [
        "",
        f"  {'─' * 52}",
        f"  {C_BOLD}{C_GREEN}  ✔  AURANET SYSTEM ONLINE{C_RESET}",
        f"  {'─' * 52}",
        f"  {C_CYAN}  FastAPI  {C_RESET}→  {C_GREEN}{API_URL}{C_RESET}",
        f"  {C_CYAN}  API docs {C_RESET}→  {C_GREEN}{API_URL}/docs{C_RESET}",
        f"  {C_CYAN}  Dashboard{C_RESET}→  {C_GREEN}http://localhost:8501{C_RESET}",
        f"  {C_DIM}  API logs  →  {LOG_PATH.name}{C_RESET}",
        f"  {'─' * 52}",
        f"  {C_DIM}  Press Ctrl+C to shut down all services{C_RESET}",
        "",
    ]
    for l in lines:
        print(l)


# ─────────────────────────────────────────────────────────────────────────────
# Process lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup() -> None:
    """Called by atexit — terminates both subprocesses."""
    global _api_proc, _ui_proc, _log_fh
    print(f"\n{C_YELLOW}  [◌] Shutting down AuraNet ...{C_RESET}", flush=True)
    for name, proc in [("Streamlit", _ui_proc), ("FastAPI", _api_proc)]:
        if proc is not None and proc.poll() is None:
            _print(C_DIM, f"      Stopping {name}  (pid {proc.pid}) ...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if _log_fh is not None:
        _log_fh.close()
    _print(C_GREEN, "  [✔] All processes stopped. Goodbye.\n")


atexit.register(_cleanup)


def _start_api() -> subprocess.Popen:
    global _log_fh
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.serve:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=_log_fh,
        stderr=_log_fh,
    )
    _print(C_DIM, f"      uvicorn started (pid {proc.pid}) → logs: {LOG_PATH.name}")
    return proc


def _start_ui() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(BASE_DIR / "app.py"),
        "--server.port", "8501",
        "--server.headless", "false",
    ]
    return subprocess.Popen(cmd, cwd=str(BASE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _api_proc, _ui_proc

    _banner()

    # ── 1. Pre-flight ─────────────────────────────────────────────────────────
    if not _preflight():
        sys.exit(1)
    print()

    # ── 2. Start API ──────────────────────────────────────────────────────────
    _print(C_CYAN, "  [◈] Launching FastAPI  (uvicorn) ...")
    _api_proc = _start_api()

    # ── 3. Wait for API ready ─────────────────────────────────────────────────
    if not _wait_for_api():
        _print(C_RED, "  [✘] API did not respond within 15 s.")
        _print(C_YELLOW, f"      Check {LOG_PATH.name} for errors.")
        _cleanup()
        sys.exit(1)

    _print(C_GREEN, f"  [✔] FastAPI ready at {API_URL}")

    # ── 4. System-online banner ───────────────────────────────────────────────
    _online_banner()

    # ── 5. Start UI (foreground — blocks until user exits) ───────────────────
    _print(C_CYAN, "  [◈] Launching Streamlit dashboard ...")
    _ui_proc = _start_ui()

    try:
        _ui_proc.wait()
    except KeyboardInterrupt:
        pass   # atexit _cleanup() will handle teardown


if __name__ == "__main__":
    main()
