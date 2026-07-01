"""Detached helper: stop port 228 and start app.py (optionally --public)."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 228


def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def _kill_port(port: int) -> None:
    if os.name != "nt":
        return
    out = subprocess.run(
        f"netstat -ano | findstr :{port} | findstr LISTENING",
        shell=True,
        capture_output=True,
        text=True,
    )
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            subprocess.run(
                ["taskkill", "/PID", parts[-1], "/F"],
                capture_output=True,
            )


def main() -> int:
    _load_dotenv()
    time.sleep(1.5)
    _kill_port(PORT)
    time.sleep(0.5)

    args = [sys.executable, "-u", str(ROOT / "app.py")]
    if os.environ.get("APP_PUBLIC", "").lower() in ("1", "true", "yes"):
        args.append("--public")

    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        args,
        cwd=str(ROOT),
        creationflags=flags,
        close_fds=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
