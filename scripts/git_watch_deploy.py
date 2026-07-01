"""Poll GitHub; on new commits run git pull + restart app (no ngrok needed for friend)."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 228
LOCK = ROOT / "tmp" / "git_watch_deploy.lock"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def _git() -> str:
    for candidate in ("git", r"C:\Program Files\Git\bin\git.exe"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    return "git"


def _rev(git_exe: str, ref: str) -> str:
    p = subprocess.run(
        [git_exe, "rev-parse", ref],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def _fetch(git_exe: str) -> bool:
    p = subprocess.run(
        [git_exe, "fetch", "origin", "main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.returncode == 0


def _pull(git_exe: str) -> tuple[bool, str]:
    p = subprocess.run(
        [git_exe, "pull", "--ff-only", "origin", "main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, out.strip()


def _restart_app() -> None:
    helper = ROOT / "scripts" / "restart_after_deploy.py"
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, str(helper)],
        cwd=str(ROOT),
        creationflags=flags,
        close_fds=True,
    )


def _server_busy() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=3) as r:
            import json

            data = json.loads(r.read().decode("utf-8"))
            return data.get("status") == "running"
    except Exception:
        return False


def main() -> int:
    _load_dotenv()
    interval = int(os.environ.get("GIT_AUTO_PULL_SECONDS", "90"))
    if interval < 30:
        interval = 30

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            age = time.time() - LOCK.stat().st_mtime
            if age < interval * 2:
                print("[git-watch] already running, exit")
                return 0
        except OSError:
            pass
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    git_exe = _git()
    branch = os.environ.get("GIT_AUTO_PULL_BRANCH", "main")
    print(f"[git-watch] poll every {interval}s | branch={branch}")

    try:
        while True:
            try:
                if _fetch(git_exe):
                    local = _rev(git_exe, "HEAD")
                    remote = _rev(git_exe, f"origin/{branch}")
                    if remote and local and local != remote:
                        if _server_busy():
                            print("[git-watch] training busy, skip this round")
                        else:
                            print(f"[git-watch] new commit {local[:8]} -> {remote[:8]}")
                            ok, msg = _pull(git_exe)
                            print(f"[git-watch] pull {'OK' if ok else 'FAIL'}: {msg[:200]}")
                            if ok:
                                print("[git-watch] scheduling app restart...")
                                _restart_app()
                                time.sleep(15)
            except Exception as ex:
                print(f"[git-watch] error: {ex}")

            time.sleep(interval)
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
