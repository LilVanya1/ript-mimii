"""Download and organize MIMII dataset from Zenodo with retry + resume."""

import time
import zipfile
import requests
import re
from pathlib import Path
from tqdm import tqdm

from src.config import DATA_DIR

MIMII_FILES = {
    "fan":    [
        "https://zenodo.org/record/3384388/files/6_dB_fan.zip",
        "https://zenodo.org/record/3384388/files/0_dB_fan.zip",
        "https://zenodo.org/record/3384388/files/-6_dB_fan.zip",
    ],
    "pump":   [
        "https://zenodo.org/record/3384388/files/6_dB_pump.zip",
        "https://zenodo.org/record/3384388/files/0_dB_pump.zip",
        "https://zenodo.org/record/3384388/files/-6_dB_pump.zip",
    ],
    "slider": [
        "https://zenodo.org/record/3384388/files/6_dB_slider.zip",
        "https://zenodo.org/record/3384388/files/0_dB_slider.zip",
        "https://zenodo.org/record/3384388/files/-6_dB_slider.zip",
    ],
    "valve":  [
        "https://zenodo.org/record/3384388/files/6_dB_valve.zip",
        "https://zenodo.org/record/3384388/files/0_dB_valve.zip",
        "https://zenodo.org/record/3384388/files/-6_dB_valve.zip",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}
CHUNK  = 1 << 17   # 128 KB
RETRIES = 5
BACKOFF = [5, 15, 30, 60, 120]


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            return bad is None
    except Exception:
        return False


def download_file(url: str, dest: Path, progress_cb=None) -> Path:
    """Download with resume and retry on ConnectionResetError / 5xx."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If existing file is corrupted, delete it and start fresh
    if dest.exists():
        if not zipfile.is_zipfile(dest):
            print(f"  Corrupt file detected, deleting: {dest.name}")
            dest.unlink()

    existing = dest.stat().st_size if dest.exists() else 0

    for attempt in range(RETRIES):
        headers = dict(HEADERS)
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=120)
            if resp.status_code == 416:          # range not satisfiable → already complete
                print(f"  Already downloaded: {dest.name}")
                return dest
            if resp.status_code not in (200, 206):
                resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0)) + existing
            mode  = "ab" if resp.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0

            with open(dest, mode) as f, tqdm(
                total=total, initial=existing,
                unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    f.write(chunk)
                    existing += len(chunk)
                    pbar.update(len(chunk))
                    if progress_cb:
                        progress_cb(existing, total)
            return dest

        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as exc:
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            print(f"  [{dest.name}] Attempt {attempt+1}/{RETRIES} failed: {exc}. "
                  f"Retrying in {wait}s…")
            existing = dest.stat().st_size if dest.exists() else 0
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {RETRIES} attempts")


def extract_zip(zip_path: Path, extract_to: Path):
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupted zip deleted: {zip_path.name} — re-run to re-download")
    print(f"  Extracting: {zip_path.name}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _parse_snr_machine_from_name(fname: str) -> tuple[int, str] | None:
    m = re.match(r"^(-?\d+)_dB_([a-zA-Z0-9_]+)\.zip$", fname)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def download_mimii(machine_types: list[str] | None = None, progress_cb=None):
    if machine_types is None:
        machine_types = list(MIMII_FILES.keys())

    for mt in machine_types:
        for url in MIMII_FILES[mt]:
            fname    = url.split("/")[-1]
            zip_path = DATA_DIR / "zips" / fname
            marker   = DATA_DIR / f".extracted_{fname}"
            parsed   = _parse_snr_machine_from_name(fname)
            # Keep each SNR in a dedicated folder to avoid overwriting
            # fan/id_xx/.. across -6/0/6 dB archives.
            if parsed:
                snr_db, parsed_mt = parsed
                if parsed_mt != mt:
                    parsed_mt = mt
                extract_root = DATA_DIR / f"{snr_db}_dB_{parsed_mt}"
            else:
                # Fallback for unexpected naming.
                extract_root = DATA_DIR

            if not marker.exists():
                print(f"  Downloading: {fname}")
                download_file(url, zip_path, progress_cb=progress_cb)
                extract_zip(zip_path, extract_root)
                marker.touch()
            else:
                print(f"  Already extracted: {fname}")

    print("Download complete.")


if __name__ == "__main__":
    import sys
    download_mimii(sys.argv[1:] or None)
