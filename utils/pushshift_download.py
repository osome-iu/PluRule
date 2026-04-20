"""
Shared helpers for fetching the Pushshift/Arctic Shift subreddit torrent.

Used by:
  - pipeline/0_download_data.py  (full torrent → PUSHSHIFT_DATA)
  - hydrate/0_download.py        (subset referenced by dehydrated dataset)

Normalises on-disk layout to `<root>/<first-letter>/<Subreddit>_{comments,submissions}.zst`
regardless of where the files came from (torrent native shards or a pre-existing
local mirror), so downstream pipeline stages can look up files consistently via
`<PUSHSHIFT_DATA>/<letter>/<sub>_comments.zst`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import requests
    import torf
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        f"utils.pushshift_download requires `requests`, `torf`, and `tqdm` "
        f"(missing: {e.name}). Install: pip install -r requirements-hydrate.txt"
    )


PUSHSHIFT_INFOHASH = "3e3f64dee22dc304cdd2546254ca1f8e8ae542b4"
PUSHSHIFT_TORRENT_URL = (
    f"https://academictorrents.com/download/{PUSHSHIFT_INFOHASH}.torrent"
)


# ---------------------------------------------------------------------------
# Torrent metadata
# ---------------------------------------------------------------------------

def _decode(x):
    """bencode values come back as bytes; normalise to str."""
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x


def fetch_torrent(url: str, dest: Path) -> None:
    """Download .torrent metadata to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1 << 16):
            f.write(chunk)


def ensure_torrent(torrent_path: Path, fallback_url: str = PUSHSHIFT_TORRENT_URL) -> Path:
    """Return a usable .torrent path. Fetches from `fallback_url` if missing."""
    if torrent_path.exists() and torrent_path.stat().st_size > 0:
        print(f"📄 Using cached torrent: {torrent_path}")
        return torrent_path
    print(f"📥 Fetching torrent from {fallback_url}")
    fetch_torrent(fallback_url, torrent_path)
    return torrent_path


def parse_torrent(torrent_path: Path) -> List[Tuple[int, str, str]]:
    """
    Return [(1-based index, basename_lower, relative_path)] for every file in the torrent.

    Uses raw metainfo iteration (orders of magnitude faster than torf's File tree).
    """
    print("   parsing torrent metadata...")
    t = torf.Torrent.read(str(torrent_path))
    info = t.metainfo.get("info", {})
    files_meta = info.get("files")

    out: List[Tuple[int, str, str]] = []

    if files_meta is None:
        # Single-file torrent
        basename = _decode(info.get("name", ""))
        if basename:
            out.append((1, basename.lower(), basename))
        return out

    print(f"   torrent contains {len(files_meta):,} file entries")
    for idx, f in enumerate(files_meta, start=1):
        path_parts = f.get("path") or f.get(b"path") or []
        if not path_parts:
            continue
        rel_path = "/".join(_decode(p) for p in path_parts)
        basename_lower = _decode(path_parts[-1]).lower()
        out.append((idx, basename_lower, rel_path))
    return out


def match_basenames(
    all_files: Iterable[Tuple[int, str, str]],
    needed: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Tuple[int, str]], Set[str]]:
    """
    Filter parsed torrent entries by a `needed` basename set (all lowercase).

    If `needed` is None, returns every entry (full-torrent mode).
    Returns (basename_lower -> (index, rel_path), missing-needed set).
    """
    matched: Dict[str, Tuple[int, str]] = {}
    for idx, basename_lower, rel_path in all_files:
        if needed is None or basename_lower in needed:
            matched[basename_lower] = (idx, rel_path)
    missing = set() if needed is None else (needed - set(matched.keys()))
    return matched, missing


# ---------------------------------------------------------------------------
# aria2c driver
# ---------------------------------------------------------------------------

def check_aria2c() -> None:
    """Abort with install instructions if aria2c isn't on PATH."""
    if shutil.which("aria2c") is None:
        sys.exit(
            "aria2c not found. Install it first:\n"
            "  conda (no root):        conda install -c conda-forge aria2\n"
            "  Debian/Ubuntu:          sudo apt install aria2\n"
            "  macOS:                  brew install aria2\n"
            "  Fedora/CentOS:          sudo dnf install aria2\n"
            "Or download the .torrent manually in any BitTorrent client that supports "
            "file selection (qBittorrent, Transmission)."
        )


def run_aria2c(
    torrent_path: Path,
    indices: Optional[List[int]],
    output_dir: Path,
) -> int:
    """
    Invoke aria2c. Returns exit code.

    If `indices` is None → download all files in the torrent.
    Else → pass `--select-file=i,j,...` (1-based).
    """
    cmd = [
        "aria2c",
        f"--torrent-file={torrent_path}",
        f"--dir={output_dir}",
        "--seed-time=0",
        "--max-connection-per-server=16",
        "--split=16",
        "--continue=true",
        "--file-allocation=none",
        "--bt-save-metadata=false",
        "--bt-remove-unselected-file=true",
        "--console-log-level=warn",
        "--summary-interval=30",
    ]
    if indices is not None:
        cmd.insert(2, f"--select-file=" + ",".join(str(i) for i in sorted(indices)))
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Layout: first-letter buckets
# ---------------------------------------------------------------------------

def _first_bucket_for(sub_or_basename: str) -> str:
    """First-letter bucket for a subreddit or basename. '_' for non-alphanumeric."""
    head = sub_or_basename.split("_", 1)[0]
    if not head:
        return "_"
    c = head[0].lower()
    return c if c.isalnum() else "_"


def reorganize_to_letter_buckets(
    output_dir: Path,
    basename_to_current_path: Dict[str, str],
    cleanup_empty_dirs: bool = True,
) -> Dict[str, str]:
    """
    Move files into `<output_dir>/<first-letter>/<filename>` layout.
    Same-filesystem renames are instantaneous. Returns updated map.
    """
    output_dir = Path(output_dir)
    new_map: Dict[str, str] = {}
    moved = already = 0

    for basename_lower, current_path in basename_to_current_path.items():
        current = Path(current_path)
        target_dir = output_dir / _first_bucket_for(basename_lower)
        target_dir.mkdir(exist_ok=True)
        # Preserve original filename casing.
        target = target_dir / current.name

        try:
            same = current.resolve() == target.resolve()
        except FileNotFoundError:
            same = False

        if same:
            new_map[basename_lower] = str(target)
            already += 1
            continue

        if target.exists():
            # Duplicate: prefer the target location, remove the stray.
            if current.exists():
                try:
                    current.unlink()
                except OSError:
                    pass
            new_map[basename_lower] = str(target)
            already += 1
            continue

        try:
            current.rename(target)
        except OSError:
            # Cross-filesystem fallback
            shutil.move(str(current), str(target))
        new_map[basename_lower] = str(target)
        moved += 1

    if cleanup_empty_dirs:
        for entry in os.scandir(output_dir):
            if not entry.is_dir(follow_symlinks=False):
                continue
            if len(entry.name) == 1:  # keep letter buckets
                continue
            try:
                os.rmdir(entry.path)  # non-empty dirs raise OSError; we ignore
            except OSError:
                pass

    print(f"   reorganized: {moved} moved, {already} already in place")
    return new_map


def scan_local_files(
    source_dir: Path,
    needed: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Enumerate files in a local mirror.

    Fast path: first-letter bucket layout → ~36 `os.scandir` calls.
    Falls back to full `os.walk` if no letter buckets detected.

    Returns (present_basenames, missing_basenames, basename_lower -> abs path).
    If `needed` is None → every file found; `missing` is empty.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return [], sorted(needed or []), {}

    try:
        top_dirs = [e for e in os.scandir(source_dir) if e.is_dir(follow_symlinks=False)]
    except OSError as e:
        raise RuntimeError(f"Cannot scan {source_dir}: {e}")

    letter_buckets = [e for e in top_dirs if len(e.name) == 1]
    basename_to_path: Dict[str, str] = {}

    if letter_buckets:
        print(f"   detected first-letter bucket layout ({len(letter_buckets)} buckets)")
        try:
            bucket_iter = tqdm(letter_buckets, desc="   scanning buckets", unit="bucket")
        except Exception:
            bucket_iter = letter_buckets
        for bucket in bucket_iter:
            try:
                with os.scandir(bucket.path) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            basename_to_path[entry.name.lower()] = entry.path
            except OSError:
                continue
    else:
        print("   no bucket layout detected; full recursive walk...")
        for root, _, files in os.walk(source_dir):
            for f in files:
                basename_to_path[f.lower()] = os.path.join(root, f)

    if needed is None:
        return sorted(basename_to_path.keys()), [], basename_to_path

    present = [b for b in needed if b in basename_to_path]
    missing = [b for b in needed if b not in basename_to_path]
    kept = {b: basename_to_path[b] for b in present}
    return present, missing, kept
