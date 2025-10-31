#!/usr/bin/env python3
"""Fetch the most recent release artifact and atomically refresh local weights."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib import request
from urllib.error import HTTPError

API_ROOT = "https://api.github.com"
LOGGER = logging.getLogger("pull_latest")


def _github_request(url: str, token: Optional[str] = None) -> dict:
    req = request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with request.urlopen(req) as response:
            payload = response.read()
    except HTTPError as exc:  # noqa: BLE001
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"GitHub API error {exc.code}: {body}") from exc
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))


def _download_asset(url: str, token: Optional[str], output: Path) -> Path:
    req = request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with request.urlopen(req) as response, output.open("wb") as handle:  # type: ignore[arg-type]
        shutil.copyfileobj(response, handle)
    return output


def _find_asset(release: dict, pattern: str) -> Optional[dict]:
    for asset in release.get("assets", []):
        if fnmatch.fnmatch(asset.get("name", ""), pattern):
            return asset
    return None


def atomic_swap(new_file: Path, dest: Path, keep_backup: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_dest.exists():
        tmp_dest.unlink()
    new_file.replace(tmp_dest)
    backup = dest.with_suffix(dest.suffix + ".bak")
    if dest.exists():
        if keep_backup:
            if backup.exists():
                backup.unlink()
            dest.replace(backup)
        else:
            dest.unlink()
    tmp_dest.replace(dest)
    return backup if keep_backup else dest


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull the latest release weights from GitHub")
    parser.add_argument("--repo", required=True, help="owner/repo to read releases from")
    parser.add_argument(
        "--asset-pattern",
        default="*.pt",
        help="fnmatch pattern that selects the weight artifact to download",
    )
    parser.add_argument("--dest", required=True, type=Path, help="Destination path for the weight file")
    parser.add_argument(
        "--tag", help="Optional release tag to pin; defaults to the most recent release"
    )
    parser.add_argument(
        "--token-env",
        default="GH_TOKEN,GITHUB_TOKEN",
        help="Comma-separated list of environment variable names to read for an API token",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which asset would be downloaded without modifying local files",
    )
    return parser.parse_args(argv)


def resolve_token(names: str) -> Optional[str]:
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        value = os.getenv(name)
        if value:
            return value
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = resolve_token(args.token_env)
    tag = args.tag or "latest"
    release_url = (
        f"{API_ROOT}/repos/{args.repo}/releases/tags/{args.tag}" if args.tag else f"{API_ROOT}/repos/{args.repo}/releases/latest"
    )
    LOGGER.info("Fetching release metadata (%s)", tag)
    release = _github_request(release_url, token)
    if not release:
        raise SystemExit(f"No release found for {tag}")
    asset = _find_asset(release, args.asset_pattern)
    if asset is None:
        raise SystemExit(
            f"No asset matching pattern '{args.asset_pattern}' in release {release.get('name') or tag}"
        )
    browser_url = asset.get("browser_download_url")
    if not browser_url:
        raise SystemExit("Asset is missing download URL")
    LOGGER.info("Latest asset: %s (%d bytes)", asset.get("name"), asset.get("size", 0))
    if args.dry_run:
        LOGGER.info("Dry run enabled; skipping download")
        return 0
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download_asset(browser_url, token, tmp_path)
        LOGGER.info("Downloaded to %s", tmp_path)
        backup = atomic_swap(tmp_path, args.dest)
        if backup != args.dest:
            LOGGER.info("Previous weights backed up to %s", backup)
        LOGGER.info("Installed new weights at %s", args.dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return 0
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
