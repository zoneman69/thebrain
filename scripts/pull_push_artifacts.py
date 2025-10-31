#!/usr/bin/env python
"""Utility helpers for moving training artifacts to/from GitHub releases."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request
from urllib.error import HTTPError

API_ROOT = "https://api.github.com"
UPLOAD_ROOT = "https://uploads.github.com"
logger = logging.getLogger("artifact_io")


def github_request(
    url: str,
    *,
    method: str = "GET",
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
) -> bytes:
    req = request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with request.urlopen(req) as response:
            return response.read()
    except HTTPError as exc:  # noqa: BLE001
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"GitHub API error {exc.code}: {body}") from exc


def github_json(**kwargs: Any) -> Dict[str, Any]:
    raw = github_request(**kwargs)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def get_release(repo: str, tag: Optional[str], token: Optional[str]) -> Optional[Dict[str, Any]]:
    if tag:
        url = f"{API_ROOT}/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"{API_ROOT}/repos/{repo}/releases/latest"
    try:
        return github_json(url=url, token=token)
    except RuntimeError as exc:
        if "404" in str(exc):
            return None
        raise


def ensure_release(repo: str, tag: str, token: Optional[str], *, title: Optional[str] = None) -> Dict[str, Any]:
    release = get_release(repo, tag, token)
    if release is not None:
        return release
    payload = json.dumps({"tag_name": tag, "name": title or tag, "draft": False, "prerelease": True}).encode("utf-8")
    return github_json(url=f"{API_ROOT}/repos/{repo}/releases", method="POST", token=token, data=payload)


def download_asset(repo: str, asset_pattern: str, output: Path, tag: Optional[str], token: Optional[str]) -> Path:
    release = get_release(repo, tag, token)
    if not release:
        raise RuntimeError(f"Release for tag '{tag or 'latest'}' not found")
    assets = release.get("assets", [])
    match = None
    for asset in assets:
        if fnmatch.fnmatch(asset["name"], asset_pattern):
            match = asset
            break
    if match is None:
        raise RuntimeError(f"No asset matching pattern '{asset_pattern}' in release {release.get('name')}")
    url = match["browser_download_url"]
    req = request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with request.urlopen(req) as response, output.open("wb") as handle:  # type: ignore[arg-type]
        shutil.copyfileobj(response, handle)
    logger.info("Downloaded %s to %s", match["name"], output)
    return output


def upload_asset(
    repo: str,
    file_path: Path,
    *,
    tag: str,
    name: Optional[str],
    token: Optional[str],
    title: Optional[str],
    content_type: str,
    overwrite: bool,
) -> Dict[str, Any]:
    release = ensure_release(repo, tag, token, title=title)
    release_id = release["id"]
    existing_assets = release.get("assets", [])
    asset_name = name or file_path.name
    for asset in existing_assets:
        if asset.get("name") == asset_name:
            if not overwrite:
                raise RuntimeError(
                    f"Asset '{asset_name}' already exists in release {release.get('name')} (use --overwrite)"
                )
            github_request(
                url=f"{API_ROOT}/repos/{repo}/releases/assets/{asset['id']}",
                method="DELETE",
                token=token,
            )
            break
    upload_url = f"{UPLOAD_ROOT}/repos/{repo}/releases/{release_id}/assets?name={asset_name}"
    with file_path.open("rb") as handle:
        data = handle.read()
    headers = {"Content-Type": content_type}
    response = github_json(url=upload_url, method="POST", token=token, headers=headers, data=data)
    logger.info("Uploaded %s (%d bytes) to release %s", asset_name, len(data), release.get("name"))
    return response


def parse_cli(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move weights between GitHub releases and local storage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download a matching asset from a release")
    dl.add_argument("--repo", required=True, help="owner/repo name")
    dl.add_argument("--asset", required=True, help="fnmatch pattern for the asset name")
    dl.add_argument("--output", required=True, help="Destination path for the downloaded file")
    dl.add_argument("--tag", help="Release tag (defaults to latest)")

    ul = subparsers.add_parser("upload", help="Upload an artifact to a release")
    ul.add_argument("--repo", required=True, help="owner/repo name")
    ul.add_argument("--file", required=True, help="Path to artifact file")
    ul.add_argument("--tag", required=True, help="Release tag to create/update")
    ul.add_argument("--name", help="Override asset name (defaults to file name)")
    ul.add_argument("--title", help="Release title when creating a new release")
    ul.add_argument("--content-type", default="application/octet-stream", help="MIME type for upload")
    ul.add_argument("--overwrite", action="store_true", help="Replace an existing asset with the same name")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_cli(argv)
    token = os.getenv("GITHUB_TOKEN")

    if args.command == "download":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        download_asset(args.repo, args.asset, output_path, args.tag, token)
    elif args.command == "upload":
        file_path = Path(args.file)
        if not file_path.exists():
            raise SystemExit(f"Artifact file '{file_path}' not found")
        upload_asset(
            args.repo,
            file_path,
            tag=args.tag,
            name=args.name,
            token=token,
            title=args.title,
            content_type=args.content_type,
            overwrite=args.overwrite,
        )
    else:  # pragma: no cover - argparse ensures we never hit this
        raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
