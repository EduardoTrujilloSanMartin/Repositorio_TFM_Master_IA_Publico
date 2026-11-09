#!/usr/bin/env python3
"""Descarga reproducible de los originales HAM10000 desde Harvard Dataverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PERSISTENT_ID = "doi:10.7910/DVN/DBW86T"
VERSION = "1.0"
API_BASE = "https://dataverse.harvard.edu/api"
DATASET_API_URL = (
    f"{API_BASE}/datasets/:persistentId/versions/{VERSION}"
    f"?persistentId={PERSISTENT_ID}"
)
REQUIRED_FILENAMES = {
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
    "HAM10000_metadata.tab",
}
MINIMUM_FREE_BYTES = 12 * 1024**3


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(
    session: requests.Session,
    url: str,
    destination: Path,
    expected_size: int | None = None,
) -> None:
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        print(f"Ya existe: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
    response.raise_for_status()

    append = offset > 0 and response.status_code == 206
    if offset and not append:
        partial.unlink(missing_ok=True)
        offset = 0
    mode = "ab" if append else "wb"
    total = expected_size or 0
    written = offset
    next_report = 0
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
            if total:
                percentage = int(written * 100 / total)
                if percentage >= next_report:
                    print(f"  {destination.name}: {percentage:3d}%", flush=True)
                    next_report = percentage + 5
    if expected_size is not None and written != expected_size:
        raise RuntimeError(
            f"Tamaño inesperado para {destination.name}: {written} != {expected_size}"
        )
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/ham10000_v1"),
        help="Directorio donde conservar los originales.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    archives = root / "archives"
    provenance = root / "provenance"
    archives.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"Espacio insuficiente: {free_bytes / 1024**3:.1f} GiB libres; "
            f"se requieren al menos {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB."
        )

    session = requests.Session()
    session.headers.update({"User-Agent": "TFM-HAM10000-audit/1.0"})
    response = session.get(DATASET_API_URL, timeout=(30, 120))
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"Respuesta inesperada de Dataverse: {payload.get('status')}")
    version = payload["data"]
    if (version.get("versionNumber"), version.get("versionMinorNumber")) != (1, 0):
        raise RuntimeError("La API no devolvió la versión 1.0 solicitada.")

    (provenance / "dataverse_dataset_v1.0.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    files = {item["dataFile"]["filename"]: item["dataFile"] for item in version["files"]}
    if set(files) != REQUIRED_FILENAMES:
        raise RuntimeError(f"Contenido inesperado en la versión 1.0: {sorted(files)}")

    downloaded: list[dict[str, Any]] = []
    for filename in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
        info = files[filename]
        file_id = int(info["id"])
        target = archives / filename
        download(session, f"{API_BASE}/access/datafile/{file_id}", target, int(info["filesize"]))
        observed_md5 = digest(target, "md5")
        expected_md5 = info["checksum"]["value"].lower()
        if observed_md5 != expected_md5:
            raise RuntimeError(f"MD5 incorrecto para {filename}: {observed_md5} != {expected_md5}")
        downloaded.append(
            {
                "filename": filename,
                "file_id": file_id,
                "size_bytes": target.stat().st_size,
                "official_md5": expected_md5,
                "observed_md5": observed_md5,
                "sha256": digest(target, "sha256"),
                "download_url": f"{API_BASE}/access/datafile/{file_id}",
            }
        )

    metadata_info = files["HAM10000_metadata.tab"]
    tab_target = archives / "HAM10000_metadata.tab"
    download(
        session,
        f"{API_BASE}/access/datafile/{metadata_info['id']}",
        tab_target,
        int(metadata_info["filesize"]),
    )
    downloaded.append(
        {
            "filename": tab_target.name,
            "file_id": int(metadata_info["id"]),
            "size_bytes": tab_target.stat().st_size,
            "official_unf": metadata_info.get("UNF"),
            "official_md5": None,
            "observed_md5": digest(tab_target, "md5"),
            "sha256": digest(tab_target, "sha256"),
            "download_url": f"{API_BASE}/access/datafile/{metadata_info['id']}",
            "note": (
                "TAB derivado por Dataverse al ingerir el CSV. La huella MD5 publicada "
                "corresponde al CSV original, no a esta representación tabular."
            ),
        }
    )
    original_name = metadata_info.get("originalFileName", "HAM10000_metadata.csv")
    csv_target = archives / "HAM10000_metadata.csv"
    download(
        session,
        f"{API_BASE}/access/datafile/{metadata_info['id']}?format=original",
        csv_target,
        int(metadata_info["originalFileSize"]),
    )
    original_md5 = digest(csv_target, "md5")
    expected_original_md5 = metadata_info["checksum"]["value"].lower()
    if original_md5 != expected_original_md5:
        raise RuntimeError(
            f"MD5 incorrecto para {csv_target.name}: "
            f"{original_md5} != {expected_original_md5}"
        )
    downloaded.append(
        {
            "filename": csv_target.name,
            "official_original_filename": original_name,
            "file_id": int(metadata_info["id"]),
            "size_bytes": csv_target.stat().st_size,
            "official_md5": expected_original_md5,
            "observed_md5": original_md5,
            "sha256": digest(csv_target, "sha256"),
            "download_url": f"{API_BASE}/access/datafile/{metadata_info['id']}?format=original",
            "note": "CSV original depositado; coincide con la huella MD5 publicada.",
        }
    )

    downloaded_at = datetime.now(timezone.utc)
    record = {
        "dataset": "HAM10000",
        "title": version["metadataBlocks"]["citation"]["fields"][0]["value"],
        "persistent_id": PERSISTENT_ID,
        "persistent_url": "https://doi.org/10.7910/DVN/DBW86T",
        "publisher": "Harvard Dataverse",
        "dataset_version": "1.0",
        "version_release_time": version["releaseTime"],
        "dataset_unf": version.get("UNF"),
        "license": "Creative Commons Attribution-NonCommercial 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "downloaded_at_utc": downloaded_at.isoformat(),
        "download_date_local": downloaded_at.astimezone().date().isoformat(),
        "files": downloaded,
        "originals_read_only": True,
    }
    (provenance / "download_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (provenance / "TERMS_OF_USE.html").write_text(
        version.get("termsOfUse", ""), encoding="utf-8"
    )

    for path in archives.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
