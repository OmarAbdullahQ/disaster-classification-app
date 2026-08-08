"""Download and extract AIDERv2 directly from Zenodo."""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests

RECORD_API = "https://zenodo.org/api/records/10891054"


def download(output: Path) -> None:
    output = output.resolve()
    if all((output / split).exists() for split in ("Train", "Val", "Test")):
        print(f"Dataset already exists at {output}; nothing downloaded.")
        return

    download_dir = output.parent / "aiderv2_downloads"
    extract_dir = output.parent / "aiderv2_extracted"
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(RECORD_API, timeout=60)
    response.raise_for_status()
    files = response.json()["files"]

    for item in files:
        filename = item["key"]
        if not filename.lower().endswith(".zip"):
            continue
        destination = download_dir / filename
        if not destination.exists():
            print("Downloading", filename)
            with requests.get(item["links"]["self"], stream=True, timeout=120) as stream:
                stream.raise_for_status()
                with destination.open("wb") as file_handle:
                    for chunk in stream.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
        print("Extracting", filename)
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(extract_dir)

    candidates = [
        path for path in extract_dir.rglob("*")
        if path.is_dir()
        and all((path / split).is_dir() for split in ("Train", "Val", "Test"))
    ]
    if not candidates:
        raise RuntimeError("Could not locate a directory containing Train, Val, and Test.")

    source_root = min(candidates, key=lambda path: len(path.parts))
    output.mkdir(parents=True, exist_ok=True)
    for split in ("Train", "Val", "Test"):
        destination = output / split
        if not destination.exists():
            shutil.copytree(source_root / split, destination)
    print("AIDERv2 is ready at", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    download(arguments.output)
