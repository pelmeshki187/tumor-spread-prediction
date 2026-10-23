"""Download a small, fixed subset of the official Duke/TCIA release."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile


TCIA = "https://www.cancerimagingarchive.net/wp-content/uploads/"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
PRE_UID = "1.3.6.1.4.1.14519.5.2.1.185777849803665244536713316058283493877"
POST_UID = "1.3.6.1.4.1.14519.5.2.1.15147203499230039585785258602759759529"
TABLES = {
    name: TCIA + name
    for name in (
        "Clinical_and_Other_Features.xlsx",
        "Annotation_Boxes.xlsx",
        "Imaging_Features.xlsx",
    )
}
SAMPLE = {
    "sample-series.json": API + "getSeries?" + urlencode({
        "Collection": "Duke-Breast-Cancer-MRI",
        "PatientID": "Breast_MRI_001", "format": "json",
    }),
    "sample-pre.zip": API + "getImage?" + urlencode({"SeriesInstanceUID": PRE_UID}),
    "sample-post1.zip": API + "getImage?" + urlencode({"SeriesInstanceUID": POST_UID}),
}


def validate(path, filename):
    if filename.endswith(".json"):
        rows = json.loads(path.read_text())
        if not rows or any(r["PatientID"] != "Breast_MRI_001" for r in rows):
            raise ValueError("Unexpected sample metadata")
    else:
        with ZipFile(path) as archive:
            if filename.endswith(".xlsx"):
                if "xl/workbook.xml" not in archive.namelist():
                    raise ValueError(f"Not an XLSX workbook: {filename}")
            elif not any(n.lower().endswith(".dcm") for n in archive.namelist()):
                raise ValueError(f"No DICOM instances: {filename}")
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt ZIP: {filename}")


def download(url, destination):
    if destination.exists():
        validate(destination, destination.name)
        print(f"Already present: {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "tumor-spread-prediction/0.1"})
        with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 100_000_000:
                    raise ValueError("Unexpected download size (>100 MB per file)")
                output.write(chunk)
        validate(temporary, destination.name)
        temporary.replace(destination)
        print(f"Downloaded: {destination.name} ({size:,} bytes)")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/duke"))
    parser.add_argument("--sample-mri", action="store_true",
                        help="Also fetch pre/first-post series of one fixed patient")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    sources = TABLES | SAMPLE if args.sample_mri else TABLES
    for name, url in sources.items():
        download(url, args.data_dir / name)


if __name__ == "__main__":
    main()
