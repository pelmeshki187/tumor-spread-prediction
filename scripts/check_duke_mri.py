"""Check two downloaded DICOM series without extracting archives or requiring a GPU."""

import argparse
from array import array
import json
import math
from pathlib import Path
import statistics
import sys
from zipfile import ZipFile

import pydicom

from download_duke import PRE_UID, POST_UID


def check_series(path, expected_uid):
    instances, positions, sop_uids, matrices = [], [], [], set()
    normals, pixel_range, geometries = [], None, set()
    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".dcm")]
        for name in names:
            with archive.open(name) as source:
                image = pydicom.dcmread(source)
            if image.PatientID != "Breast_MRI_001" or image.SeriesInstanceUID != expected_uid:
                raise ValueError("Wrong patient or series in archive")
            if image.Modality != "MR" or image.file_meta.TransferSyntaxUID.is_compressed:
                raise ValueError("Expected uncompressed MR images")
            if image.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.2":
                raise ValueError("Expected little-endian sample")
            expected_bytes = image.Rows * image.Columns * image.BitsAllocated // 8
            if len(image.PixelData) != expected_bytes:
                raise ValueError("Unexpected pixel payload length")
            if pixel_range is None:
                if image.BitsAllocated != 16 or image.PixelRepresentation != 0:
                    raise ValueError("Expected unsigned 16-bit sample")
                pixels = array("H", image.PixelData)
                if sys.byteorder != "little":
                    pixels.byteswap()
                pixel_range = [min(pixels), max(pixels)]
                if pixel_range[0] == pixel_range[1]:
                    raise ValueError("Constant first sample image")
            orientation = list(map(float, image.ImageOrientationPatient))
            geometries.add(tuple(round(v, 6) for v in orientation + list(map(float, image.PixelSpacing))))
            a, b = orientation[:3], orientation[3:]
            normal = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
            normals.append(tuple(round(v, 6) for v in normal))
            positions.append(tuple(map(float, image.ImagePositionPatient)))
            instances.append(int(image.InstanceNumber))
            sop_uids.append(str(image.SOPInstanceUID))
            matrices.add((image.Rows, image.Columns))
    if not names or len(sop_uids) != len(set(sop_uids)):
        raise ValueError("Empty or duplicate DICOM objects")
    if sorted(instances) != list(range(1, len(instances) + 1)):
        raise ValueError("Non-contiguous or duplicate InstanceNumber")
    if len(set(normals)) != 1 or len(matrices) != 1 or len(geometries) != 1:
        raise ValueError("Inconsistent image geometry")
    coordinates = sorted(sum(p*n for p, n in zip(position, normals[0])) for position in positions)
    spacings = [b-a for a, b in zip(coordinates, coordinates[1:])]
    if not spacings or min(spacings) <= 0 or not math.isclose(min(spacings), max(spacings), abs_tol=0.001):
        raise ValueError("Duplicate positions or irregular slice spacing")
    return {"patient_id": "Breast_MRI_001", "series_uid": expected_uid,
        "dicom_instances": len(instances), "matrix": list(next(iter(matrices))),
        "median_slice_spacing_mm": statistics.median(spacings),
        "orientation_and_pixel_spacing": list(next(iter(geometries))),
        "first_file_pixel_range": pixel_range,
        "positions": sorted(tuple(round(v, 4) for v in pos) for pos in positions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/duke"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/duke"))
    args = parser.parse_args()
    pre = check_series(args.data_dir / "sample-pre.zip", PRE_UID)
    post = check_series(args.data_dir / "sample-post1.zip", POST_UID)
    geometry_matches = pre.pop("positions") == post.pop("positions")
    if (not geometry_matches or pre["matrix"] != post["matrix"]
            or pre["orientation_and_pixel_spacing"] != post["orientation_and_pixel_spacing"]):
        raise ValueError("Pre/post geometry does not match")
    summary = {"pre": pre, "first_post": post, "geometry_matches": geometry_matches,
        "scope": "one patient; no annotation overlay or cohort-wide quality guarantee"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mri_sample.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
