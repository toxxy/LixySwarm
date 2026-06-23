"""Robust aggregation of independently produced gradient candidates."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifact_store import ArtifactStore, digest_file
from .training_worker import TrainingWorkError, validate_gradient_artifact


_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GradientCandidate:
    artifact_id: str
    peer_id: str
    path: str | Path
    receipt: dict | None = None


class GradientAggregator:
    """Create an unapplied coordinate-median gradient artifact."""

    def __init__(self, store: ArtifactStore, *, chunk_values: int = 1_000_000):
        self.store = store
        self.chunk_values = max(1024, min(int(chunk_values), 10_000_000))

    def aggregate(
        self,
        candidates: list[GradientCandidate],
        model,
        *,
        model_artifact_id: str,
        dataset_artifact_id: str,
        start_token: int,
        token_count: int,
    ) -> tuple[object, dict]:
        if not 3 <= len(candidates) <= 31:
            raise TrainingWorkError("gradient quorum must contain 3 to 31 candidates")
        peer_ids = {candidate.peer_id for candidate in candidates}
        if len(peer_ids) != len(candidates):
            raise TrainingWorkError("gradient quorum requires distinct peer identities")
        for candidate in candidates:
            if not _HEX_64_RE.fullmatch(candidate.peer_id):
                raise TrainingWorkError("gradient candidate peer identity is invalid")
            if not _HEX_64_RE.fullmatch(candidate.artifact_id):
                raise TrainingWorkError("gradient candidate artifact ID is invalid")
            actual_id, _ = digest_file(candidate.path)
            if actual_id != candidate.artifact_id:
                raise TrainingWorkError("gradient candidate content hash does not match")
            validate_gradient_artifact(
                candidate.path,
                model,
                expected_model_id=model_artifact_id,
                expected_dataset_id=dataset_artifact_id,
                expected_start_token=start_token,
                expected_token_count=token_count,
            )

        parameter_names = self._common_parameter_names(candidates)
        work_dir = self.store.incoming / (
            f"aggregate.{os.getpid()}.{os.urandom(8).hex()}"
        )
        archive_path = self.store.incoming / (
            f"aggregate.{os.getpid()}.{os.urandom(8).hex()}.npz.tmp"
        )
        work_dir.mkdir(mode=0o700)
        metadata = {
            "format": 2,
            "aggregation": "coordinate_median",
            "model_artifact_id": model_artifact_id,
            "dataset_artifact_id": dataset_artifact_id,
            "start_token": int(start_token),
            "token_count": int(token_count),
            "quorum": len(candidates),
            "candidates": [
                {
                    "artifact_id": item.artifact_id,
                    "peer_id": item.peer_id,
                    "receipt": item.receipt,
                }
                for item in sorted(candidates, key=lambda item: item.peer_id)
            ],
            "applied": False,
        }
        try:
            with zipfile.ZipFile(
                archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as output_archive:
                for parameter_name in parameter_names:
                    extracted = self._extract_parameter(
                        candidates, parameter_name, work_dir
                    )
                    arrays = [
                        np.load(path, mmap_mode="r", allow_pickle=False)
                        for path in extracted
                    ]
                    output_path = work_dir / f"aggregate-{len(extracted)}.npy"
                    output = np.lib.format.open_memmap(
                        output_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=arrays[0].shape,
                    )
                    flat_output = output.reshape(-1)
                    for offset in range(0, flat_output.size, self.chunk_values):
                        end = min(flat_output.size, offset + self.chunk_values)
                        values = np.stack(
                            [array.reshape(-1)[offset:end] for array in arrays],
                            axis=0,
                        )
                        flat_output[offset:end] = np.median(values, axis=0).astype(
                            np.float32, copy=False
                        )
                    output.flush()
                    del output, arrays
                    output_archive.write(output_path, arcname=f"{parameter_name}.npy")
                    for path in extracted:
                        path.unlink(missing_ok=True)
                    output_path.unlink(missing_ok=True)

                metadata_path = work_dir / "metadata.npy"
                encoded = json.dumps(
                    metadata, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                np.save(
                    metadata_path,
                    np.frombuffer(encoded, dtype=np.uint8),
                    allow_pickle=False,
                )
                output_archive.write(
                    metadata_path, arcname="__lixy_metadata__.npy"
                )
            manifest = self.store.commit_generated(
                archive_path,
                kind="gradient",
                media_type="application/x-lixy-gradient-quorum+npz",
            )
            return manifest, metadata
        finally:
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _common_parameter_names(candidates: list[GradientCandidate]) -> list[str]:
        expected = None
        for candidate in candidates:
            with zipfile.ZipFile(candidate.path) as archive:
                names = {
                    info.filename[:-4]
                    for info in archive.infolist()
                    if info.filename.endswith(".npy")
                    and info.filename != "__lixy_metadata__.npy"
                }
            if expected is None:
                expected = names
            elif names != expected:
                raise TrainingWorkError("gradient candidates contain different parameters")
        if not expected:
            raise TrainingWorkError("gradient candidates contain no common parameters")
        return sorted(expected)

    @staticmethod
    def _extract_parameter(
        candidates: list[GradientCandidate],
        parameter_name: str,
        work_dir: Path,
    ) -> list[Path]:
        paths = []
        entry_name = f"{parameter_name}.npy"
        for index, candidate in enumerate(candidates):
            destination = work_dir / f"candidate-{index}.npy"
            with zipfile.ZipFile(candidate.path) as archive:
                info = archive.getinfo(entry_name)
                with archive.open(info) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            paths.append(destination)
        return paths
