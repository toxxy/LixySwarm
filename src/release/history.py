"""Bounded signed release-chain discovery from consenting artifact peers."""
from __future__ import annotations

import re
import time

from src.contribution import ResourceRequirements
from src.network.work_protocol import ResultReceipt
from .manifest import ReleaseError, ReleaseManifest

HISTORY_OPERATION = "release.history.v1"
_IDENTIFIER = re.compile(r"^[0-9a-f]{64}$")


class ReleaseHistoryService:
    def __init__(self, coordinator, registry, policy):
        self.registry, self.policy = registry, policy
        coordinator.register_handler(HISTORY_OPERATION, "artifact", self._page)

    def _page(self, payload, execution):
        if set(payload) != {"release_id"} or not _IDENTIFIER.fullmatch(str(payload["release_id"])):
            raise ReleaseError("Invalid release history request")
        cursor, releases = payload["release_id"], []
        # Two maximum-size manifests fit inside the 256 KiB work result limit.
        for _ in range(2):
            execution.raise_if_cancelled()
            manifest = ReleaseManifest.load(self.registry.manifests / f"{cursor}.json")
            if manifest.release_id != cursor:
                raise ReleaseError("Release history content does not match its ID")
            self.policy.authorize(manifest)
            releases.append(manifest.to_dict())
            cursor = manifest.previous_release_id
            if not cursor:
                break
        return {"releases": releases, "next_release_id": cursor}


def fetch_release_history(coordinator, *, release_id, peer_id, policy,
                          timeout_s=60, max_releases=64, cancel_event=None):
    """Validate signatures and every predecessor before returning genesis first."""
    if not _IDENTIFIER.fullmatch(str(release_id)) or not 1 <= max_releases <= 1024:
        raise ReleaseError("Invalid release history bounds")
    deadline = time.monotonic() + min(max(float(timeout_s), 1), 3600)
    cursor, descending = release_id, []
    while cursor:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (cancel_event is not None and cancel_event.is_set()):
            raise ReleaseError("Release history acquisition timed out or was cancelled")
        result = coordinator.submit(HISTORY_OPERATION, {"release_id": cursor},
            ResourceRequirements(kind="artifact"), peer_id=peer_id,
            timeout_s=remaining, cancel_event=cancel_event)
        if result.status != "ok":
            raise ReleaseError(result.error or "Release history request failed")
        if not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=peer_id,
                expected_requester_id=coordinator.node.identity.node_id_hex):
            raise ReleaseError("Release history receipt is invalid")
        page = result.output
        if set(page) != {"releases", "next_release_id"} or not isinstance(page["releases"], list) or not 1 <= len(page["releases"]) <= 2:
            raise ReleaseError("Invalid release history page")
        for value in page["releases"]:
            manifest = ReleaseManifest.from_dict(value)
            policy.authorize(manifest)
            if manifest.release_id != cursor or (descending and manifest.sequence != descending[-1].sequence - 1):
                raise ReleaseError("Release history does not extend the expected chain")
            descending.append(manifest)
            if len(descending) > max_releases:
                raise ReleaseError("Release history exceeds the acquisition bound")
            cursor = manifest.previous_release_id
        if cursor != page["next_release_id"]:
            raise ReleaseError("Release history continuation does not match its predecessor")
    if descending[-1].sequence != 0:
        raise ReleaseError("Release history does not reach genesis")
    return list(reversed(descending))
