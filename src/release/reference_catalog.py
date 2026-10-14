"""Initial public research catalog. This reference is not a chat model."""
from dataclasses import dataclass

from .manifest import TrustPolicy


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    release_id: str
    policy: TrustPolicy
    max_chain_length: int = 16


REFERENCE_CATALOG = CatalogEntry(
    family="lixyswarm-byte-reference-v1",
    release_id="59f68cdc1e6985fd539c1299e42bd200a4a24c42959d368fc3748780bcea98a3",
    policy=TrustPolicy(
        trusted_signers=("53cea6456aced3a5a31ad24505aaa47d88e1827d1e7976592c91b54fc0e8980f",),
        pinned_genesis_release_id="9b6ee48ae1f58733f930d6cb2491fa937f1c577292ff04034e68d056c64fc53e",
    ),
)
