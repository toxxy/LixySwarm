"""Initial public research catalog. This reference is not a chat model."""
from dataclasses import dataclass

from .manifest import TrustPolicy


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    release_id: str
    policy: TrustPolicy
    max_chain_length: int = 16
    allow_signed_updates: bool = False
    bootstrap_providers: tuple[str,...] = ()


REFERENCE_CATALOG = CatalogEntry(
    family="lixyswarm-byte-reference-v1",
    release_id="59f68cdc1e6985fd539c1299e42bd200a4a24c42959d368fc3748780bcea98a3",
    policy=TrustPolicy(
        trusted_signers=("53cea6456aced3a5a31ad24505aaa47d88e1827d1e7976592c91b54fc0e8980f",),
        pinned_genesis_release_id="9b6ee48ae1f58733f930d6cb2491fa937f1c577292ff04034e68d056c64fc53e",
    ),
    max_chain_length=64,
    allow_signed_updates=True,
    bootstrap_providers=("c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255",
                         "ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0"),
)
