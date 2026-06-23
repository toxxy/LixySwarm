# LixySwarm

LixySwarm is a research prototype for bio-inspired distributed language intelligence. It combines transformer agents (ants), persistent memory (Matriarca/elephant), an echolocation router (dolphin), and the persistent LSP v3 peer protocol.

**Current status (2026-06-23):** local research prototype. The model, memory, runtime, LAN/explicit-peer transport, status publisher, and dashboard paths exist. The repository is not ready for untrusted Internet exposure or mass deployment.

## Paper

The design reference is [LixySwarm: AntElephantDolphin](paper/LixySwarm_AntElephantDolphin.pdf), dated June 22, 2026. The paper deliberately distinguishes implemented, partial, and future work. [PAPER_COMPLIANCE.md](PAPER_COMPLIANCE.md) maps those claims to the current code and tests.

The manuscript's experimental numbers are historical results for a specific checkpoint and evaluation setup. They are not guarantees for every checkout or checkpoint.

## Implemented system

The default swarm configuration contains:

- Three GPT-style `AgentBase` instances with `FeromonGate`, fixed identity vectors, pheromone outputs, and confidence heads.
- A `DolphinPool` whose primary dolphin creates five learned pings and an attention-based acoustic map.
- A dual `Matriarca` runtime with separate personal and global banks, retrieval, importance updates, compression, and sect legacy storage.
- Two swarm rounds followed by confidence aggregation with a 20% Matriarca bias.
- `RuntimeSession` for cross-turn state, dynamic task profiles, sampling controls, and response-memory feedback.
- Node, sect, and ant lifecycle managers exercised by local tests.
- LSP v3 persistent sessions, mandatory signatures, anti-replay, DNS/bootstrap discovery, peer exchange, resource declaration, and global-memory deltas.
- Consent-gated typed work for isolated peer inference and bounded gradient computation; peers never provide executable code.
- Three-to-31-peer gradient quorums with exact-input validation and chunked coordinate-median aggregation; aggregate results remain unapplied candidates.
- Ed25519 work-result receipts bound to worker, requester, job, output, and timestamp, retained in quorum provenance.
- Validated gradient-quorum contributions produce dual-signed useful-work credits: the worker receipt proves the result and the requester attests that it entered an aggregate. Workers present a bounded set over encrypted sessions; requesters verify it locally and prioritize firsthand accepted work, while duplicate contributions cannot mint additional credit.
- Persistent requester-local scheduling history gives an identity-aged newcomer one exploration opportunity every five selections without reducing available quorum network-group diversity; only pseudonymous node IDs and counters are stored.
- Inbound work admission has a fixed global queue, per-identity concurrent and per-minute quotas, and portable signed overload rejections; remote offers cannot grow the executor queue without bound.
- Requester timeout or explicit local cancellation sends an authenticated `WORK_CANCEL`; compatible handlers receive a `WorkUnit` subclass with deadline/cancellation checks. Remote inference checks per generated token and gradient work checks between expensive phases and parameters.
- Deterministic replicated inference across three-to-nine model-matched peers with coarse network-group diversity and an exact-output majority rule.
- Optional persistent Hashcash-style identity-work stamps bound to Ed25519 keys; operators may configure a minimum with `LIXYSWARM_IDENTITY_WORK_BITS`, but the default is zero because useful validated training—not expendable hashing—is the intended reputation basis.
- Threshold-signed model release manifests, local trust roots, pinned genesis support, monotonic activation, revocation lists, and explicit rollback.
- SHA-256 content-addressed model, dataset, evaluation, and gradient artifacts with resumable chunk transfer and end-to-end verification.
- A `lixyswarm` CLI for contribution policy, persistent node startup, and privacy-safe artifact import/listing.
- FastAPI status/chat endpoints, a status publisher, and two static frontends.
- Continuous training and an opt-in metabolic-hunger decision function.

Some paper descriptions are only partially represented. In particular, the main forward pass does not implement the paper's exact `fitness × confidence × role_weight` equation. Useful-work credits are verifiable evidence issued by pseudonymous requesters, not proof that operators are independent or that a gradient is beneficial. The scheduler now prefers its own previously accepted contributors, bounded issuer-diverse evidence, and periodic newcomer exploration. Inbound queue/identity limits and cooperative cancellation exist, but Sybil-independent issuer trust, hardware validation, durable fair-share scheduling, forced termination of non-cooperative work, and network-wide promotion governance remain.

## Free participation model

LixySwarm has no token, payment, fee, stake, or mandatory expendable proof of work. A useful-work credit is scheduling evidence, not money or a transferable asset. Connectivity starts immediately; consented compute is eligible immediately when capacity is needed, and a still-uncredited identity joins the explicit exploration rotation after one minute of continuity. Hashcash identity work is optional and disabled by default.

## Architecture

```text
tokens
  -> DolphinPool: five pings + attention triangulation + sleep state
  -> Matriarca: personal/global retrieval -> infrasound
  -> Ant agents: parallel transformer passes with pheromone gating
  -> confidence aggregation + Matriarca bias
  -> repetition-penalized top-k/top-p sampling in RuntimeSession
```

LSP v3 multiplexes pheromones, peer exchange, global-memory deltas, work offers, and work results over persistent TCP `7338` sessions. Large artifacts move as verified chunks through typed work. A node loads saved peers, bootstraps from multiple configured seeds, learns direct routes, and continues if a seed disappears. There are no built-in public DNS seed domains yet; configure endpoints with `LIXYSWARM_BOOTSTRAP_SEEDS=host:7338[,host:7338]`.

## Install and verify

Python 3.12 and a recent PyTorch installation are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
pytest -q
```

The full suite passed **188 tests** on 2026-06-23.

Join as a connectivity/artifact node, or explicitly consent to compute:

```bash
lixyswarm init --mode relay --yes
lixyswarm start  # connectivity and explicitly imported artifacts

# Opt-in compute contribution with a trusted local checkpoint:
lixyswarm init --mode balanced --yes
lixyswarm start --checkpoint ./checkpoints/swarm_best.pt
```

Files are never shared automatically. Explicit artifact import publishes a hash-based manifest without the source filename:

```bash
lixyswarm artifact-add ./tokens.npy --kind dataset --media-type application/x-npy
lixyswarm artifact-list
```

Release trust is local and threshold-based. Keep release private keys offline and outside Git/synchronized project folders:

```bash
lixyswarm release-keygen /secure/offline/release-a.pem
lixyswarm trust-init --threshold 2 --signer SIGNER_A --signer SIGNER_B
lixyswarm release-create --model-id SHA256 --model-format pytorch-weights-only-v1 --sequence 0 --output release.json
lixyswarm release-sign release.json --key /secure/offline/release-a.pem
lixyswarm release-accept release.json --activate
lixyswarm start --release
```

The repository provides the mechanism but does not ship official signer keys, a genesis release, or model weights.

Connected peers announce locally trusted releases and receivers download/verify referenced artifacts directly from the announcer. Add `--auto-activate` to `trust-init` only when automatic monotonic activation is desired; the choice is persisted locally and defaults off.

Run `pytest --collect-only -q` for the current collection count. Test totals in old experiment reports refer to earlier scripts or revisions.

Useful entry points:

```bash
python3 lixy_chat.py
python3 lixy_orchestrator.py --status
python3 auto_train.py --status
python3 benchmark.py --health-only --cpu
uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Large checkpoints and training datasets are intentionally excluded from Git. A fresh clone therefore cannot reproduce paper metrics without separately obtained artifacts and data.

## Network safety

LSP v3 closes the v2 signature, replay, framing, outbound-NAT, and application-payload confidentiality gaps and adds coarse path diversity/local bans, but the public network is not release-ready. Remaining blockers include key rotation/cryptographic review, Sybil-independent quorum membership and result reputation, authenticated API access, official redundant DNS seeds, adversarial load/fuzz testing, official threshold trust roots/genesis artifacts, process-level job isolation, and network-wide promotion governance. See [INTERNET_SCALE_READINESS.md](INTERNET_SCALE_READINESS.md).

Publisher authentication uses `LIXYSWARM_PUBLISH_TOKEN`. Personal Matriarca encryption is enabled only when `LIXYSWARM_MATRIARCA_KEY` is set. Network addresses are not published or exposed by default; enabling that requires explicit environment flags documented in [SECURITY.md](SECURITY.md).

Never commit checkpoints, corpora, session histories, identities, peer databases, logs, `.env` files, or operator addresses.

## Documentation

- [ARQUITECTURA.md](ARQUITECTURA.md): implemented architecture
- [PAPER_COMPLIANCE.md](PAPER_COMPLIANCE.md): paper-to-code compliance matrix
- [INTERNET_SCALE_READINESS.md](INTERNET_SCALE_READINESS.md): production and mass-Internet gap analysis
- [LSP_SPEC.md](LSP_SPEC.md): current wire protocol
- [LSP_ARCHITECTURE.md](LSP_ARCHITECTURE.md): protocol decisions and target architecture
- [DISTRIBUTED_PROTOCOL.md](DISTRIBUTED_PROTOCOL.md): operator-facing network overview
- [ORCHESTRATOR_RUNTIME.md](ORCHESTRATOR_RUNTIME.md): runtime behavior
- [SECURITY.md](SECURITY.md): security and privacy policy
- [VPS_SETUP.md](VPS_SETUP.md): LSP v3 public seed deployment
- [PENDIENTES_2026-06-05.md](PENDIENTES_2026-06-05.md): current backlog retained under its historical filename

Historical experiment notes are labeled as such and must not be read as current release status.
