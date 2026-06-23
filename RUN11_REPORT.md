# Run 11 Historical Report

**Status:** archival experiment note; not current release status

The manuscript reports Run 11 as the longest continuous training segment, approximately steps 12k to 54k, with validation loss improving from about 4.27 to 3.57. The subsequent Dolphin Phase A experiment reports a best value near 3.44 at approximately step 54.5k.

These values belong to a specific local checkpoint, dataset mixture, tokenizer, and evaluation procedure. Large checkpoints and datasets are excluded from the public Git tree, so a fresh clone cannot independently reproduce the result.

Do not compare this loss directly across different corpora, tokenizers, sequence lengths, or evaluation code. Short bilingual correctness and repetition measurements are smoke-test indicators, not broad quality guarantees.

For a reproducible release, publish:

- exact commit and clean working-tree state;
- checkpoint and dataset hashes;
- model/training configuration and random seeds;
- dependency lock, CUDA/PyTorch versions, and hardware;
- raw train/evaluation logs and metric JSON with private content removed;
- commands that regenerate every table in the paper.

Current implementation status belongs in `README.md` and `PAPER_COMPLIANCE.md`, not in this historical report.
