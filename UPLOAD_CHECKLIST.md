# GitHub replacement checklist

1. Replace the public repository contents with this release directory while preserving the repository's `.git/` directory.
2. Confirm that `docs/`, local environments, raw PDB files, generated tensors, run logs, and smoke-test outputs are not staged.
3. Commit the source code, fixed splits, manifests, summary tables, compact figures, environment files, and the three included contact-pretraining checkpoints.
4. Upload the final validation-selected affinity checkpoints as GitHub Release assets and add their exact URLs and SHA-256 values to `checkpoints/README.md` and `checkpoints/SHA256SUMS.txt`.
5. Keep retrospective test-selected checkpoints separate and label them explicitly if they are retained for historical purposes.
6. Use the public repository URL `https://github.com/WenYuxin2003/StructRSMA` in the revised manuscript.

No smoke test was run while preparing this replacement package, following the author's instruction.
