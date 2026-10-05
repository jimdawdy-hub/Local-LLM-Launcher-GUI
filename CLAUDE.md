## LESSONS

- When evaluating an engine release for a named GPU, check architecture-specific kernel selection and optional dependencies as well as its CLI flags; distinguish automatic improvements from missing launcher controls.
- Before supporting an engine update, verify existing catalog flags against the target binary's help; upstream can remove flags such as llama.cpp's `--no-mmap` and `--mlock`.
- Design hardware controls for supported users' OS-visible topology, including multi-socket bare-metal hosts, instead of treating this one-node development VM as the product limit.
- Check each engine's own build requirements; a shared tool name does not imply a shared minimum version.
- Pair every newly exposed cache precision with its memory-estimate multiplier and a fit-boundary regression test.
- Treat saved settings as authoritative on remount; historical background-job results must not undo a later manual rollback.
- Browser tests for shared polling must wait for initial in-flight requests instead of assuming fixed startup delays.
- Verify documented CLI choice names against the tagged argument registry; vLLM 0.30 release prose says B12X_ATTN but its actual attention selector is B12X.
- Run full project tests with the project's virtual-environment Python and the active worktree's PYTHONPATH; the system Python lacks the API dependencies.

- Recheck upstream engine support before making an old-version limitation a universal launcher rejection; distinguish model weight quantization from K/V cache precision.
