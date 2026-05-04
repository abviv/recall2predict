# Third-Party Notices

This repository contains selected third-party or derived components required
by the Recall2Predict AV2 training path.

## HPTR

Files under `src/HPTR/` are derived from HPTR:

- Upstream: https://github.com/zhejz/HPTR
- License: Creative Commons Attribution-NonCommercial 4.0 International
- Local license copy: `src/HPTR/LICENSE`

These files are not covered by the root MIT license. Their use is subject to
the HPTR license terms, including the non-commercial restriction.

## layers_in_my_way

`src/layers_in_my_way/` is tracked as a Git submodule:

- Upstream: https://github.com/abviv/layers_in_my_way
- License: MIT
- Local license copy after submodule initialization:
  `src/layers_in_my_way/LICENSE`

## Templates and Frameworks

The training scaffold follows conventions from Lightning, Hydra, and the
Lightning-Hydra template ecosystem. Runtime dependencies retain their own
licenses.
