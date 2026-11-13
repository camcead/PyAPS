<!-- graphify-kit:start -->
## Code graph (graphify-kit)
This repository has a generated, offline code graph in `graphify-out/` (see `GRAPHIFY.md`).
- To find where something lives or what calls it, run `graphify query "<question>"`, `graphify explain "<symbol>"` or `graphify path "<A>" "<B>"` before searching files, then read the code to confirm.
- Rebuild with `graphify update .` after large changes (git hooks do it on commit and checkout).
- Use only the `graphify` command on PATH (a restricted wrapper). Never run `pip install` or `uv` for graphify, never run `/graphify`, `graphify install`, `graphify add` or any command that sends documents to a model.
- `graphify-out/` is generated and private: never commit, paste or upload it.
<!-- graphify-kit:end -->
