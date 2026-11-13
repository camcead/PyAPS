# Code graph (Graphify) for this repository

A generated map of this repository's code: functions, classes, modules and who calls or imports whom.
Use it to answer "where is X handled?", "what calls X?", "how is A connected to B?" before reading files.
It covers source code only (no documentation, data or configuration) and is built **locally and offline**.

## Use it

```sh
graphify query "database connection keepalive"     # symbols related to a question (add --budget 6000 for more)
graphify explain "apply_client_keepalive"          # one symbol and its neighbours
graphify path "run_sync" "acquire"                 # shortest connection between two symbols
graphify update .                                  # rebuild now (git hooks already do it after commit/checkout/merge)
```

The graph is in `graphify-out/` (`graph.json`, `GRAPH_REPORT.md`, `graph.html`). It is **generated, ignored by git
and must never be committed, emailed or uploaded**: it contains symbol names, paths and comment snippets that can
describe internal infrastructure. It is exactly as private as this repository.
It follows the checked-out branch; it describes the code you have, not what is deployed. For a deployed system,
compare the commit in `graphify-out/` with the running commit first.

A query is a keyword-seeded walk through the graph, not a search engine: use real symbol or file names where you can,
and confirm what you find by reading the code.

## Set up on a machine (once per machine, then once per clone)

The `graphify` command is a restricted wrapper around a pinned, hash-checked Graphify release (code-only, offline,
no API keys, no self-update). It comes from the graphify-kit (`install_graphify.sh` once per machine,
`setup_repo.sh <this repo>` once per clone). Do not install Graphify any other way and do not run its own
`install`, `add`, `label` or `/graphify` commands: the wrapper refuses them on purpose.

Edit `.graphifyignore` to change what is left out (gitignore syntax).
