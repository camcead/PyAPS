# Install pyaps-standalone

Needs on the machine: Python 3.10+ for the MCP server (any), **Python 3.12+ for PyAPS itself** (the tool finds `python3.12`, `python3.13`, ...; or set `"python"` in the config), `git`, network access to GitHub and PyPI.

1. A small environment for the MCP server:

       python3 -m venv ~/weave-standalone-venv
       ~/weave-standalone-venv/bin/pip install mcp

2. Check the machine (`FAIL` lines say what is missing):

       cd ~/weave-standalone/pyaps-standalone
       ~/weave-standalone-venv/bin/python -m pyaps_standalone.cli doctor

3. Tell Claude about the server (prints the exact line):

       ~/weave-standalone-venv/bin/python -m pyaps_standalone.cli mcp-config

4. Ask Claude: *"install PyAPS"*, *"what branches can I install"*, *"check the PyAPS environment"*, *"start the explorer on /path/to/file.fits"*. Or at a terminal: `... cli install --ref casper --profile explorer`.

Profiles: `explorer` (viewers and the interactive explorer: open and inspect products) and `pipeline` (also the processing pipeline: a large install with torch, redrock and others; the frozen set `requirements-lock-*.txt` is used when the branch has one). Everything is installed under `~/pyaps-standalone` (`home` in the config): `PyAPS/` (the checkout), `venv/`, `jobs/`, `state.json`.

The pipeline needs data and templates that are not in the repository (`pyaps_templates` shows their status); this tool does not download survey data.

Config file `~/.config/casper-pyaps-standalone/config.json`, all keys optional: `access` (member|admin), `home`, `remote`, `python`.

Tools (MCP): `pyaps_status`, `pyaps_refs`, `pyaps_install`, `pyaps_update`, `pyaps_check_env`, `pyaps_templates` (read-only), `pyaps_explorer_start`, `pyaps_run`, `pyaps_job_list|status|log|stop`.
