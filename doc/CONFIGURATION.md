# Configuration: first-time setup

PyAPS keeps **no** installation-specific values in git. The repository carries *templates* (`*.example`);
every installation copies them to a local, git-ignored name and fills in its own values. The local files are
never committed (they are listed in `.gitignore`), see [PUBLIC_HYGIENE.md](PUBLIC_HYGIENE.md).

| Template (tracked) | Copy it to (local, ignored) | Needed for |
|---|---|---|
| `configs/script_params.yaml.example` | `configs/script_params.yaml` | `aps_runner` and every module started through it (`--config_file configs/script_params.yaml`) |
| `configs/explorer.env.example` | `configs/explorer.env` (`chmod 600`) | the explorer server / Docker deployment (shared secret, upstream URL, default data folders) |

Other host-local names that are ignored by git and must never be committed: `configs/PyAPS_dms_config*`
(database connection of the optional, separately maintained database tooling), `configs/script_params_*.yaml`, `configs/*.bak`,
`configs/*.orig`, `configs/ACTIVE_SITE`, `configs/pyaps_site.env`, `configs/*.local.*`, any `*.env`.

## Steps

```bash
cp configs/script_params.yaml.example configs/script_params.yaml
cp configs/explorer.env.example       configs/explorer.env && chmod 600 configs/explorer.env   # only if you run the explorer
$EDITOR configs/script_params.yaml configs/explorer.env       # fill in every value (see below)
python tools/check_config.py                                  # lists every <...> still unfilled, every ${VAR} not set
```

`tools/check_config.py` exits non-zero until the local files are complete. It also refuses a local file that
git tracks.

## What to enter

Placeholder conventions: `<...>` = a value you must supply, `${PYAPS_HOME}` = root of your PyAPS working
tree (exported, or defaulted by `aps_runner` to the checkout), `<env_suffix>` = see below.

**`configs/script_params.yaml`**

| Key(s) | Enter |
|---|---|
| `PyAPS_DIR` | leave as `${PYAPS_PKG_DIR}` (the folder with the `aps_*.py` modules, set by the runner) |
| `PyAPS_RES`, `CS_RES`, `PyAPS_CAT`, `PyAPS_CAL`, `PyAPS_XML` | the folders for L2 results, CS results, catalogues, calibration files and XML, normally `${PYAPS_HOME}/PyAPS_data/<name><env_suffix>` (`PyAPS_data` is a git-ignored directory or a symlink to the data volume). Replace `<env_suffix>` (below) |
| `PyAPS_CONFIG`, `PyAPS_EXGALCONFIG` | normally unchanged (`${PYAPS_HOME}/configs`, `.../ExGal_configs`) |
| `templates_*`, `IFU_templates_*`, `SQ_templates`, `FESWI_*` grids | where your template libraries live (default: under `${PYAPS_HOME}/PyAPS_templates`, a git-ignored directory or symlink). `templates_RVS` is also what the RVS jobs use: the job scripts export it as `PYAPS_RVS_TEMPLATES`, which `template_lib: '${PYAPS_RVS_TEMPLATES}'` of `configs/rvs_config.yaml` refers to. A missing directory stops the job with an error naming this key; there is no fallback location |
| `FERRE_EXE`, `FESWI_path`, `SQ_model`, `config_RVS` ... | location of the external programs / models (default: under `${PYAPS_HOME}/externals` and `${PYAPS_HOME}/CS`) |
| `use_venv`, `venv_path` | `'True'` plus the activate script if generated scripts must activate a virtual environment |
| `*_RR`, `*_RVS`, `*_FR`, `*_ExGal`, `IFU_*`, `SQ_*`, `FESWI_*`, `SPACE_*`, `AMY_*` ... | per-module processing defaults (booleans and numbers as quoted strings). The shipped values are the project defaults; change them only to change processing behaviour |

**`<env_suffix>` (development and production side by side).** Production uses the plain data-tree names
(`L1`, `L2`, `CS`, `CAT`, `CAL`, `XML`). A development environment on the same machine uses the same names
with a `_dev` suffix (`L1_dev`, `L2_dev`, `CS_dev`, `CAT_dev`, `CAL_dev`, `XML_dev`). In the local file replace
`<env_suffix>` by nothing (production) or `_dev` (development). A development environment must never point at a
production tree. The same applies to the result database of the optional DMS tooling: use `<db_name>` for
production and the same name with the `_dev` suffix for development, and never configure a development install
with the production database name.

**`configs/explorer.env`** (reference for every variable: [aps_explorer.md](aps_explorer.md))

| Key | Enter |
|---|---|
| `PYAPS_EXPLORER_MULTI_SESSION` | `1` for any shared server |
| `PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH` | `1` to accept only signed tokens from your upstream app, else `0` |
| `PYAPS_EXPLORER_WEAVEOR_SECRET` | a long random secret shared with the token-issuing app (`python -c "import secrets; print(secrets.token_urlsafe(48))"`); different for every environment |
| `PYAPS_EXPLORER_WEAVEOR_URL` | https URL of the upstream app (or remove the line) |
| `PYAPS_EXPLORER_WEAVEOR_TOKEN_MAX_AGE` | seconds a token stays valid (default 300) |
| `PYAPS_EXPLORER_URL_PREFIX` | sub-path when sharing a reverse-proxied host (or remove the line) |
| `PYAPS_EXPLORER_DEFAULT_CALDIR`, `..._CATDIR` | calibration / catalogue folders as seen by the server process |
| `PYAPS_CONFIGDIR` | writable persistent config/cache folder (needed when `configs/ExGal_configs` is not shipped) |
| `PYAPS_EXPLORER_IDLE_TIMEOUT_MINUTES`, `..._SESSION_TTL_SECONDS` | session housekeeping (defaults 10 and 14400) |

The local files are ignored by git. **Never commit them, never paste their contents into issues or pull
requests.** If a secret ever leaves the machine, rotate it.

## Migrating an existing installation (script_params.yaml used to be tracked)

Before this change `configs/script_params.yaml` was a tracked file that each host edited in place; it is now
the ignored local copy of `configs/script_params.yaml.example`. A plain `git pull` on a host that modified the
file stops with *"Your local changes ... would be overwritten"*, and on a host that did not modify it git
deletes it. Use these steps, one host at a time, from the PyAPS working tree (`$PYAPS_HOME`):

```bash
cd "$PYAPS_HOME"
# 1. Back up the live file OUTSIDE the repository, with a timestamp
cp -p configs/script_params.yaml ~/script_params.yaml.bak.$(date +%Y%m%d-%H%M%S)
ls -l ~/script_params.yaml.bak.*                      # confirm the copy exists and is not empty

# 2. Put the tracked file back to the committed content so the pull cannot conflict
#    (your values are safe in the backup)
git checkout -- configs/script_params.yaml

# 3. Update (fast-forward only: stops instead of merging if the branch diverged)
git pull --ff-only                                     # removes the tracked file, adds ...yaml.example and the new .gitignore

# 4. Restore the local file from the backup (git removed it in step 3)
cp -p ~/script_params.yaml.bak.<timestamp-from-step-1> configs/script_params.yaml

# 5. Optional: bring new template keys/comments across, comparing your file with the template
diff configs/script_params.yaml.example configs/script_params.yaml | less

# 6. Confirm: the working tree is clean and the file is ignored
git status                                             # must report: nothing to commit, working tree clean
git check-ignore -v configs/script_params.yaml         # must name the .gitignore rule
python tools/check_config.py configs/script_params.yaml
```

The data-tree keys (`PyAPS_RES`, `CS_RES`, ...) in your restored file already carry the right
(`_dev` or plain) names for that host: keep them as they are. If step 3 reports other local modifications,
stop and sort them out first (`git status`, `git diff`); do not use `git stash` or `git reset --hard` for this.
After the last host is migrated no host edits tracked configuration files any more.
