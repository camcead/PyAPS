# Contributing to PyAPS

Thank you for your interest. Bug reports, documentation fixes, tests and code contributions are welcome.

## Reporting a problem
Open an [issue](https://github.com/camcead/PyAPS/issues) and include: what you ran (command line or
code), what you expected, what happened (full traceback), the PyAPS version (`python -c "import PyAPS;
print(PyAPS.__version__)"`), Python version and operating system. Please do not attach proprietary
survey data; a small file header or a few rows is usually enough. Security problems: see
[SECURITY.md](SECURITY.md) instead.

## Development setup
```bash
git clone https://github.com/camcead/PyAPS.git && cd PyAPS
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",pipeline" if you work on the processing modules
pytest                              # data-dependent tests skip themselves without WEAVE data
```
To run the data-dependent tests point `PYAPS_TEST_DATA` at a directory containing `L1/`, `L2/`, `CAL/`, `CAT/`.

## Branches
* `main` - the latest released version. Stable; releases and tags are made from it.
* `develop` - ongoing work. Pull requests go here.

## Making a change
1. Fork, then create a branch from `develop`: `git switch -c my-change develop`.
2. Keep the change focused; add or update a test when behaviour changes.
3. Run `pytest` and, if you touched packaging, `python -m build && twine check dist/*`.
4. Do not commit machine-specific paths, host names, credentials or real data. Use the placeholders
   documented in the README (`<PYAPS_DIR>`, `<PYAPS_DATA>`, ...) and the environment variables
   (`PYAPS_HOME`, ...).
5. Add a line to `CHANGELOG.md` under "Unreleased".
6. Open a pull request against `develop` and fill in the template.

## Conventions
* Python >= 3.12. Match the style of the surrounding code; the project does not enforce a formatter.
* Command-line flags shared by the `aps_*.py` scripts live in `aps_common_args.py`; a test guards
  against accidental flag changes.
* Every new module should have a short docstring saying what it does and how to run it.

## Releases (maintainers)
Update `version.txt` and `CHANGELOG.md` on `develop`, then merge `develop` into `main`
(`git switch main && git merge --ff-only develop && git push`), and create the GitHub release with tag
`vMAJOR.MINOR[.PATCH]` targeting `main`. The release workflow attaches the built distributions.
