# External Dependencies for PyAPS

This directory holds external software that PyAPS calls but does not ship. Nothing here is
tracked in version control (see `.gitignore`); install what you need under
`<PYAPS_DIR>/externals/`.

## FERRE - stellar spectral analysis (optional)

Needed only by the stellar-parameter modules (`aps_ferre`, `aps_ifu_ferre`, `aps_feswi`).

FERRE is a Fortran 90 code that matches models to data by finding the model parameters
that best reproduce observations in a chi-squared sense (model interpolation, optimal
parameters, error covariance, model predictions).

```bash
git clone https://github.com/callendeprieto/ferre.git <PYAPS_DIR>/externals/ferre
cd <PYAPS_DIR>/externals/ferre/src
make
```

Use upstream commit f50a006 (2026-10-21) or later. The commits 60be172 .. 0e05a6a (July to October 2026) carry `OPT = -O0` in `src/makefile` (an accident, reverted
upstream in be05232): a `ferre.x` built from them is about 3 times slower and has no OpenMP. Check that the makefile says `OPT = -O2 -fopenmp -ftree-vectorize` and that the
binary links the OpenMP runtime: `ldd bin/ferre.x | grep gomp` must print a line.

Check the build:

```bash
<PYAPS_DIR>/externals/ferre/bin/ferre.x   # prints its usage banner
```

PyAPS finds the executable through `FERRE_EXE` in `configs/script_params.yaml`
(`${PYAPS_HOME}/externals/ferre/bin/ferre.x` by default) or the `--ferre_exe` option of
the module you run.

## Other tools

Contributed-software (CS) tools are installed separately under `<PYAPS_DIR>/CS/`; see
[CS/README.md](../CS/README.md).

## Troubleshooting

- **`ferre.x: command not found` / "FERRE executable not found"**: the path in
  `FERRE_EXE` does not point at a built `ferre.x`. Rebuild, or edit your
  `script_params.yaml`.
- **Compiler errors while building FERRE**: install `gfortran` (`apt install gfortran`,
  `brew install gcc`, or `conda install gfortran`).
