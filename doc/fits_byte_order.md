# FITS arrays are big-endian: make them native where they are read

Written 2026-11-15 after the IFU_ExGal serial-against-pool difference (CHANGELOG, "Fixed (IFU_ExGal: the serial path ...").

## What happens

FITS stores numbers big-endian. astropy hands a table column over as it is (`dtype('>f8')`) and `np.array(col)` keeps the
byte order. numpy computes correctly with such an array, but not in the same way as with native `float64`:

- a reduction (`sum`, `mean`, `std`, `dot` of long vectors, ...) over byte-swapped data goes through numpy's buffered cast in
  chunks of 8192 values; a native contiguous array is summed in one pairwise pass. The additions happen in another order and
  the last digits differ. Measured on 10 711 pixels: `np.sum` differs in 8 of 21 columns, `np.std` in one.
- a smooth fit does not notice. An ill-conditioned one (a low-S/N Voronoi bin, a parameter on its bound, a regularisation set
  from `np.std`) amplifies a rounding step of 1e-16 to 1e-10 in the model and 1e-6 to 1e-1 in the weak emission lines.

`multiprocessing` queues pickle each job, and a pickled non-contiguous slice (`bin_data[:, i]`) comes out **native**. So a pool
worker computes on native numbers while the serial loop of the same function computes on the big-endian view of the same
numbers. Same input, two answers, and nothing in the logs says so. It looks like "the serial path has a bug" or "the pool is
not deterministic" and is neither.

## Rule

An array read from a FITS table that is going to be fitted, summed or compared between runs is made native **once, where it is
read**:

```python
from PyAPS import aps_native
bin_data = aps_native.native_array(hdu[1].data.SPEC.T)     # replaces np.array(...); same values, order, still a copy
```

Then a serial loop, a pool worker, a lane and a test see the same bytes. Do not "fix" a serial/pool difference by changing the
serial loop to copy or pickle its arguments (tried on 2026-11-14: no effect, the dtype is the problem, not the copy).

## How to recognise it, and how it was found

1. Compare the **arguments** of the function in both modes, not only its results: dtype, byte order, strides, a hash of the
   bytes (`>f8` against `float64` was the first line of the report).
2. Replay the one bad bin in one process with the same values as `>f8` and as native copies; if only that bin changes it is
   this, and the operation can be bisected (`np.sum`, `np.std` of the same data).
3. Compare modes after the change on real data, column by column, ignoring the random `ERR_*` columns: serial = every
   worker count, and the pool before = the pool after (the production path must not move).

Related: the same difference can come from BLAS threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) and
from the number of lanes / workers when a result depends on the order jobs finish. Exclude those first; they cost a minute.

## State (2026-11-15)

Done: `IFUExGalPPXF` and `IFUExGalEMIPPXF` (spectra, errors, stellar model). Line strengths build their arrays with `np.zeros`
(native) and need nothing.

Not done, same read pattern, serial branch present (each wants an A/B on real data like the one in the CHANGELOG):
`MOSExGalPPXF.py` 427-428, `MOSExGalPrepare.py` 221-222, `MOSExGalEMIPPXF.py` 2866-2868 and 2888, `MOSExGalLS.py` 952-953;
parent-only: `IFUExGalPrepare.loadAllSpectra` 210-211 (feeds the Voronoi binning; a change there can move a bin edge).
