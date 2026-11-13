"""IFU FERRE: bins run side by side (run_bins) and every bin has its own control files.

Before 2026-11-13 the IFU_Gal stage ran the ferre.x processes of one bin after the other, bin after bin (about 3 minutes per bin on the -O0 build, one
core busy of the two it asked for). The bins are independent, so `nthreads` of them now run at the same time. The control file of a (bin, grid) pair
used to be `input.nml_<grid>` in the shared work directory: a second bin would have overwritten the file of the first one while ferre.x read it.
"""
import os
import threading
import time

from PyAPS import aps_ifu_ferre as F


def test_run_bins_serial_keeps_order_and_uses_one_thread():
    seen = []

    def f(i):
        seen.append(threading.get_ident())
        return i * 10
    assert F.run_bins(f, 5, 1) == [0, 10, 20, 30, 40]
    assert set(seen) == {threading.get_ident()}


def test_run_bins_parallel_keeps_order_and_overlaps_the_bins():
    seen = []

    def f(i):
        seen.append(threading.get_ident())
        time.sleep(0.2)
        return ('bin', i)
    t0 = time.time()
    out = F.run_bins(f, 8, 4)
    dt = time.time() - t0
    assert out == [('bin', i) for i in range(8)]
    assert len(set(seen)) > 1
    assert dt < 8 * 0.2 * 0.75, dt          # 8 bins of 0.2 s, 4 at a time: about 0.4 s, serial would be 1.6 s


def test_run_bins_edge_cases():
    assert F.run_bins(lambda i: i, 0, 4) == []
    assert F.run_bins(lambda i: i, 3, 16) == [0, 1, 2]       # more threads than bins
    assert F.run_bins(lambda i: i, 3, None) == [0, 1, 2]     # no thread count: serial
    assert F.run_bins(lambda i: None if i == 1 else i, 3, 3) == [0, None, 2]    # a bin that gave nothing keeps its place


def test_every_bin_and_grid_has_its_own_control_file(tmp_path, monkeypatch):
    from PyAPS import aps_ferre
    work = tmp_path / 'fr_wd'
    work.mkdir()
    tmpl = tmp_path / 'templates'
    tmpl.mkdir()
    grids = ['1', '2', '3']
    for g in grids:
        (tmpl / f'n_rweave{g}.hdr').write_text('x')
    ran = []

    def fake_worker(cmdstr):
        nml = cmdstr[0]
        ran.append(nml)
        assert os.path.isfile(nml)
        text = open(nml).read()
        pixel = os.path.basename(nml).split('.')[0]
        assert "PFILE='%s" % os.path.join(str(work), pixel) in text      # the file belongs to this bin
        base = os.path.basename(nml)
        grid = base.rsplit('.nml', 1)[1]
        open(os.path.join(str(work), '%s.opf%s' % (pixel, grid)), 'w').write('0')
        return True, 0, None
    monkeypatch.setattr(aps_ferre, 'ferre_exe_worker', fake_worker)
    monkeypatch.setattr(aps_ferre, 'head_synth', lambda f: {'N_OF_DIM': '3'})
    monkeypatch.setattr(aps_ferre, 'opfmerge', lambda **kw: {'success': True})
    monkeypatch.setattr(F, 'parse_ferre_output_ifu', lambda work_dir, pixel, bin_id, prefix: {'BIN_ID': bin_id, 'pixel': pixel})
    cfg = {'orig_setups': [''], 'targs_mode': 'IFU'}
    res = F.run_bins(lambda i: F.run_ferre_on_bin(i, 'bin%04d' % i, str(work), '/bin/true', str(tmpl), grids, 'n', cfg), 6, 3)
    assert [r['BIN_ID'] for r in res] == list(range(6))
    names = sorted(os.path.basename(n) for n in ran)
    assert len(names) == len(set(names)) == 6 * len(grids)
    assert not any(n.startswith('input.nml') for n in names)
    assert sorted(os.listdir(work)) == sorted(set(os.listdir(work)))
