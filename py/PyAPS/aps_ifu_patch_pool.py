"""aps_ifu_patch_pool.py -- run the patches of IFU_Gal / IFU_ExGal side by side (standard library only).

Why (OB 14699, night 20260315): IFU_Gal worked on one patch after another and used one or two cores of the job; 15 of the 44 patches took 24 h. The
patches of a patch table are independent (every patch has its own `<headname>_PNNNN` files, work directory and output), so they can run at the same
time. Both stages (aps_ifu_Gal.ifu_Gal, aps_ifu_ExGal.ifu_ExGal) share this driver.

The budget
    `mp` is the CPU budget of the job (--mp_Gal / --mp_ExGal = the CPUs of the sbatch line). It is split into LANES. A lane is one process that takes
    patches from a common queue, one after the other, and gives each patch `capacity` inner workers (the bins of FERRE and RVS for IFU_Gal, the pool of
    PPXF / EMIPPXF / LS for IFU_ExGal). The capacities of all lanes add up to `mp`, so the job never runs more worker processes than it asked for.

        patch_workers = 1 (or mp = 1, or one patch):  one lane of capacity `mp`, run in this process in the order of the table = the old serial code.
        patch_workers = N > 1:                         N lanes (at most mp, at most the number of patches), `mp` shared out as evenly as it goes.
        patch_workers = 0 / None (auto):               the number of lanes with the shortest estimated wall time (see plan_workers): many small patches
                                                       use one lane per CPU; a few big patches use fewer lanes with several inner workers each.

        IFU_ExGal: every lane gets at least 2 inner workers (`min_inner`): PPXF / EMIPPXF / LS have a pool path (the one production always ran) and a separate
        serial path for one worker; a budget of 1 stays the plain serial loop.

Memory
    A lane holds the aperture of its patch and (ExGal) the template library: PSS about 2 GB for a Gal lane and 12 GB for an ExGal lane (OB 14699 / OB 26140,
    2026-11-13), the inner workers of a lane share its pages. So the number of lanes is also capped by the memory of the job (`lane_cap`: SLURM --mem,
    the cgroup limit or the free RAM, 90 % of it, divided by the GB per lane); the CPUs that no lane can use go to the inner workers.

The order
    Patches are handed out largest first (longest processing time first), by the estimated number of bins (area of the aperture). The patch sizes of a
    real field are very uneven (91 bins ... 2 bins), so the big ones start first and the small ones fill the gaps.

Isolation
    A patch that fails (an exception, even SystemExit, or a worker that dies) is reported as failed and does not stop the others; the parent
    adds the answers up in the order of the table, so the summary (IFU_PATCH_SUMMARY) and the exit rule do not depend on the order of completion.
    Lines printed by a patch in a lane start with `[PNNNN] `.
"""
import heapq
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback

# Bins per arcsec^2 of an extraction aperture (A x B, full axis lengths after the aperture factor): OB 14699 (LIFU HR, Gal): 91 bins for 484 arcsec^2,
# 2-4 bins for the single stars. Only used to rank patches and to plan the lanes: a wrong estimate costs wall time, never a result.
BINS_PER_ARCSEC2 = 0.19
MIN_BINS = 2.0
# per-patch work that does not scale with the inner workers (read the aperture, bin it, merge the product), in bins of one worker
PATCH_OVERHEAD_BINS = 1.0
# a smaller number of lanes is taken when it is within this factor of the best estimated wall time (fewer processes, less memory)
AUTO_TOLERANCE = 1.05
# PSS of one lane in GB (PatchPlan memory cap): measured on OB 14699 (Gal, HR: 13.2 GB for 4 lanes, 19.6 GB for 10) and OB 26140 (ExGal, LR: 48.8 GB for 4 lanes)
LANE_GB = {"Gal": 2.5, "ExGal": 12.0}


def estimate_bins(a_arcsec, b_arcsec):
    """Rough number of bins of an aperture with the full axis lengths `a_arcsec` x `b_arcsec` (what the patch costs, in bins)."""
    try:
        v = float(a_arcsec) * float(b_arcsec) * BINS_PER_ARCSEC2
    except (TypeError, ValueError):
        return MIN_BINS
    if not math.isfinite(v):
        return MIN_BINS
    return max(MIN_BINS, v)


def lpt_order(costs):
    """Positions of `costs` from the largest to the smallest; equal costs keep the order of the table."""
    return sorted(range(len(costs)), key=lambda i: (-float(costs[i]), i))


def split_even(total, parts):
    """`total` in `parts` integers that differ by at most 1, the larger ones first. ``split_even(10, 3) == [4, 3, 3]``."""
    parts = max(1, int(parts))
    base, extra = divmod(int(total), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def lane_time(cost, capacity):
    """Estimated time of a patch of `cost` bins on a lane of `capacity` inner workers (in bin-times of one worker): the bins go out in rounds
    of `capacity`, plus the work that does not scale (reading and binning the aperture, the merge)."""
    return PATCH_OVERHEAD_BINS + math.ceil(float(cost) / max(1, int(capacity)))


def simulate(costs, lanes, order=None):
    """Estimated wall time of the dispatch: the lane that is free first takes the next patch (largest first). Same rule as the real queue."""
    order = lpt_order(costs) if order is None else order
    free = [(0.0, -c, i) for i, c in enumerate(lanes)]      # ties: the lane with more capacity starts first
    heapq.heapify(free)
    end = 0.0
    for pos in order:
        t, negc, i = heapq.heappop(free)
        t += lane_time(costs[pos], -negc)
        end = max(end, t)
        heapq.heappush(free, (t, negc, i))
    return end


def available_memory_gb(env=None, meminfo="/proc/meminfo"):
    """GB this job may use: the smallest of SLURM --mem (SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x CPUs), the cgroup limit and the available RAM
    now. None when nothing can be read."""
    env = os.environ if env is None else env
    found = []
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            found.append(float(env["SLURM_MEM_PER_NODE"]) / 1024.0)
        elif env.get("SLURM_MEM_PER_CPU"):
            cpus = float(env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_NTASKS") or 1)
            found.append(float(env["SLURM_MEM_PER_CPU"]) * cpus / 1024.0)
    except ValueError:
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw.isdigit() and int(raw) < 2 ** 50:
                found.append(int(raw) / 2.0 ** 30)
                break
        except (OSError, ValueError):
            pass
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    found.append(float(line.split()[1]) / 1048576.0)
                    break
    except (OSError, ValueError):
        pass
    return min(found) if found else None


def lane_cap(lane_gb, avail_gb="detect", fraction=0.9):
    """Lanes that fit in memory: ``floor(fraction x available / lane_gb)``, at least 1; None when the memory is unknown."""
    if avail_gb == "detect":
        avail_gb = available_memory_gb()
    if not avail_gb or not lane_gb:
        return None
    return max(1, int(fraction * float(avail_gb) / float(lane_gb)))


class PatchPlan(object):
    """How the budget is spent: `lanes` = inner workers of every lane (their sum never exceeds `mp`), `order` = positions in dispatch order."""

    def __init__(self, mp_budget, n, lanes, order, reason, est_wall=None, est_serial=None, cap=None, warning=None):
        self.mp, self.n, self.lanes, self.order = mp_budget, n, list(lanes), list(order)
        self.reason, self.est_wall, self.est_serial = reason, est_wall, est_serial
        self.cap, self.warning = cap, warning                   # memory cap on the lanes (None = unknown), a warning to print

    @property
    def serial(self):
        """One lane: the patches run in this process, in the order of the table (the old behaviour)."""
        return len(self.lanes) <= 1

    @property
    def inner(self):
        return self.lanes[0] if self.lanes else max(1, self.mp)

    def describe(self, label="IFU"):
        if self.n == 0:
            return "%s patch plan: no patch to run" % label
        if self.serial:
            return ("%s patch plan: %d patch(es) one after the other in table order, %d inner worker(s) each (budget mp=%d; %s)"
                    % (label, self.n, self.inner, self.mp, self.reason)) + ("\n" + self.warning if self.warning else "")
        s = ("%s patch plan: %d patches on %d lane(s) with %s inner worker(s) (budget mp=%d, %d workers busy at most; largest patch first; %s)"
             % (label, self.n, len(self.lanes), "+".join(str(c) for c in self.lanes), self.mp, sum(self.lanes), self.reason))
        if self.est_wall and self.est_serial:
            s += "; estimated speed-up over one patch at a time with %d inner workers: %.1fx" % (self.mp, self.est_serial / self.est_wall)
        if self.cap is not None:
            s += "; memory room for %d lane(s)" % self.cap
        return s + ("\n" + self.warning if self.warning else "")


def plan_workers(costs, mp_budget, patch_workers=None, max_lanes=None, min_inner=1):
    """Split the CPU budget `mp_budget` over lanes for patches with the estimated `costs` (bins). Returns a :class:`PatchPlan`.

    ``patch_workers``: 1 = the old serial behaviour; N > 1 = N lanes (limited by the budget and the number of patches); 0 / None = auto.
    Auto simulates one lane of `mp` and 2 .. min(n, mp, max_lanes) lanes and takes the fewest lanes within 5 % of the shortest estimated wall time.
    ``max_lanes``: what the memory of the job allows (see :func:`lane_cap`); None = not limited. An explicit N above it is kept and warned about.
    ``min_inner``: fewest inner workers of a lane when there is more than one (IFU_ExGal: 2, because the PPXF / EMIPPXF / LS modules have a pool path, the one
    that always ran in production, and a serial path for one worker that is a different piece of code); a budget of 1 stays the serial loop.
    """
    n = len(costs)
    budget = max(1, int(mp_budget or 1))
    pw = int(patch_workers or 0)
    if pw < 0:
        pw = 0
    cap = None if max_lanes is None else max(1, int(max_lanes))
    if n == 0:
        return PatchPlan(budget, 0, [], [], "nothing to do", cap=cap)
    one = [budget]
    t_one = simulate(costs, one, list(range(n)))

    def serial(why):
        return PatchPlan(budget, n, one, list(range(n)), why, t_one, t_one, cap=cap)
    if budget == 1:
        return serial("mp=1")
    if pw == 1:
        return serial("patch_workers=1")
    if n == 1:
        return PatchPlan(budget, n, one, [0], "one patch", t_one, t_one, cap=cap)
    order = lpt_order(costs)
    if pw > 1:
        p = max(1, min(pw, n, budget // max(1, int(min_inner))))
        why = "patch_workers=%d" % pw + ("" if p == pw else " limited to %d by the budget / the number of patches" % p)
        warning = None
        if cap is not None and p > cap:
            warning = ("WARNING: %d patches at a time may need more memory than this job has (room for about %d); lower --patch_workers or ask for more memory"
                       % (p, cap))
        if p == 1:
            return serial(why)
        lanes = split_even(budget, p)
        return PatchPlan(budget, n, lanes, order, why, simulate(costs, lanes, order), t_one, cap=cap, warning=warning)
    room = max(1, min(n, budget // max(1, int(min_inner))))
    top = room if cap is None else max(1, min(room, cap))
    walls = []
    for p in range(1, top + 1):
        lanes = split_even(budget, p)
        walls.append((simulate(costs, lanes, order if p > 1 else list(range(n))), p, lanes))
    best = min(w[0] for w in walls)
    for wall, p, lanes in walls:
        if wall <= best * AUTO_TOLERANCE:
            if p == 1:
                return serial("auto: one lane is as fast" + ("" if cap is None or top >= room else "; memory allows %d lanes" % cap))
            why = "auto" + ("" if cap is None or top >= room else ", memory allows %d lanes" % cap)
            return PatchPlan(budget, n, lanes, order, why, wall, t_one, cap=cap)
    return serial("auto")


# ------------------------------------------------------------------------------------------------------------------------------------------
class LinePrefixer(object):
    """A text stream that writes whole lines with a prefix, so that the lines of patches running side by side stay readable."""

    def __init__(self, raw, prefix):
        self.raw, self.prefix, self._buf = raw, prefix, ""
        import threading
        self._lock = threading.Lock()

    def write(self, text):
        if not isinstance(text, str):
            text = text.decode("utf-8", "replace")
        with self._lock:
            self._buf += text
            if "\n" not in self._buf:
                return len(text)
            *lines, self._buf = self._buf.split("\n")
            out = "".join("%s%s\n" % (self.prefix, ln) for ln in lines)
        try:
            self.raw.write(out)
            self.raw.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        with self._lock:
            rest, self._buf = self._buf, ""
        try:
            if rest:
                self.raw.write("%s%s\n" % (self.prefix, rest))
            self.raw.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):                       # fileno, encoding, ... of the raw stream
        return getattr(self.raw, name)


def failed_result(task, why, outcome="failed", fault=True):
    """The answer of a patch that could not even report (worker died, SystemExit)."""
    return {"id": task.get("id"), "pos": task.get("pos"), "outcome": outcome, "fault": fault, "proper_drop": False, "rvs_ok": None, "ferre_ok": None,
            "seconds": None, "error": why}


def _limit_threads():
    """One BLAS/OpenMP thread in a lane (the runner exports OMP_NUM_THREADS=1; a standalone user may not have)."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _lane_main(lane, capacity, work, tasks, taskq, conn):
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # the parent turns SIGTERM into SystemExit (to clean up); a lane must simply die when terminated
    except Exception:
        pass
    _limit_threads()
    try:                                               # lanes forked from one parent must not share the random stream (PPXF Monte Carlo errors)
        import numpy as np
        np.random.seed(None)
    except Exception:
        pass
    raw_out, raw_err = sys.stdout, sys.stderr
    try:
        while True:
            pos = taskq.get()
            if pos is None:
                break
            task = tasks[pos]
            conn.send(("start", pos, os.getpid()))     # a pipe, not a queue: written before the next line runs, so a killed lane cannot lose it
            prefix = "[P%04d] " % int(task["id"]) if str(task.get("id", "")).lstrip("-").isdigit() else "[%s] " % task.get("id")
            sys.stdout, sys.stderr = LinePrefixer(raw_out, prefix), LinePrefixer(raw_err, prefix)
            t0 = time.time()
            try:
                res = work(task, capacity)
            except KeyboardInterrupt:                  # Ctrl-C stops the lane (and so the run), it is not a failed patch
                raise
            except BaseException as exc:               # SystemExit included: one patch never stops the others
                traceback.print_exc()
                res = failed_result(task, "%s: %s" % (type(exc).__name__, exc))
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                sys.stdout, sys.stderr = raw_out, raw_err
            res = dict(res)
            if res.get("seconds") is None:
                res["seconds"] = round(time.time() - t0, 1)
            conn.send(("done", pos, res))
    finally:
        try:
            conn.send(("exit", None, None))
            conn.close()
        except Exception:
            pass


def can_fork():
    return "fork" in mp.get_all_start_methods()


def run_patches(tasks, work, plan, label="IFU", out=None):
    """Run ``work(task, inner)`` for every task and return the answers in the order of `tasks`.

    tasks   list of dicts with at least ``id`` (the patch id) and ``pos`` (its position in this list)
    work    callable ``work(task, inner) -> dict`` (see aps_ifu_Gal._run_gal_patch); it should catch its own Exceptions; one that raises
            anything in a lane is reported as a failed patch
    plan    :class:`PatchPlan`; one lane runs in this process in table order (the old serial loop: exceptions of `work` propagate as before)
    """
    out = out or sys.stdout
    if not tasks:
        return []
    for i, t in enumerate(tasks):
        t["pos"] = i
    if plan.serial or not can_fork():
        if not plan.serial:
            print("WARNING: %s: no fork start method on this platform, patches run one after the other" % label, file=out)
        return [work(t, plan.inner if plan.serial else max(1, plan.mp)) for t in tasks]

    from multiprocessing import connection
    ctx = mp.get_context("fork")
    taskq = ctx.SimpleQueue()                          # no feeder thread: nothing is half-sent when the lanes are forked
    for pos in plan.order:
        taskq.put(pos)
    for _ in range(len(plan.lanes) + len(tasks)):      # end markers for the lanes and for replacements
        taskq.put(None)
    procs, caps, readers = {}, {}, {}                  # lane -> Process, lane -> capacity, reader connection -> lane
    inflight, started, results, ended = {}, set(), {}, set()
    respawns = [0]

    def start(lane, cap):
        reader, writer = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_lane_main, args=(lane, cap, work, tasks, taskq, writer), name="patch-lane-%d" % lane)
        p.daemon = False                               # a lane starts its own pools (RVS, PPXF, ...): daemonic processes may not have children
        p.start()
        writer.close()                                 # only the lane holds the writing end: its death is an EOF here
        procs[lane], caps[lane], readers[reader] = p, cap, lane

    def lane_gone(reader):
        lane = readers.pop(reader)
        try:
            reader.close()
        except Exception:
            pass
        procs[lane].join(timeout=5)
        pos = inflight.pop(lane, None)
        if pos is not None and pos not in results:
            code = procs[lane].exitcode
            results[pos] = failed_result(tasks[pos], "worker process died (exit code %s)" % code)
            print("** %s FAILED  id=%s: worker process died (exit code %s)" % (label, tasks[pos].get("id"), code), file=out)
        if lane not in ended and len(started) < len(tasks) and respawns[0] < len(tasks):     # patches are left: a new lane takes its place
            respawns[0] += 1
            start(max(procs) + 1, caps[lane])
        ended.add(lane)

    old_term = None
    try:
        old_term = signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(SystemExit(128 + s)))
    except Exception:
        old_term = None
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        for lane, cap in enumerate(plan.lanes):
            start(lane, cap)
        while len(results) < len(tasks) and readers:
            for reader in connection.wait(list(readers), timeout=5.0):
                lane = readers.get(reader)
                try:
                    kind, pos, payload = reader.recv()
                except (EOFError, OSError):
                    lane_gone(reader)
                    continue
                if kind == "start":
                    inflight[lane] = pos
                    started.add(pos)
                elif kind == "done":
                    inflight.pop(lane, None)
                    results[pos] = payload
                elif kind == "exit":
                    ended.add(lane)
        for pos, t in enumerate(tasks):                # every lane is gone and a patch has no answer: it was never run
            if pos not in results:
                results[pos] = failed_result(t, "not run: all workers are gone")
    finally:
        for p in procs.values():
            if p.is_alive():
                p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        if old_term is not None:
            try:
                signal.signal(signal.SIGTERM, old_term)
            except Exception:
                pass
    return [results[i] for i in range(len(tasks))]


def apply_results(results, tally, proper_targets):
    """Add the answers of the patches (in table order) to the PatchTally; return ``(proper_targets, fault_counter)`` as the serial loops kept them."""
    fault = 0
    for r in results:
        oc = r.get("outcome")
        if oc == "written":
            tally.written()
        elif oc == "no_valid_spaxels":
            tally.no_valid_spaxels()
        else:
            tally.failed()
        if r.get("proper_drop"):
            proper_targets -= 1
        if r.get("fault"):
            fault += 1
    return proper_targets, fault
