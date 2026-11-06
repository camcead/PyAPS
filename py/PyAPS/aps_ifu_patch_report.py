"""aps_ifu_patch_report.py -- what happened to the patches of an IFU_ExGal / IFU_Gal run (stdlib only).

Why (OB 26133, 2026-11-06): 226 patch rows went into IFU_ExGal / IFU_Gal, 222 of them logged
"No valid spaxels remaining after filters!" and were skipped, and both jobs still ended COMPLETED
with no number anywhere that said how many patches were attempted, written or lost. Now every run
counts them and says so:

    IFU_PATCH_SUMMARY {"stage": "ExGal", "attempted": 226, "written": 4, "no_valid_spaxels": 222, "failed": 0, ...}

printed on stdout (grep-able in the job log), written to `<outpath>/<headname>_<stage>_patches.json`,
with a WARNING banner when less than half of the attempted patches produced a product. The job fails
(exit status != 0, like the existing "all targets failed" rule) only when at least MIN_ATTEMPTED_TO_FAIL
patches were attempted and NONE produced a product."""
import json
import sys
from pathlib import Path

MIN_ATTEMPTED_TO_FAIL = 10
LOW_YIELD = 0.5


class PatchTally(object):
    def __init__(self, stage, outpath, headname):
        self.stage, self.outpath, self.headname = stage, outpath, headname
        self.n = {"attempted": 0, "written": 0, "no_valid_spaxels": 0, "failed": 0}

    def attempt(self):
        self.n["attempted"] += 1

    def written(self):
        self.n["written"] += 1

    def no_valid_spaxels(self):
        self.n["no_valid_spaxels"] += 1

    def failed(self):
        self.n["failed"] += 1

    def summary(self):
        s = dict(self.n)
        s["stage"] = self.stage
        s["headname"] = str(self.headname)
        a = s["attempted"]
        s["yield"] = round(s["written"] / a, 3) if a else None
        return s

    def finish(self, out=None):
        """Print and write the summary; return True when the job should FAIL (nothing written from many patches)."""
        out = out or sys.stdout
        s = self.summary()
        print("IFU_PATCH_SUMMARY " + json.dumps(s, sort_keys=True), file=out)
        a = s["attempted"]
        if a and s["written"] / a < LOW_YIELD:
            print("WARNING: IFU_%s produced a product for only %d of %d attempted patches (%d without valid spaxels, "
                  "%d failed). Normal for a few patches; this many means the patch table holds apertures without "
                  "signal -- check SEG2D_SUMMARY of IFU_PR and the patch table." % (
                      self.stage, s["written"], a, s["no_valid_spaxels"], s["failed"]), file=out)
        try:
            Path(self.outpath).mkdir(parents=True, exist_ok=True)
            with open(Path(self.outpath) / ("%s_%s_patches.json" % (self.headname, self.stage)), "w") as fh:
                json.dump(s, fh, indent=1, sort_keys=True)
        except Exception as e:      # reporting must never break the run
            print("WARNING: patch summary not written: %s" % e, file=out)
        out.flush()
        return a >= MIN_ATTEMPTED_TO_FAIL and s["written"] == 0
