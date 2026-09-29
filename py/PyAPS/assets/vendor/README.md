# Vendored third-party libraries

## aladin.js — Aladin Lite v3.8.2

Vendored (not auto-injected — see `assets_ignore` on the `Dash(...)`
constructor in `aps_explorer.py`) to fix a real, reported failure:
loading it live from `https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js`
on every 2D-map page load depends on the *user's own browser* being able
to reach that external CDN — some institutional VPNs/firewalls block or
reroute it, producing "Could not load the sky-map library... check your
network connection" with the 2D coordinate map never rendering at all.
Self-hosting the library file removes that dependency for the library
itself; the app's own server (already reachable, or nothing else would
have loaded either) serves it instead.

**Not fixed by this**: the actual DSS/HiPS background imagery tiles and
catalog cross-match lookups Aladin Lite performs *at runtime* (e.g.
`alaskybis.cds.unistra.fr`) are still separate live requests to CDS's
own infrastructure — self-hosting the *library* only fixes the
"library itself won't even load" failure, not a VPN/firewall that
blocks CDS's imagery servers specifically. A handful of small hardcoded
CDS-hosted logo/credit images (e.g. the SIMBAD watermark) are similarly
unaffected.

- **Source**: `https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js`
  (downloaded verbatim, unmodified — the same build the CDN itself was
  serving)
- **Version**: 3.8.2 (the bundle's own embedded version string)
- **License**: LGPL-3.0-or-later (Thomas Boch & Matthieu Baumann, CDS —
  https://github.com/cds-astro/aladin-lite)
- **Trade-off**: pinned, not auto-updating — CDS's own "latest" URL
  moves forward on its own; this file needs a manual re-download
  (same URL, same command) to pick up any future upstream fix/feature.
  Re-check this comment's own version number against the file's own
  embedded `"3.x.y"` string (`grep -o '"3\.[0-9.]*"' aladin.js | head -1`)
  before assuming it's still current.
