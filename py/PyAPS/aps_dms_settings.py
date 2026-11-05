"""aps_dms_settings.py - the pod's own configured values of the machine-check settings, and the ONE precedence chain every consumer uses.

Why it exists. On a pod the DMS (``aps_dms`` / ``pyaps_admin``) is configured in ``~/.private_env_configs/PyAPS_dms_config.json`` and passes
what it finds there to the runner on the command line (``--slurm_profile`` ...). A tool that does NOT go through the DMS (``aps_slurm_resources.py
check|show|explain|site``, a runner started by hand, ``aps-offline``) used to see only the environment and ``script_params.yaml`` and so printed
the table's default profile although the pod runs another one. ``script_params.yaml`` is no place for the value: it is RENDERED from the public
template by ``pyaps-admin local`` (every 15 minutes, as soon as the pod is idle) and a key added by hand is overwritten (doc/SLURM_RESOURCES.md,
section "How the profile is resolved"). The DMS config is host-local and never rendered.

What it reads. Exactly five keys of the DMS config, and nothing else of it (the file holds the database password)::

    slurm_profile  plot_export  slurm_limits  slurm_site  max_inflight_obs

Order, for the four settings that have an environment variable and a script_params key (first one that is set wins)::

    option on the command line (--slurm_profile ...)  >  environment variable  >  DMS config key  >  script_params key  >  site / profile / table default

``max_inflight_obs`` is the DMS's own cap on the observations in flight; the runner does not use it, the tools only report it.

Where the file is. ``PYAPS_DMS_CONFIG`` (a path; ``none`` or empty switches the lookup off), else ``$HOME/.private_env_configs/PyAPS_dms_config.json``. The
HOME is the one of the ``env`` mapping the caller passes (the real environment by default), so a caller that passes ``env={}`` (the tests) never sees a
pod's file. A missing, unreadable or malformed file is an empty answer: this module never raises and never prints a value it was not asked for.

Only the standard library is used (it is imported by the runner, by ``aps_slurm_resources`` and by ``apsPlot.export``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__version__ = "1.0.0"

DMS_CONFIG_NAME = "PyAPS_dms_config.json"
DMS_KEYS = ("slurm_profile", "plot_export", "slurm_limits", "slurm_site", "max_inflight_obs")

# setting -> (the label of its command-line option, its environment variable, its key in script_params AND in the DMS config)
SETTINGS = {
    "profile": ("--slurm_profile", "PYAPS_SLURM_PROFILE", "slurm_profile"),
    "limits": ("option", "PYAPS_SLURM_LIMITS", "slurm_limits"),
    "site": ("option", "PYAPS_SLURM_SITE", "slurm_site"),
    "plot_export": ("option --plot_export", "PYAPS_PLOT_EXPORT", "plot_export"),
}
# words that mean "nothing was said" for each setting (the next source decides)
_UNSET_WORDS = {"profile": ("", "auto", "none"), "limits": ("", "none"), "site": ("", "none"), "plot_export": ("", "auto", "default")}


class DmsSettings(dict):
    """The (up to five) keys of the DMS config that are present and not empty, plus where they came from. ``path`` is None when no file was read."""
    path = None
    error = None


def dms_config_path(env=None):
    """The DMS config file to read (a :class:`Path`), or None when the lookup is off or there is nothing to look at."""
    env = os.environ if env is None else env
    explicit = env.get("PYAPS_DMS_CONFIG")
    if explicit is not None:
        word = str(explicit).strip()
        if word.lower() in ("", "none", "off"):
            return None
        return Path(os.path.expandvars(os.path.expanduser(word)))
    home = env.get("HOME")
    if not home:
        return None
    return Path(home) / ".private_env_configs" / DMS_CONFIG_NAME


def read_dms_settings(env=None, path=None) -> DmsSettings:
    """The five keys of the DMS config (:data:`DMS_KEYS`) that are set. Reads nothing else of the file; never raises."""
    out = DmsSettings()
    path = Path(path) if path is not None else dms_config_path(env)
    if path is None:
        return out
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return out
    except (OSError, ValueError) as exc:                       # unreadable, or not JSON: say so without quoting the file
        out.error = "%s could not be read (%s)" % (path, type(exc).__name__)
        return out
    out.path = str(path)
    if not isinstance(data, dict):
        out.error = "%s is not a JSON object" % path
        return out
    for key in DMS_KEYS:
        val = data.get(key)
        if val is not None and str(val).strip() != "":
            out[key] = val
    del data                                                   # the rest of the file (credentials) is not kept anywhere
    return out


def is_set(setting, value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in _UNSET_WORDS.get(setting, ("",))


def chain(setting, cli=None, conf=None, env=None, dms=None):
    """Every source of ``setting`` that is set, in order of precedence: ``[(label, value), ...]``. The first one wins; the others are what it overrides.

    ``dms`` is a :class:`DmsSettings` (or any dict with the DMS keys); None reads the DMS config of the ``env``."""
    cli_label, env_var, key = SETTINGS[setting]
    env = os.environ if env is None else env
    conf = conf or {}
    if dms is None:
        dms = read_dms_settings(env)
    cands = [(cli_label, cli), ("environment " + env_var if setting == "plot_export" else env_var, env.get(env_var)),
             ("DMS config " + key, dms.get(key)), ("script_params " + key, conf.get(key))]
    return [(label, val) for label, val in cands if is_set(setting, val)]


def first(setting, cli=None, conf=None, env=None, dms=None):
    """``(value, label)`` of the source that decides, or ``(None, None)``."""
    got = chain(setting, cli, conf, env, dms)
    return (got[0][1], got[0][0]) if got else (None, None)


def overridden(setting, cli=None, conf=None, env=None, dms=None):
    """The sources that are set but lose to a stronger one: ``[(label, value), ...]``."""
    return chain(setting, cli, conf, env, dms)[1:]
