# PyAPS_local

Your personal, machine-specific scratch area for your own development work. Everything in this
directory except this file is ignored by git.

The official development and production runs never read anything from here: they take every
location (data, templates, calibration, external programs) from the configuration files in
`configs/` (`script_params*.yaml`, filled from `configs/script_params.yaml.example`). If you want to
use files from this directory for your own experiments, point your own config file at them.
