# PyAPS_local

Your machine-specific working area. Everything in this directory except this file is
ignored by git.

Suggested layout (this is what `configs/script_params.yaml` expects by default):

```
PyAPS_local/
├── PyAPS_data/            # or a symlink to your real data volume
│   ├── L1/  L2/  CS/  CAL/  CAT/
└── PyAPS_templates/       # classification / RVS / FERRE / ExGal templates
    ├── templates_RR/  templates_ARC_RR/  templates_RVS/  templates_FR/  templates_ExGal/
```

A minimal template set can be downloaded as described in the top-level README ("Data
Resources"). Set `PYAPS_HOME` if you keep this tree somewhere other than the checkout.
