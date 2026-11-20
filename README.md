# EpiDetect

Reproducible benchmark of high-frequency oscillation (HFO) detectors on realistic simulations
and on public intraoperative ECoG recordings.

> Work in progress. The full README (results, figures, limitations) is written at the end of the
> project. Data notes: [docs/DATA_NOTES.md](docs/DATA_NOTES.md).

## Data

The data are **not** stored in this repository. Download the 22 `Data_Patient_XX.h5` files from
<https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO> (folder `data/`) into `data/raw/`.

## License

The **code** in this repository is released under the [MIT License](LICENSE)
(Copyright (c) 2026 Chiheb Ayari). The MIT license covers the code only: the **data** are not
part of this repository and keep their own license, **CC BY-SA 4.0**, set by their authors
(Boran et al., 2019, University Hospital Zurich). Any derived data (e.g. the BIDS conversion
produced by `scripts/01_convert_to_bids.py`) must also be shared under CC BY-SA 4.0 with
attribution.

## Quick start (Windows PowerShell)

```powershell
.\run.ps1 setup       # create .venv and install pinned dependencies
.\run.ps1 run         # run every numbered script in scripts/ in order (see below)
.\run.ps1 notebooks   # re-execute the notebooks in notebooks/
.\run.ps1 test        # unit tests (no real data needed)
.\run.ps1 lint
```

| Script | What it produces |
|---|---|
| `scripts/00_inventory_raw.py` | `docs/data_inventory.md`, a read-only inventory of the raw NIX files |
| `scripts/01_convert_to_bids.py` | `data/bids/` (BIDS iEEG, git-ignored), `results/bids_conversion.csv` |
| `scripts/02_quality_control.py` | `results/qc_*.csv`, `figures/qc_*.png`, bad channels marked in BIDS |

Optional: validate the BIDS dataset with the official validator (requires Node.js):
`npx bids-validator@1.15.0 data/bids`.
