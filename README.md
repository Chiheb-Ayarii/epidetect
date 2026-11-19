# EpiDetect

Reproducible benchmark of high-frequency oscillation (HFO) detectors on realistic simulations
and on public intraoperative ECoG recordings.

> Work in progress. The full README (results, figures, limitations) is written at the end of the
> project. Data notes: [docs/DATA_NOTES.md](docs/DATA_NOTES.md).

## Data

The data are **not** stored in this repository. Download the 22 `Data_Patient_XX.h5` files from
<https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO> (folder `data/`) into `data/raw/`.

## Quick start (Windows PowerShell)

```powershell
.\run.ps1 setup       # create .venv and install pinned dependencies
.\run.ps1 inventory   # regenerate docs/data_inventory.md from data/raw/
.\run.ps1 test
```
