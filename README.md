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
| `scripts/03_calibrate_simulator.py` | background spectrum and spike sharpness fitted on the real data: `results/background_fit*.csv`, `results/spike_*.csv`, `figures/sim_*_calibration.png` |
| `scripts/04_simulation_examples.py` | `figures/sim_examples.png`, `figures/sim_snr_levels.png`, `results/sim_example_truth.csv` |
| `scripts/05_spike_filter_demo.py` | why a filtered spike looks like an HFO: `figures/false_hfo_demo.png`, `results/false_hfo_demo.csv` |

The BIDS output passes the official validator (requires Node.js):

```powershell
npx bids-validator@1.15.0 data/bids
# -> "This dataset appears to be BIDS compatible": 0 errors, 0 warnings
#    (488 files, 902 MB, 22 subjects, 2 sessions, iEEG, task "intraop")
```

## Limitations

This section is completed at every step; the final README discusses each point with the results.

**Data**
- Only **fast ripples** are marked in the real data, by an automatic detector followed by visual
  validation. This reference is detector-biased, and ripples can only be evaluated on
  simulations.
- No channel-level resection labels exist, so the clinical analysis is done at the patient level.
- The pre-resection recordings of patients 12, 15 and 17 are byte-identical in the source files.
  They are excluded.
- **P22 pre-resection, channel 3-4** contains duplicated samples: every sample appears twice,
  so it was effectively sampled at 1000 Hz and up-sampled to 2000 Hz. Its effective Nyquist
  frequency is 500 Hz, so it cannot carry fast ripples. It is flagged by the quality control
  and excluded from fast-ripple analyses.
- Small cohort: 22 patients, of whom 8 were not seizure-free.

**Simulator**
- The background spectrum (1/f^β with a knee, plus a white amplifier floor) and the spike-apex
  sharpness are calibrated on the real data. The spike shape itself (amplitudes, time
  constants, slow wave) is a modelling choice.
- The real anti-aliasing roll-off above 500 Hz and mains interference are not simulated.
- One channel per simulation: no spatial correlation between channels.
