# WhatsApp-community-factchecking-colombia

Supplementary materials for the WhatsApp community fact-checking study in Colombia.

## Contents

- `analysis/` — data and code that reproduce the numbers and figures in the paper.
  - `long.csv` — the 662 analyzed judgments: participant study code (P01–P38), item (LC1–LC40), schedule, position, response, correctness, and participant covariates (gender, years of education, political self-placement and group, internet use, religiosity, cognitive reflection answers, age in whole years).
  - `items.csv` — veracity, modality, genre, and partisan slant of each item.
  - `aggregation_curves.csv`, `R.npy`, `lab.npy` — outputs of `aggregation.py` that `make_figures.py` reads.
  - `numbers.json`, `numbers_extra.json` — the statistics reported in the paper, as produced for it.
  - Scripts (Python 3 with pandas, numpy, scipy, statsmodels, matplotlib): `aggregation.py`, `analysis_main.py`, `cv_weighting.py`, `bayes_mixed.py`, `make_figures.py`, `extra_analyses.py`. `build_long.py` documents how `long.csv` was built from the raw files, which are not shared.
- `Demographic Data.xlsx` — pre-survey answers of the 36 participants who completed it, using the same study codes as `long.csv`, with a data dictionary.
- `codebook.md` — interview codebook: 30 codes in four themes, definitions, participant-level assignments, and quotes in Spanish with English translations.
- `Interview Notes - WoC Pilot.docx` — narrative notes on the nine interviews (P01–P09).
- `Interviews Transcription (Spanish).docx` — Spanish transcripts of the interviews, redacted as described below.
- `stimuli/` — the items shown to participants, by week. `stimuli/index.csv` maps each file to its item code in the paper, with veracity, format, genre, slant, a short description, and the fact-check used for false items.

## Reproducing the results

From the repository root:

```
cd analysis && python aggregation.py && python analysis_main.py && python cv_weighting.py && cd ..
python analysis/make_figures.py
python analysis/extra_analyses.py
```

`make_figures.py` and `extra_analyses.py` also run on their own with the files provided. Figures are written to `img/`.

- Ages are given in whole years, so the mean age and the age terms of the demographic regression differ slightly from the paper, which used exact ages (mean 33.1 rather than 33.6). All other values in `numbers.json` and `numbers_extra.json` are reproduced exactly from the files provided.
- Re-running `aggregation.py` repeats the crowd simulations, and points on the crowd-size curves can then differ from the paper's by up to about 0.004.

## Redactions and withheld material

- Transcripts: the interviewer's name, the names of participants and of other private individuals they mention, and the names of villages are replaced by "Entrevistador", "[nombre]", and "[localidad]". Public figures and the municipality are not redacted.
- Demographics: dates of birth and response timestamps are removed, and age is given in whole years. WhatsApp numbers are replaced by study codes. One pre-survey that could not be linked to any participant's judgments is not included.
- Stimuli: four items (LC13, LC22, LC31, LC35) are withheld because they show the names, faces, or contact details of third parties; `stimuli/index.csv` and the paper's appendix describe them.
- Not shared: interview audio recordings, WhatsApp numbers and their mapping to study codes, the raw fieldwork spreadsheets, and donated chat content other than the stimuli.
