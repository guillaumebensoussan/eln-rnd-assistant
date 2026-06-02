# Model Card — ELN Yield & QC Predictor

## Purpose
Predict reaction `yield_pct` (regression) and `qc_pass` (classification) from synthesis parameters, to give R&D chemists an expected outcome before running an experiment.

## Training data
Synthetic R&D dataset (synthesis subset). Features: temperature_c, time_h, catalyst_loading_molpct, pressure_bar, purity_in_pct, solvent.

## Metrics
- Best regressor: LinearRegression (R2=0.8043, MAE=4.7625)
- Best classifier: LogisticRegression (acc=0.8667, F1=0.7143)

## Limitations
- Trained on synthetic data with a known latent model; real lab data will differ.
- Does not extrapolate beyond observed parameter ranges.
- Solvent encoded as categorical; unseen solvents are ignored.

## Ethical considerations
- Decision support only — does not replace chemist judgement or safety review.
- No personal data; operator field is a pseudonymous label.
