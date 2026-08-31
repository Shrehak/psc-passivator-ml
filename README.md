# PSC Passivator Workfunction Prediction

Reproducing and extending the molecular-property-prediction stage of:
Gao, D., Lu, S., Zhang, C. et al. "Autonomous closed-loop framework for
reproducible perovskite solar cells." Nature (2026).

## What this does
Given a candidate passivator molecule (as a SMILES string), predicts its workfunction
plus a calibrated confidence estimate -- so molecules can be shortlisted before
spending lab time synthesizing and testing them.

## Usage
    import sys
    sys.path.insert(0, "src")
    from predict import predict_workfunction

    predict_workfunction("Nc1ccccc1")
    # -> {'smiles': 'Nc1ccccc1', 'predicted_workfunction_eV': 4.654, 'confidence_std_eV': 0.32}

## Results

### Headline (Leave-One-Out Cross-Validation)
With only 212 molecules, a single train/test split proved unreliable (R2 varied
0.39-0.55 across different random splits). Switched to LOOCV -- train on 211
molecules, test on the 1 left out, repeat for every molecule -- as the most
rigorous validation choice for a dataset this small.

- **LOOCV R2: 0.492**
- **LOOCV MAE: 0.288 eV**
- Dataset: 212 molecules from the paper's public data
- Features: 63 RDKit descriptors (from an original 130, after dropping
  constant, redundant, and target-uncorrelated columns)

### Model comparison (same LOOCV setup, all models)
| Model              | R2    | MAE (eV) |
|---------------------|-------|----------|
| SVR                  | 0.524 | 0.279 |
| Ridge                | 0.521 | 0.277 |
| XGBoost              | 0.496 | 0.286 |
| **NGBoost (chosen)** | 0.492 | 0.288 |
| Random Forest        | 0.482 | 0.293 |
| Linear Regression    | 0.426 | 0.290 |

NGBoost did not top the accuracy table -- SVR and Ridge both edge it out
by a small margin (within noise given the sample size). NGBoost was chosen
anyway because it natively outputs a predictive distribution, not just a
point estimate, which is directly useful for molecule shortlisting: knowing
the model is confident about one candidate and unsure about another is
actionable information the other five models don't provide.

### Uncertainty calibration
NGBoost's raw confidence estimates were checked against actual LOOCV coverage
and found to be significantly overconfident:

| Claimed confidence | Raw actual coverage |
|---------------------|---------------------|
| 90% | 21.2% |
| 95% | 25.9% |

This traces to the hyperparameter search optimizing purely for point-prediction
accuracy (MAE) -- nothing in tuning ever rewarded calibrated uncertainty.

**Fix applied**: post-hoc recalibration by scaling predicted std by 7.95x
(fit via LOOCV coverage matching). After recalibration:

| Claimed confidence | Recalibrated coverage |
|---------------------|------------------------|
| 90% | 91.0% |
| 95% | 93.9% |

This scale factor is baked into `predict.py` -- confidence values returned by
`predict_workfunction()` are the calibrated, verified numbers, not NGBoost's
raw (overconfident) internal estimate.

### Feature importance (SHAP)
Top predictors: **Dipole_Moment** and **HOMO energy** -- both chemically
sensible for workfunction, since it fundamentally depends on how easily an
electron leaves the surface (HOMO) and how the molecule's charge distribution
shapes the local electric field (dipole moment). Direction of effect also
matches physical expectation: higher dipole moment pushes predicted
workfunction up; higher HOMO pushes it down.

## Known limitations
- Small dataset (212 molecules) means even LOOCV has real uncertainty in
  the reported R2/MAE
- Model comparison gaps (SVR/Ridge vs NGBoost) are small enough to be within
  noise at this sample size, not a strong claim of NGBoost's superiority
- HOMO and Dipole_Moment (the two most important features) require quantum
  chemistry calculations, not just RDKit -- a from-scratch prediction on a
  brand-new molecule can't use them without an extra computation step
- Confidence estimates are now calibrated for the population as a whole, but
  per-molecule calibration (e.g. for molecules very different from the
  training set) hasn't been separately verified

## Project structure
    data/       raw dataset
    src/        data_loading.py, train_model.py, predict.py
    models/     trained model, scaler, feature list, uncertainty scale factor
    notebooks/  step-by-step development notebook
    results/    key plots (SHAP, model comparison)

## Next steps
- Swap in real experimental passivator data once available
- Consider adding quantum-chemistry features back in for accuracy, at the
  cost of losing instant prediction
- Per-molecule calibration checking for out-of-distribution candidates
