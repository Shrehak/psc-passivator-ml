# PSC Passivator Workfunction Prediction

Reproducing and extending the molecular-property-prediction stage of a recent
Nature paper on ML-guided perovskite solar cell passivator discovery.

## Two versions of the model, two different purposes

This project actually contains **two models**, evaluated and used differently:

1. **Full-feature model (63 descriptors, includes HOMO/Dipole_Moment)** -- used
   for the headline LOOCV evaluation, the 6-model comparison, and the SHAP
   interpretability analysis below. This is the more accurate version, but
   HOMO and Dipole_Moment require quantum-chemistry calculations (not just
   RDKit), so it can't be used for instant prediction on a brand-new molecule
   without an extra computation step.
2. **RDKit-only model (60 descriptors, deployed in `predict.py`)** -- built by
   dropping HOMO/LUMO/Dipole_Moment so any new molecule's SMILES string can be
   scored instantly with no extra setup. This is a real accuracy trade-off,
   made deliberately for usability, and documented below.

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

## Results (full-feature model, 63 descriptors)

### Headline (Leave-One-Out Cross-Validation)
With only 212 molecules, a single train/test split proved unreliable (R2 varied
0.39-0.55 across different random splits). Switched to LOOCV -- train on 211
molecules, test on the 1 left out, repeat for every molecule -- as a data-efficient evaluation approach: the model trains on 211 molecules and tests on the remaining molecule, repeating this process for all 212 molecules.

- **LOOCV R2: 0.492**
- **LOOCV MAE: 0.288 eV**
- Dataset: 212 molecules from the paper's public data
- Features: 63 molecular descriptors, including quantum-chemical features
  (HOMO, LUMO, Dipole_Moment are not RDKit descriptors) (from an original 130, after dropping
  constant, redundant, and target-uncorrelated columns). Note: feature
  selection was done once on the full dataset before cross-validation, which
  can introduce mild optimism into the reported score -- a stricter version
  would re-select features inside each LOOCV fold.

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
point estimate, which is directly useful for molecule shortlisting.

### Uncertainty calibration
NGBoost's raw confidence estimates were checked against actual LOOCV coverage
and found to be significantly overconfident:

| Claimed confidence | Raw actual coverage |
|---------------------|---------------------|
| 90% | 21.2% |
| 95% | 25.9% |

**Fix applied**: post-hoc recalibration by scaling predicted std by 7.95x
(fit via LOOCV coverage matching). After recalibration:

| Claimed confidence | Recalibrated coverage |
|---------------------|------------------------|
| 90% | 91.0% |
| 95% | 93.9% |

Caveat: this scale factor was fit and evaluated on the same LOOCV predictions
-- it hasn't been independently validated on a separate held-out calibration
set, so the true out-of-sample calibration may differ somewhat from the
numbers above.

This scale factor is baked into `predict.py`'s RDKit-only model.

### Feature importance (SHAP, full-feature model)
Top predictors: **Dipole_Moment** and **HOMO energy** -- both chemically
sensible for workfunction, since it fundamentally depends on how easily an
electron leaves the surface (HOMO) and how the molecule's charge distribution
shapes the local electric field (dipole moment). Direction of effect also
matches physical expectation: higher dipole moment pushes predicted
workfunction up; higher HOMO pushes it down.

## Results (deployed RDKit-only model, 60 descriptors)
Dropping HOMO/LUMO/Dipole_Moment for instant, no-extra-setup prediction costs
real accuracy: R2 drops from 0.44 (5-fold CV, full features) to 0.29 (5-fold
CV, RDKit-only). This trade-off was made deliberately -- see `predict.py`.

## Known limitations
- Small dataset (212 molecules) means even LOOCV has real uncertainty in
  the reported R2/MAE
- Feature selection was done before cross-validation, not inside each fold --
  a source of mild optimistic bias in the reported scores
- Model comparison gaps (SVR/Ridge vs NGBoost) are small enough to be within
  noise at this sample size, not a strong claim of NGBoost's superiority
- Uncertainty calibration was fit and tested on the same data (not
  independently validated on a held-out set)
- The deployed model (60 features) is less accurate than the analysis model
  (63 features) described above, by design, for usability

## Project structure
    data/       raw dataset
    src/        data_loading.py, train_model.py, predict.py
    models/     trained model, scaler, feature list, uncertainty scale factor
    notebooks/  full development notebook
    results/    key plots (SHAP, model comparison)

## Next steps
- Swap in real experimental passivator data once available
- Re-do feature selection inside each CV fold to remove optimistic bias
- Independently validate the calibration scale factor on a held-out set
- Per-molecule calibration checking for out-of-distribution candidates
