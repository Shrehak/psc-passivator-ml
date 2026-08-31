"""
Load the trained model and predict workfunction for a new molecule from its SMILES.

Usage:
    from predict import predict_workfunction
    result = predict_workfunction("Nc1ccccc1")
"""
import json
import os
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(_BASE_DIR, "models", "workfunction_model.pkl")
SCALER_PATH = os.path.join(_BASE_DIR, "models", "workfunction_scaler.pkl")
FEATURES_PATH = os.path.join(_BASE_DIR, "models", "feature_names.json")
SCALE_FACTOR_PATH = os.path.join(_BASE_DIR, "models", "uncertainty_scale_factor.json")

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    _feature_names = json.load(f)
with open(SCALE_FACTOR_PATH) as f:
    _scale_factor = json.load(f)["scale_factor"]


def predict_workfunction(smiles: str) -> dict:
    """Predict workfunction (eV) for a molecule given its SMILES string.

    Returns a dict with the prediction and a calibrated confidence (std dev),
    verified via LOOCV coverage checking rather than trusting NGBoost's raw
    (found to be overconfident) internal estimate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    all_descriptors = {name: func(mol) for name, func in Descriptors.descList}
    row = [[all_descriptors.get(name, 0) for name in _feature_names]]

    row_scaled = _scaler.transform(row)
    prediction = _model.predict(row_scaled)[0]
    raw_std = _model.pred_dist(row_scaled).dist.std()[0]
    calibrated_std = raw_std * _scale_factor

    return {
        "smiles": smiles,
        "predicted_workfunction_eV": round(float(prediction), 3),
        "confidence_std_eV": round(float(calibrated_std), 3),
    }
