import pandas as pd
import numpy as np


def load_raw_data(path, target_col):
    """Load the dataset, split into features (X) and target (y)."""
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    X = df.drop(columns=["ID", "SMILES", target_col])
    y = df[target_col]
    return X, y


def clean_features(X, y, redundant_threshold=0.90, target_corr_threshold=0.05):
    """Drop constant, redundant, and target-uncorrelated columns."""
    start_n = X.shape[1]

    nunique = X.nunique()
    X = X.loc[:, nunique > 2]
    print(f"After dropping constant columns: {X.shape[1]} (was {start_n})")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant = [c for c in upper.columns if any(upper[c] > redundant_threshold)]
    n_before = X.shape[1]
    X = X.drop(columns=redundant)
    print(f"After dropping redundant columns: {X.shape[1]} (was {n_before})")

    target_corr = X.corrwith(y).abs()
    n_before = X.shape[1]
    X = X.loc[:, target_corr > target_corr_threshold]
    print(f"After dropping target-uncorrelated columns: {X.shape[1]} (was {n_before})")

    return X

def compute_descriptors_from_smiles(smiles, feature_names):
    """Turn a SMILES string into the same RDKit descriptors used in training."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    all_descriptors = {name: func(mol) for name, func in Descriptors.descList}

    # keep only the features our model was actually trained on, in the right order
    row = {name: all_descriptors.get(name, 0) for name in feature_names}
    return pd.DataFrame([row])
