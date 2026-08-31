from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from bayes_opt import BayesianOptimization

RANDOM_STATE = 42

def build_model(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=5):
    """Build an NGBoost model with given settings."""
    return NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                    random_state=RANDOM_STATE),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        natural_gradient=True,
        verbose=False,
        random_state=RANDOM_STATE,
    )


def tune_hyperparameters(X_train, y_train, X_val, y_val, init_points=8, n_iter=15):
    """Search for good settings using Bayesian optimization.

    init_points = how many random tries to start with
    n_iter = how many 'smart guess' tries after that
    """
    def objective(n_estimators, learning_rate, max_depth, min_samples_leaf):
        model = build_model(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        # we want to MINIMIZE error, but BayesianOptimization always MAXIMIZES,
        # so we return the negative error
        return -mean_absolute_error(y_val, preds)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "n_estimators": (50, 400),
            "learning_rate": (0.01, 0.2),
            "max_depth": (2, 6),
            "min_samples_leaf": (4, 15),
        },
        random_state=RANDOM_STATE,
        verbose=1,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max["params"]
    return {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": best["learning_rate"],
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }

def cross_validate(X, y, params, n_splits=5):
    """Test the model on 5 different splits and average the results."""
    from sklearn.model_selection import KFold
    import numpy as np

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    Xarr, yarr = X.values, y.values
    r2_scores, mae_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(Xarr), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xarr[train_idx])
        Xte = scaler.transform(Xarr[test_idx])

        model = build_model(**params)
        model.fit(Xtr, yarr[train_idx])
        preds = model.predict(Xte)

        r2 = r2_score(yarr[test_idx], preds)
        mae = mean_absolute_error(yarr[test_idx], preds)
        r2_scores.append(r2)
        mae_scores.append(mae)
        print(f"Fold {fold}: R2 = {r2:.3f}   MAE = {mae:.4f}")

    print(f"\nAverage R2:  {np.mean(r2_scores):.3f} (+/- {np.std(r2_scores):.3f})")
    print(f"Average MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")
    return r2_scores, mae_scores
