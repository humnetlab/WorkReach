"""Row-restricted fitting used for the out-of-sample evaluation.

Each model is fitted on a subset of origin rows and then predicts every origin,
so that the held-out rows are genuinely out of sample while the choice set stays
unchanged.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from evaluation import common_part_of_commuters
from models import (
    WORKREACH_INIT_PARAMS,
    DEFAULT_INIT_PARAMS,
    predict_flows_utility,
    predict_flows_gravity_pow,
    predict_flows_radiation_extended,
    predict_flows_bms_plausible,
)

MODEL_NAMES = ["WorkReach", "Gravity", "Radiation Ext", "BMS Plausible"]


def _masked(prediction, flows, row_mask):
    off_diagonal = ~np.eye(prediction.shape[0], dtype=bool)
    return (flows[row_mask], prediction[row_mask], off_diagonal[row_mask])


def _poisson_nll(prediction, flows, row_mask):
    observed, predicted, mask = _masked(prediction, flows, row_mask)
    return np.sum(predicted[mask]) - np.sum(observed[mask] * np.log(predicted[mask] + 1e-10))


def _multinomial_nll(prediction, flows, row_mask):
    observed, predicted, mask = _masked(prediction, flows, row_mask)
    return -np.sum(observed[mask] * np.log(predicted[mask] + 1e-10))


def fit_workreach_rows(prep, row_mask, city=None, transition=True):
    initial = np.array(WORKREACH_INIT_PARAMS.get(city, DEFAULT_INIT_PARAMS), dtype=float)
    bounds = [(None, None), (None, None), (None, None), (0, None), (0, None)]

    def objective(params):
        prediction = predict_flows_utility(
            prep["distance_log"], prep["eci"], prep["informality"],
            prep["home_population"], *params, transition
        )
        return _poisson_nll(prediction, prep["flows"], row_mask)

    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000, "ftol": 1e-12})
    prediction = predict_flows_utility(
        prep["distance_log"], prep["eci"], prep["informality"],
        prep["home_population"], *result.x, transition
    )
    return result.x, prediction


def fit_gravity_rows(prep, row_mask):
    bounds = [(0, None)] * 4

    def objective(params):
        prediction = predict_flows_gravity_pow(
            prep["distance"], prep["home_population"], prep["work_population"], *params
        )
        return _multinomial_nll(prediction, prep["flows"], row_mask)

    result = minimize(objective, np.array([1.0, 1.0, 1.0, 1.0]), method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 5000, "ftol": 1e-12})
    prediction = predict_flows_gravity_pow(
        prep["distance"], prep["home_population"], prep["work_population"], *result.x
    )
    return result.x, prediction


def fit_radiation_rows(prep, row_mask):
    def objective(params):
        prediction = predict_flows_radiation_extended(
            prep["distance"], prep["home_population"], prep["work_population"], params[0]
        )
        return _poisson_nll(prediction, prep["flows"], row_mask)

    result = minimize(objective, np.array([1.0]), method="L-BFGS-B", bounds=[(0, None)],
                      options={"maxiter": 5000, "ftol": 1e-12})
    prediction = predict_flows_radiation_extended(
        prep["distance"], prep["home_population"], prep["work_population"], result.x[0]
    )
    return result.x, prediction


def fit_bms_rows(prep, row_mask):
    bounds = [(0, None), (None, None), (None, None), (None, None), (0, None), (0, None)]

    def objective(params):
        prediction = predict_flows_bms_plausible(
            prep["distance"], prep["home_population"], prep["work_population"], *params
        )
        return _multinomial_nll(prediction, prep["flows"], row_mask)

    result = minimize(objective, np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000, "ftol": 1e-12})
    prediction = predict_flows_bms_plausible(
        prep["distance"], prep["home_population"], prep["work_population"], *result.x
    )
    return result.x, prediction


FIT_FUNCTIONS = {
    "WorkReach": fit_workreach_rows,
    "Gravity": fit_gravity_rows,
    "Radiation Ext": fit_radiation_rows,
    "BMS Plausible": fit_bms_rows,
}


def _quantile_bins(values, n_bins):
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    return bins.fillna(0).astype(int).values


def stratified_splits(prep, n_splits=5, n_bins=4, seed=42):
    """Fold origin zones, stratified by population, ECI and informality."""
    n_zones = prep["flows"].shape[0]
    labels = (_quantile_bins(prep["home_population"], n_bins) * 100
              + _quantile_bins(prep["eci_raw"][:, 0], n_bins) * 10
              + _quantile_bins(prep["informality_raw"][:, 0], n_bins))

    folds = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, _ in splitter.split(np.arange(n_zones), labels):
        train_mask = np.zeros(n_zones, dtype=bool)
        train_mask[train_index] = True
        folds.append((train_mask, ~train_mask))
    return folds


def cpc(observed, predicted):
    """Common part of commuters, computed on rounded predictions."""
    return common_part_of_commuters(observed, np.round(predicted))
