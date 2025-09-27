import numpy as np
from scipy import optimize
from typing import Tuple, Optional

# ======================
# Gravity Models
# ======================

def gravity_model_pow_log_likelihood(
    params: np.ndarray,
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    flows: np.ndarray
) -> float:
    """Negative log-likelihood excluding self-loops, single normalization after zeroing diag."""
    k, alpha, beta, gamma = params
    pred = k * (origin_mass[:,None]**alpha) * (destination_mass[None,:]**beta) / (distance + 1e-10)**gamma

    np.fill_diagonal(pred, 0.0)

    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    pred = (pred / row_sums) * origin_mass[:,None]

    mask = ~np.eye(pred.shape[0], dtype=bool)
    f = flows[mask];  p = pred[mask]
    return -np.sum(f * np.log(p + 1e-10))

def predict_flows_gravity_pow(
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    k: float,
    alpha: float,
    beta: float,
    gamma: float
) -> np.ndarray:
    """Singly-constrained gravity with power law, matching gravity.py."""
    pred = k * (origin_mass[:,None]**alpha) * (destination_mass[None,:]**beta) / (distance + 1e-10)**gamma
    np.fill_diagonal(pred, 0.0)
    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    return (pred / row_sums) * origin_mass[:,None]

def optimize_gravity_pow_model(
    distance: np.ndarray,
    home_population: np.ndarray,
    work_population: np.ndarray,
    flows: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize the gravity model with power law decay."""
    initial_params = [1.0, 1.0, 1.0, 1.0]

    bounds = [
        (0, None),  # k
        (0, None),  # alpha
        (0, None),  # beta
        (0, None),  # gamma
    ]

    result = optimize.minimize(
        gravity_model_pow_log_likelihood,
        initial_params,
        args=(distance, home_population, work_population, flows),
        bounds=bounds,
        method='L-BFGS-B'
    )

    optimized_params = result.x

    predicted_flows = predict_flows_gravity_pow(
        distance, home_population, work_population, *optimized_params
    )

    return optimized_params, predicted_flows

# ======================
# Extended Radiation Model
# ======================

def radiation_extended_log_likelihood(
    params: np.ndarray,
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    flows: np.ndarray
) -> float:
    """Negative Poisson log-likelihood for the extended radiation model."""
    alpha, = params
    d = distance + 1e-10
    om = origin_mass + 1e-10
    dm = destination_mass + 1e-10
    n = len(om)
    raw = np.zeros_like(d)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_ij = np.sum(dm[d[i] < d[i, j]])
            num = (om[i] + dm[j] + s_ij)**alpha - (om[i] + s_ij)**alpha
            den = (om[i]**alpha + 1) * ((om[i] + s_ij)**alpha + 1) * ((om[i] + dm[j] + s_ij)**alpha + 1)
            raw[i, j] = om[i] * num / (den + 1e-10)
    np.fill_diagonal(raw, 0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    pred = raw / row_sums * origin_mass[:, None]
    mask = ~np.eye(n, dtype=bool)
    f = flows[mask]
    p = pred[mask]
    return np.sum(p) - np.sum(f * np.log(p + 1e-10))

def predict_flows_radiation_extended(
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Predict flows using the extended radiation model."""
    d = distance + 1e-10
    om = origin_mass + 1e-10
    dm = destination_mass + 1e-10
    n = len(om)
    raw = np.zeros_like(d)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_ij = np.sum(dm[d[i] < d[i, j]])
            num = (om[i] + dm[j] + s_ij)**alpha - (om[i] + s_ij)**alpha
            den = (om[i]**alpha + 1) * ((om[i] + s_ij)**alpha + 1) * ((om[i] + dm[j] + s_ij)**alpha + 1)
            raw[i, j] = om[i] * num / (den + 1e-10)
    np.fill_diagonal(raw, 0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    return raw / row_sums * origin_mass[:, None]

def optimize_radiation_extended(
    distance: np.ndarray,
    home_population: np.ndarray,
    work_population: np.ndarray,
    flows: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Optimize the extended radiation model's alpha."""
    initial_alpha = [1.0]
    bounds = [(0, None)]
    result = optimize.minimize(
        radiation_extended_log_likelihood,
        initial_alpha,
        args=(distance, home_population, work_population, flows),
        bounds=bounds,
        method='L-BFGS-B'
    )
    alpha_opt = result.x[0]
    predicted = predict_flows_radiation_extended(
        distance, home_population, work_population, alpha_opt
    )
    return alpha_opt, predicted

# ======================
# BMS Plausible Model
# ======================

def bms_plausible_model_log_likelihood(
    params: np.ndarray,
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    flows: np.ndarray
) -> float:
    """Negative log-likelihood excluding self-loops."""
    A, B, C, D, alpha, gamma = params

    om = origin_mass[:,None]
    dm = destination_mass[None,:]
    inside = B * ((om * dm + C*dm + D) / (distance + 1e-10)**alpha) + 1.0

    pred = A * (inside ** gamma)

    np.fill_diagonal(pred, 0.0)

    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    pred = (pred / row_sums) * origin_mass[:,None]

    mask = ~np.eye(pred.shape[0], dtype=bool)
    f = flows[mask];  p = pred[mask]
    return -np.sum(f * np.log(p + 1e-10))

def predict_flows_bms_plausible(
    distance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray,
    A: float,
    B: float,
    C: float,
    D: float,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """Singly-constrained BMS plausible model."""
    om = origin_mass[:,None]
    dm = destination_mass[None,:]
    inside = B * ((om * dm + C*dm + D) / (distance + 1e-10)**alpha) + 1.0
    pred = A * (inside ** gamma)

    np.fill_diagonal(pred, 0.0)

    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    return (pred / row_sums) * origin_mass[:,None]

def optimize_bms_plausible_model(
    distance: np.ndarray,
    home_population: np.ndarray,
    work_population: np.ndarray,
    flows: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize the BMS Plausible model."""
    initial_params = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    bounds = [
        (0, None),     # A > 0
        (None, None),  # B
        (None, None),  # C
        (None, None),  # D
        (0, None),     # alpha >= 0
        (0, None),     # gamma >= 0
    ]

    result = optimize.minimize(
        bms_plausible_model_log_likelihood,
        initial_params,
        args=(distance, home_population, work_population, flows),
        bounds=bounds,
        method='L-BFGS-B'
    )

    optimized_params = result.x

    predicted_flows = predict_flows_bms_plausible(
        distance, home_population, work_population, *optimized_params
    )

    return optimized_params, predicted_flows

# ======================
# Utility-based Model
# ======================

def compute_transition_weight(distance: np.ndarray, threshold: float, k: float = 1.0) -> np.ndarray:
    """Compute logistic transition weight based on distance threshold."""
    return 1 / (1 + np.exp(-k * (distance - threshold)))

def compute_utilities(
    distance: np.ndarray,
    eci: np.ndarray,
    informality: np.ndarray,
    beta_distance: float,
    beta_eci: float,
    beta_informality: float,
    threshold: Optional[float] = None,
    k: float = 1.0,
    transition: bool = False
) -> np.ndarray:
    """Compute utility matrix for the utility-based model."""
    if transition and threshold is not None:
        tw = compute_transition_weight(distance, threshold, k)
        return (beta_distance * distance +
                tw * (beta_eci * eci) +
                tw * (beta_informality * informality))
    else:
        return (beta_distance * distance +
                beta_eci * eci +
                beta_informality * informality)

def predict_flows_utility(
    distance: np.ndarray,
    eci: np.ndarray,
    informality: np.ndarray,
    home_population: np.ndarray,
    beta_distance: float,
    beta_eci: float,
    beta_informality: float,
    threshold: Optional[float] = None,
    k: float = 1.0,
    transition: bool = False
) -> np.ndarray:
    """Predict flows using the utility-based model."""
    U = compute_utilities(
        distance, eci, informality,
        beta_distance, beta_eci, beta_informality,
        threshold, k, transition
    )

    maxU = U.max(axis=1, keepdims=True)
    expU = np.exp(U - maxU)
    P = expU / expU.sum(axis=1, keepdims=True)

    np.fill_diagonal(P, 0.0)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    P = P / row_sums

    return P * home_population[:, None]

def utility_model_log_likelihood(
    params: np.ndarray,
    distance: np.ndarray,
    eci: np.ndarray,
    flows: np.ndarray,
    informality: np.ndarray,
    home_population: np.ndarray,
    transition: bool = True
) -> float:
    """Negative log-likelihood for the utility model, excluding self-flows."""
    bd, be, bf, thr, kk = params
    pred = predict_flows_utility(
        distance, eci, informality, home_population,
        bd, be, bf, thr, kk, transition
    )
    np.fill_diagonal(pred, 0.0)
    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    pred = (pred / row_sums) * home_population[:,None]

    mask = ~np.eye(pred.shape[0], dtype=bool)
    f = flows[mask];  p = pred[mask]
    return np.sum(p) - np.sum(f * np.log(p + 1e-10))

def optimize_utility_model(
    distance: np.ndarray,
    eci: np.ndarray,
    flows: np.ndarray,
    informality: np.ndarray,
    home_population: np.ndarray,
    transition: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize the utility-based model."""
    initial = [0.0, 0.0, 0.0, 0.0, 1.0]
    bounds = [(None, None), (None, None), (None, None), (0, None), (0, None)]
    res = optimize.minimize(
        utility_model_log_likelihood,
        initial,
        args=(distance, eci, flows, informality, home_population, transition),
        bounds=bounds,
        method='L-BFGS-B'
    )
    params_opt = res.x
    pred = predict_flows_utility(
        distance, eci, informality, home_population,
        *params_opt, transition
    )
    return params_opt, pred

def compute_accessibility_surplus(
    distance: np.ndarray,
    eci: np.ndarray,
    informality: np.ndarray,
    home_population: np.ndarray,
    beta_distance: float,
    beta_eci: float,
    beta_informality: float,
    threshold: Optional[float] = None,
    k: float = 1.0,
    transition: bool = False
) -> np.ndarray:
    """Compute accessibility surplus for the utility-based model."""
    home_population = home_population[:, np.newaxis]

    utilities = compute_utilities(
        distance, eci, informality, beta_distance, beta_eci,
        beta_informality, threshold, k, transition
    )

    accessibility = np.log(np.exp(utilities).sum(axis=1))

    return accessibility
