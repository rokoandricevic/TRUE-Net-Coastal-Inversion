"""TRUE-Net Unified Configuration (Chl-a).

Centralized parameters for the TRUE-Net architecture, including Prior-Informed
bio-optical regulation, circular temporal encoding, and spectral trust diagnostics.
"""

import numpy as np

# 1. UNIVARIATE BIO-OPTICAL PRIOR COEFFICIENTS (58x20 Polygon Calibration)
PRIOR_CONFIG = {
    "June21":     {"date": "20210626", "doy": 177, "a": 0.070000, "b": 3.000000, "tau_a": 0.140000},
    "November21": {"date": "20211029", "doy": 302, "a": 1.500000, "b": -0.70714, "tau_a": 0.049000},
    "March22":    {"date": "20220303", "doy": 62,  "a": 0.170000, "b": 1.100000, "tau_a": 0.045000},
    "April22":    {"date": "20220512", "doy": 132, "a": 0.011849, "b": 4.262590, "tau_a": 0.067000}
}

# 2. CIRCULAR TEMPORAL ENCODING (t_mod)
def get_temporal_modulation(doy: int) -> tuple[float, float]:
    """Calculates circular temporal encoding (t_sin, t_cos)."""
    fractional_year = (doy - 1) / 365.25
    return float(np.sin(2.0 * np.pi * fractional_year)), float(np.cos(2.0 * np.pi * fractional_year))

# 3. SPECTRAL APPLICABILITY & ALEATORY UNCERTAINTY
A_MIN = 0.1  # Protective baseline floor

def calculate_spectral_trust(z_score: np.ndarray) -> np.ndarray:
    """Computes the Spectral Applicability Index (A_spec)."""
    return A_MIN + (1.0 - A_MIN) * np.exp(-(z_score**2) / 2.0)

def calculate_aleatory_proxy(a_spec: np.ndarray, tau_a: float) -> np.ndarray:
    """Translates A_spec into aleatory uncertainty (sigma_spec)."""
    return tau_a * np.sqrt((1.0 / np.clip(a_spec, A_MIN, 1.0)))