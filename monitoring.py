"""Prometheus metrics and a lightweight online PSI drift monitor."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("pricing_requests_total", "Pricing API requests", ["route", "status"])
PREDICTION_LATENCY = Histogram("pricing_prediction_seconds", "Pricing prediction latency")
RECOMMENDATION_DELTA = Histogram(
    "pricing_recommendation_delta_pct", "Recommended change from current price"
)
FEATURE_PSI = Gauge("pricing_feature_psi", "Online PSI against training reference", ["feature"])
FEATURE_DRIFT_ALERT = Gauge(
    "pricing_feature_drift_alert", "1 when feature PSI exceeds the alert threshold", ["feature"]
)
DRIFT_THRESHOLD = 0.20
WINDOW_SIZE = 200
MIN_WINDOW = 30


class DriftMonitor:
    def __init__(self, reference: dict):
        self.reference = reference or {}
        self.windows = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def observe(self, features: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, value in features.items():
            spec = self.reference.get(name)
            if not spec:
                continue
            window = self.windows[name]
            window.append(float(value))
            if len(window) < MIN_WINDOW:
                continue
            internal_edges = spec.get("edges", [])[1:-1]
            edges = np.r_[-np.inf, internal_edges, np.inf]
            reference_counts = np.asarray(spec.get("counts", []), dtype=float)
            if len(reference_counts) != len(edges) - 1:
                continue
            reference_share = reference_counts / max(reference_counts.sum(), 1)
            observed_counts = np.histogram(np.asarray(window), bins=edges)[0].astype(float)
            observed_share = observed_counts / max(observed_counts.sum(), 1)
            epsilon = 1e-6
            psi = float(
                np.sum(
                    (observed_share - reference_share)
                    * np.log((observed_share + epsilon) / (reference_share + epsilon))
                )
            )
            FEATURE_PSI.labels(feature=name).set(psi)
            FEATURE_DRIFT_ALERT.labels(feature=name).set(float(psi >= DRIFT_THRESHOLD))
            scores[name] = psi
        return scores

