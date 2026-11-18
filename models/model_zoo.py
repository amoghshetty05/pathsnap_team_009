"""Candidate classifiers from different algorithm families.

Each model scores a (GPS observation, candidate arc) row with P(candidate is the true arc).
Scores are normalised within each observation's candidate set before evaluation / decoding.
Scaling is part of each pipeline so it is fitted on training folds only.
"""
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from configs.config import SEED


def make_models():
    """name -> (family, config string, estimator factory)."""
    return {
        "LogisticRegression": (
            "Linear / probabilistic",
            "StandardScaler + L2, C=1.0, lbfgs, max_iter=2000",
            lambda: make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ),
        "GaussianNB": (
            "Linear / probabilistic",
            "QuantileTransformer(normal) + GaussianNB, var_smoothing=1e-9",
            lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                  GaussianNB()),
        ),
        "KNN": (
            "Instance-based",
            "StandardScaler + k=25, distance-weighted, euclidean, brute-force search",
            lambda: make_pipeline(StandardScaler(),
                                  KNeighborsClassifier(n_neighbors=25, weights="distance",
                                                       algorithm="brute", n_jobs=2)),
        ),
        "RandomForest": (
            "Ensemble (bagging)",
            "200 trees, max_depth=None, min_samples_leaf=5, max_features=sqrt",
            lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                           n_jobs=2, random_state=SEED),
        ),
        "LightGBM": (
            "Boosting",
            "300 rounds, lr=0.05, num_leaves=31, min_child_samples=20, subsample=1.0",
            lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                       random_state=SEED, n_jobs=2, verbose=-1),
        ),
        "MLP": (
            "Neural network",
            "StandardScaler + MLP(64,32), relu, adam lr=1e-3, alpha=1e-4, early_stopping",
            lambda: make_pipeline(StandardScaler(),
                                  MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                learning_rate_init=1e-3, batch_size=512,
                                                max_iter=60, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=6,
                                                random_state=SEED)),
        ),
    }
