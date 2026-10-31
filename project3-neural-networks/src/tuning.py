"""
Hyperparameter tuning with random search.

Workflow (per model, per data set):
1. holdout_split() once with a fixed seed: ~20% becomes the tuning (validation) set,
   the other ~80% is what the 5x2 CV later runs on. Share this split across all three models.
2. random_search() trains each sampled config on the 80% and scores it on the 20%.
3. Fix the winning config and run cross_validation_loop on the 80%.

Epochs are not sampled. Each trial is trained in chunks of `checkpoint_every` epochs and
scored on the validation set after every chunk, so one trial yields a whole validation
curve and the best epoch count is read off it. This relies on fit() CONTINUING from the
current weights when called again (true for LogisticRegression, LinearRegression and
NeuralNetwork). A wrapper model that pretrains an autoencoder must only pretrain on its
first fit() call.
"""

import numpy as np
import pandas as pd

from src.evaluation import classification_error
from src.losses import cross_entropy, mse


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def holdout_split(n, y_labels=None, holdout_size=0.2, seed=None):
    """
    Split row indices into (cv_idx, val_idx).

    n:            number of rows.
    y_labels:     (n,) integer class indices for a stratified split (classification),
                  or None for a plain random split (regression).
    holdout_size: fraction held out for tuning (0.2).
    Returns two sorted index arrays. Use X[cv_idx] for the 5x2 CV data and X[val_idx]
    for the tuning set.
    """
    rng = np.random.default_rng(seed)

    if y_labels is not None:
        y_labels = np.asarray(y_labels)
        held = []
        for c in np.unique(y_labels):
            idx = rng.permutation(np.where(y_labels == c)[0])  # shuffle within the class
            held.append(idx[: int(round(len(idx) * holdout_size))])
        val_idx = np.concatenate(held)
    else:
        val_idx = rng.permutation(n)[: int(round(n * holdout_size))]

    val_idx = np.sort(val_idx)
    cv_idx = np.setdiff1d(np.arange(n), val_idx)
    return cv_idx, val_idx


# ---------------------------------------------------------------------------
# Sampling configs
# ---------------------------------------------------------------------------

def sample_value(spec, rng):
    """
    Draw one value from a search-space spec:
      [a, b, c]                 -> random choice from the list (None allowed, e.g. batch_size)
      ("log_uniform", lo, hi)   -> float, uniform in log space (use for learning rates)
      ("uniform", lo, hi)       -> float, uniform
      ("int", lo, hi)           -> integer in [lo, hi] inclusive
      anything else             -> fixed value, returned as is
    """
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    if isinstance(spec, tuple):
        kind, lo, hi = spec
        if kind == "log_uniform":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if kind == "uniform":
            return float(rng.uniform(lo, hi))
        if kind == "int":
            return int(rng.integers(lo, hi + 1))
        raise ValueError(f"Unknown spec type: {kind}")
    return spec


def sample_config(space, rng, constraint=None, max_tries=1000):
    """
    Sample one config dict from `space`. If `constraint` (config -> bool) is given,
    resample until it returns True, e.g. lambda c: c["hidden2"] <= c["hidden1"].
    """
    for _ in range(max_tries):
        config = {name: sample_value(spec, rng) for name, spec in space.items()}
        if constraint is None or constraint(config):
            return config
    raise RuntimeError("Could not sample a config that satisfies the constraint.")


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def _run_trial(model_factory, config, X_train, y_train, X_val, y_val, y_labels_val,
               mode, n_seeds, max_epochs, checkpoint_every, seed_base):
    """
    Train one config with n_seeds different initializations, scoring on the validation
    set after every checkpoint. Returns (loss_curve, error_curve), each averaged over
    seeds; entries are NaN once a run has diverged.
    """
    n_checkpoints = max_epochs // checkpoint_every
    loss = np.full((n_seeds, n_checkpoints), np.nan)
    err = np.full((n_seeds, n_checkpoints), np.nan)
    batch_size = config.get("batch_size")

    with np.errstate(all="ignore"):  # diverging runs overflow; we detect that via isfinite
        for s in range(n_seeds):
            model = model_factory(config, seed_base + s)
            for j in range(n_checkpoints):
                model.fit(X_train, y_train, epochs=checkpoint_every, batch_size=batch_size)

                if mode == "classification":
                    value = cross_entropy(y_val, model.predict_proba(X_val))
                    error = classification_error(y_labels_val, model.predict(X_val))
                else:
                    value = mse(y_val, model.predict(X_val))
                    error = np.nan

                if not np.isfinite(value):
                    break  # diverged: leave the rest of this seed's curve as NaN
                loss[s, j] = value
                err[s, j] = error

        return loss.mean(axis=0), err.mean(axis=0)  # NaN if ANY seed diverged at that checkpoint


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def random_search(model_factory, space, X_train, y_train, X_val, y_val, y_labels_val=None,
                  mode="classification", n_trials=30, n_seeds=2, max_epochs=300,
                  checkpoint_every=25, constraint=None, seed=None, log_path=None, verbose=True):
    """
    Random search over `space`, scored on the validation set.

    Parameters:
    model_factory (callable): (config, seed) -> fresh untrained model. It builds the model
        from the sampled config, e.g.
            lambda c, s: NeuralNetwork([d, c["hidden1"], c["hidden2"], K], "classification",
                                       "tanh", c["learning_rate"], seed=s)
    space (dict): {name: spec}, see sample_value. "batch_size" (if present) is passed to fit;
        every other key is only for model_factory.
    X_train, y_train: the ~80% used to train every trial.
    X_val, y_val, y_labels_val: the ~20% tuning set (y_labels_val = integer labels, classification only).
    mode (str): 'classification' (scored by validation cross-entropy) or 'regression' (validation MSE).
    n_trials (int): number of configs to sample.
    n_seeds (int): initializations averaged per config (reduces noise on small data sets).
    max_epochs, checkpoint_every (int): each trial trains up to max_epochs, scored every
        checkpoint_every epochs; the best checkpoint gives the epoch count.
    constraint (callable): optional config -> bool to reject invalid combinations.
    seed (int): seeds the sampling and the model initializations, so the search is reproducible.
    log_path (str): optional CSV path; every trial is saved there.

    Returns:
    (best_config, results)
      best_config: dict of the winning hyperparameters plus "epochs" (best checkpoint).
      results: DataFrame, one row per trial sorted best-first, with the config, epochs,
               val_loss, val_error (classification), trial number, diverged flag and the averaged "curve"
               (validation loss at each checkpoint, handy for plotting).
    """
    if max_epochs < checkpoint_every:
        raise ValueError("max_epochs must be at least checkpoint_every.")

    rng = np.random.default_rng(seed)  # drives config sampling only
    checkpoints = np.arange(1, max_epochs // checkpoint_every + 1) * checkpoint_every
    rows, configs = [], []

    for trial in range(n_trials):
        config = sample_config(space, rng, constraint)
        configs.append(config)  # keep the original Python values (a DataFrame row would turn ints into floats)
        loss_curve, err_curve = _run_trial(
            model_factory, config, X_train, y_train, X_val, y_val, y_labels_val,
            mode, n_seeds, max_epochs, checkpoint_every,
            seed_base=(0 if seed is None else seed) * 100003 + trial * n_seeds)

        valid = np.isfinite(loss_curve)
        if valid.any():
            best = int(np.argmin(np.where(valid, loss_curve, np.inf)))
            row = {**config, "epochs": int(checkpoints[best]), "val_loss": float(loss_curve[best]),
                   "val_error": float(err_curve[best]) if mode == "classification" else np.nan}
        else:
            row = {**config, "epochs": np.nan, "val_loss": np.inf, "val_error": np.nan}
        row["trial"] = trial
        row["diverged"] = bool(not valid.all())
        row["curve"] = loss_curve.tolist()
        rows.append(row)

        if verbose:
            tag = " (diverged)" if row["diverged"] else ""
            print(f"trial {trial + 1:3d}/{n_trials}  val_loss={row['val_loss']:.4f}  "
                  f"epochs={row['epochs']}  {config}{tag}")

    results = pd.DataFrame(rows).sort_values("val_loss", kind="stable").reset_index(drop=True)

    if log_path is not None:
        results.drop(columns=["curve"]).to_csv(log_path, index=False)

    top = results.iloc[0]
    if not np.isfinite(top["val_loss"]):
        raise RuntimeError("Every trial diverged. Lower the learning-rate range and try again.")
    best_config = dict(configs[int(top["trial"])])  # original values, exact types
    best_config["epochs"] = int(top["epochs"])
    return best_config, results