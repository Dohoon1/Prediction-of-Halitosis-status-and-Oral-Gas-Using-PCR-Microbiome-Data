"""Optional PyTorch estimators for the PCR analysis.

Classification ``predict`` returns probabilities, not thresholded labels.
Preprocessing, resampling, and regression-target transformations belong to
the calling cross-validation pipeline and are deliberately not fitted here.
Importing this module does not require PyTorch.
"""

from __future__ import annotations

import numbers
import os
import random
from pathlib import Path

import numpy as np


def _load_torch():
    # This must be configured before CUDA is initialized for deterministic GEMM.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "NeuralEstimator requires the optional 'torch' package. Install a "
            "PyTorch CPU/CUDA build compatible with the Python environment "
            "running this analysis, or run the conventional models only."
        ) from exc
    return torch


def _resolve_device(torch, requested):
    if not isinstance(requested, str):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid PyTorch device: {requested!r}.") from exc
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type != "cuda":
        raise ValueError("This estimator supports CPU and CUDA devices only.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use device='cpu'.")
    index = 0 if device.index is None else device.index
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise ValueError(
            f"CUDA device index {index} is unavailable; detected {count} device(s)."
        )
    return torch.device(f"cuda:{index}")


def _seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_network(torch, n_features, params):
    """Build locally defined networks only when the optional dependency is used."""
    nn = torch.nn
    if params["architecture"] == "logistic":

        class LinearNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def forward(self, x):
                return self.linear(x).squeeze(-1)

        return LinearNetwork()

    class SelfAttention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.tokeys = nn.Linear(dim, dim, bias=False)
            self.toqueries = nn.Linear(dim, dim, bias=False)
            self.tovalues = nn.Linear(dim, dim, bias=False)
            self.unifyheads = nn.Linear(dim, dim)

        def forward(self, x):
            batch, tokens, dim = x.shape
            shape = (batch, tokens, self.heads, self.head_dim)
            queries = self.toqueries(x).view(shape).transpose(1, 2)
            keys = self.tokeys(x).view(shape).transpose(1, 2)
            values = self.tovalues(x).view(shape).transpose(1, 2)
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim**0.5
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, values)
            attended = attended.transpose(1, 2).contiguous().view(batch, tokens, dim)
            return self.unifyheads(attended)

    class EncoderBlock(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.attn = SelfAttention(dim, heads)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim)
            )

        def forward(self, x):
            x = self.norm1(self.attn(x) + x)
            return self.norm2(self.ff(x) + x)

    class FeatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            dim = params["embedding_dim"]
            self.embedding_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
                    )
                    for _ in range(n_features)
                ]
            )
            self.enc = nn.ModuleList(
                [EncoderBlock(dim, params["heads"]) for _ in range(params["depth"])]
            )
            self.fc = nn.Linear(dim, 1)
            self.dropout = nn.Dropout(params["dropout"])

        def forward(self, x):
            # Independent feature embeddings encode feature identity; no
            # sequence positions or positional encoding are introduced.
            tokens = torch.stack(
                [embed(x[:, i : i + 1]) for i, embed in enumerate(self.embedding_blocks)],
                dim=1,
            )
            for encoder in self.enc:
                tokens = encoder(tokens)
            pooled = tokens.mean(dim=1)
            return self.fc(self.dropout(pooled)).squeeze(-1)

    return FeatureTransformer()


class NeuralEstimator:
    """Small tabular Transformer or linear PyTorch model.

    ``architecture='logistic'`` uses a scalar linear output. Classification
    trains with BCEWithLogitsLoss and exposes sigmoid probabilities; regression
    trains with MSELoss without a target transformation. ``fit`` reseeds Python,
    NumPy, and PyTorch and creates a new network for every call.
    """

    _PARAMETERS = (
        "task", "architecture", "embedding_dim", "heads", "depth", "lr",
        "epochs", "batch_size", "dropout", "seed", "device", "class_weight",
    )

    def __init__(
        self,
        task="classification",
        architecture="transformer",
        embedding_dim=16,
        heads=2,
        depth=2,
        lr=0.0001,
        epochs=100,
        batch_size=8,
        dropout=0.1,
        seed=42,
        device="auto",
        class_weight=None,
    ):
        self.task = task
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.device = device
        self.class_weight = class_weight

    def get_params(self, deep=True):
        """Return constructor parameters for parameter grids and cloning helpers."""
        return {name: getattr(self, name) for name in self._PARAMETERS}

    def set_params(self, **params):
        unknown = sorted(set(params) - set(self._PARAMETERS))
        if unknown:
            raise ValueError(f"Unknown NeuralEstimator parameter(s): {unknown}.")
        for name, value in params.items():
            setattr(self, name, value)
        self._clear_fitted_state()
        return self

    def _clear_fitted_state(self):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _validated_params(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if self.architecture not in {"transformer", "logistic"}:
            raise ValueError("architecture must be 'transformer' or 'logistic'.")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")
        if self.task != "classification" and self.class_weight is not None:
            raise ValueError("class_weight applies only to classification.")
        params = self.get_params()
        params["task"] = str(self.task)
        params["architecture"] = str(self.architecture)
        if not isinstance(self.device, str):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")
        params["device"] = str(self.device)
        for name in ("embedding_dim", "heads", "depth", "epochs", "batch_size"):
            value = params[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            params[name] = int(value)
        if params["embedding_dim"] % params["heads"]:
            raise ValueError("embedding_dim must be divisible by heads.")
        for name in ("lr", "dropout"):
            value = params[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            params[name] = float(value)
        if params["lr"] <= 0:
            raise ValueError("lr must be greater than zero.")
        if not 0 <= params["dropout"] < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, numbers.Integral)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        params["seed"] = int(self.seed)
        return params

    @staticmethod
    def _as_X(X, allow_empty=False):
        try:
            array = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a two-dimensional numeric array.") from exc
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must have shape (samples, features) with at least one feature.")
        if not allow_empty and array.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.isfinite(array).all():
            raise ValueError("X must contain finite values after numeric conversion.")
        return np.ascontiguousarray(array)

    def fit(self, X, y):
        """Train a fresh model; no preprocessing or resampling is performed."""
        self._clear_fitted_state()
        params = self._validated_params()
        X_array = self._as_X(X)
        try:
            y_array = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must be numeric.") from exc
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array[:, 0]
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must have one value per row of X.")
        if not np.isfinite(y_array).all():
            raise ValueError("y must contain finite values.")
        if self.task == "classification" and not np.array_equal(
            np.unique(y_array), np.array([0.0, 1.0])
        ):
            raise ValueError("Classification requires both binary labels 0 and 1.")
        y_array = np.ascontiguousarray(y_array, dtype=np.float32)
        if not np.isfinite(y_array).all():
            raise ValueError("y must remain finite when represented as float32.")

        torch = _load_torch()
        device = _resolve_device(torch, params["device"])
        _seed_everything(torch, params["seed"])
        network = _make_network(torch, X_array.shape[1], params).to(device)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_array), torch.from_numpy(np.ascontiguousarray(y_array))
        )
        generator = torch.Generator().manual_seed(params["seed"])
        batches = torch.utils.data.DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        if self.task == "classification":
            weight = None
            if params["class_weight"] == "balanced":
                negative, positive = np.bincount(y_array.astype(int), minlength=2)
                weight = torch.tensor(float(negative / positive), dtype=torch.float32, device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=params["lr"])
        history = []
        for epoch in range(params["epochs"]):
            network.train()
            total_loss = 0.0
            for batch_X, batch_y in batches:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(batch_X), batch_y)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}; check "
                        "predictor/target scaling and learning rate."
                    )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item()) * len(batch_y)
            history.append(total_loss / len(dataset))

        network.eval()
        self.model_ = network
        self.n_features_in_ = X_array.shape[1]
        self.device_ = str(device)
        self.loss_history_ = history
        self.fitted_params_ = params
        self.n_parameters_ = sum(p.numel() for p in network.parameters() if p.requires_grad)
        self.stopping_rule_ = "fixed_epochs_no_early_stopping"
        if self.task == "classification":
            self.classes_ = np.array([0, 1])
        return self

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("NeuralEstimator is not fitted; call fit(X, y) first.")

    def predict(self, X):
        """Return 1D class-1 probabilities or 1D regression predictions."""
        self._check_fitted()
        array = self._as_X(X, allow_empty=True)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {array.shape[1]}."
            )
        if len(array) == 0:
            return np.empty(0, dtype=float)
        torch = _load_torch()
        values = []
        self.model_.eval()
        batch_size = self.fitted_params_["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(array), batch_size):
                batch = torch.from_numpy(array[start : start + batch_size]).to(self.device_)
                output = self.model_(batch)
                if self.fitted_params_["task"] == "classification":
                    output = torch.sigmoid(output)
                values.append(output.detach().cpu().numpy().reshape(-1))
        predictions = np.concatenate(values).astype(float, copy=False)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Model produced non-finite predictions.")
        return predictions

    def predict_proba(self, X):
        """Return columns P(y=0), P(y=1) for a fitted classifier."""
        self._check_fitted()
        if self.fitted_params_["task"] != "classification":
            raise ValueError("predict_proba is available for classification only.")
        probability = self.predict(X)
        return np.column_stack((1.0 - probability, probability))

    def get_coefficients(self):
        """Return linear predictor coefficients, or None for a Transformer."""
        self._check_fitted()
        if self.fitted_params_["architecture"] != "logistic":
            return None
        return self.model_.linear.weight.detach().cpu().numpy().reshape(-1).copy()

    def save(self, path):
        """Save primitive metadata and CPU tensors, never a pickled network class."""
        self._check_fitted()
        torch = _load_torch()
        payload = {
            "format_version": 1,
            "params": dict(self.fitted_params_),
            "n_features_in": int(self.n_features_in_),
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model_.state_dict().items()
            },
            "loss_history": [float(value) for value in self.loss_history_],
        }
        torch.save(payload, Path(path))
        return Path(path)

    @classmethod
    def load(cls, path, device="auto"):
        """Restore a checkpoint written by save using restricted tensor loading."""
        torch = _load_torch()
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Checkpoint loading requires a PyTorch version supporting "
                "torch.load(..., weights_only=True). Upgrade the optional dependency."
            ) from exc
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unrecognized NeuralEstimator checkpoint format.")
        params = dict(payload["params"])
        params["device"] = device
        estimator = cls(**params)
        params = estimator._validated_params()
        n_features = payload["n_features_in"]
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
            raise ValueError("Checkpoint has an invalid feature count.")
        resolved = _resolve_device(torch, device)
        network = _make_network(torch, n_features, params)
        network.load_state_dict(payload["state_dict"], strict=True)
        network.to(resolved).eval()
        estimator.model_ = network
        estimator.n_features_in_ = n_features
        estimator.device_ = str(resolved)
        estimator.fitted_params_ = params
        estimator.loss_history_ = list(payload.get("loss_history", []))
        if params["task"] == "classification":
            estimator.classes_ = np.array([0, 1])
        return estimator
