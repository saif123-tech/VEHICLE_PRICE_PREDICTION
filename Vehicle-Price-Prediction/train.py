#!/usr/bin/env python3
"""
Vehicle Price Prediction – Model Training

- Loads outputs/processed_data.pkl and outputs/preprocessor.joblib produced by data/dataloader.py
- Trains a few baseline & strong regressors with randomized hyperparameters
- Picks the best on validation MAE, evaluates on test set
- Saves: models/best_model.pkl, outputs/metrics.json, outputs/feature_importance.csv (if supported)
- Supports GPU/CUDA acceleration and training tracking

Usage:
    python train.py --data outputs/processed_data.pkl --out models/ --metrics_out outputs/

"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Optional: xgboost/lightgbm if installed
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Check for GPU availability
try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception:
    HAS_TORCH = False
    DEVICE = "cpu"
    print("💻 Using CPU (PyTorch not available)")


class TrainingTracker:
    """Track training progress and metrics"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.training_log = []
        self.current_model = None

    def start_model(self, model_name: str):
        """Start tracking a new model"""
        self.current_model = {
            "name": model_name,
            "start_time": time.time(),
            "status": "training"
        }
        print(f"🚀 Training {model_name}...")

    def finish_model(self, model_name: str, metrics: Dict[str, float], best_params: Dict = None):
        """Finish tracking the current model"""
        if self.current_model and self.current_model["name"] == model_name:
            self.current_model.update({
                "end_time": time.time(),
                "duration": time.time() - self.current_model["start_time"],
                "status": "completed",
                "metrics": metrics,
                "best_params": best_params
            })
            self.training_log.append(self.current_model.copy())
            duration = self.current_model["duration"]
            print(
                f"✅ {model_name} completed in {duration:.1f}s - MAE: {metrics['MAE']:.0f}")

    def save_log(self):
        """Save training log to file"""
        total_duration = time.time() - self.start_time
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "device": DEVICE,
            "models": self.training_log,
            "summary": {
                "total_models": len(self.training_log),
                "total_time": f"{total_duration:.1f}s"
            }
        }

        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        print(f"📝 Training log saved to: {log_path}")


def get_gpu_params():
    """Get GPU-optimized parameters for tree-based models"""
    if DEVICE == "cuda":
        return {
            "device": "cuda",
            "tree_method": "hist"  # Use hist with device=cuda for new XGBoost API
        }
    return {"tree_method": "hist"}


def evaluate(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE manually for compatibility
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    ap = argparse.ArgumentParser(
        description="Train vehicle price regression models")
    ap.add_argument("--data", default="outputs/processed_data.pkl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--metrics_out", default="outputs")
    ap.add_argument("--n_iter", type=int, default=25)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true",
                    help="Force GPU usage if available")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.metrics_out, exist_ok=True)

    # Initialize training tracker
    tracker = TrainingTracker(args.metrics_out)
    print(
        f"🎯 Starting training session at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    blob = joblib.load(args.data)
    X_train = blob["X_train"]
    y_train = blob["y_train"]
    X_val = blob["X_val"]
    y_val = blob["y_val"]
    X_test = blob["X_test"]
    y_test = blob["y_test"]

    print(
        f"📊 Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"📈 Features: {X_train.shape[1]}")

    models: Dict[str, Any] = {}
    gpu_params = get_gpu_params() if args.use_gpu else {"tree_method": "hist"}

    # 1) Ridge baseline
    tracker.start_model("Ridge")
    ridge = Ridge(random_state=args.random_state)
    ridge_params = {
        "alpha": np.logspace(-3, 3, 50)
    }
    ridge_cv = RandomizedSearchCV(ridge, ridge_params, n_iter=args.n_iter, cv=args.cv,
                                  scoring="neg_mean_absolute_error", n_jobs=-1, random_state=args.random_state)
    ridge_cv.fit(X_train, y_train)
    models["Ridge"] = ridge_cv.best_estimator_
    ridge_pred = ridge_cv.predict(X_val)
    ridge_metrics = evaluate(y_val, ridge_pred)
    tracker.finish_model("Ridge", ridge_metrics, ridge_cv.best_params_)

    # 2) RandomForest
    tracker.start_model("RandomForest")
    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    rf_params = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_cv = RandomizedSearchCV(rf, rf_params, n_iter=args.n_iter, cv=args.cv,
                               scoring="neg_mean_absolute_error", n_jobs=-1, random_state=args.random_state)
    rf_cv.fit(X_train, y_train)
    models["RandomForest"] = rf_cv.best_estimator_
    rf_pred = rf_cv.predict(X_val)
    rf_metrics = evaluate(y_val, rf_pred)
    tracker.finish_model("RandomForest", rf_metrics, rf_cv.best_params_)

    # 3) XGBoost (optional) with GPU support
    if HAS_XGB:
        tracker.start_model("XGBRegressor")
        xgb_base_params = {
            "random_state": args.random_state,
            "n_estimators": 800,
            "n_jobs": -1 if DEVICE == "cpu" else 1,  # GPU doesn't use n_jobs
            **gpu_params
        }
        xgb = XGBRegressor(**xgb_base_params)
        xgb_params = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1.0, 5.0],
        }
        xgb_cv = RandomizedSearchCV(xgb, xgb_params, n_iter=args.n_iter, cv=args.cv,
                                    scoring="neg_mean_absolute_error", n_jobs=-1, random_state=args.random_state)
        xgb_cv.fit(X_train, y_train)
        models["XGBRegressor"] = xgb_cv.best_estimator_
        xgb_pred = xgb_cv.predict(X_val)
        xgb_metrics = evaluate(y_val, xgb_pred)
        tracker.finish_model("XGBRegressor", xgb_metrics, xgb_cv.best_params_)

    # 4) LightGBM (optional) with GPU support
    if HAS_LGBM:
        tracker.start_model("LGBMRegressor")
        lgbm_base_params = {
            "random_state": args.random_state,
            "n_estimators": 1500,
            "device": "gpu" if DEVICE == "cuda" and args.use_gpu else "cpu",
            "verbose": -1,
            "force_col_wise": True
        }
        lgbm = LGBMRegressor(**lgbm_base_params)
        lgbm_params = {
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 8, 12, 16],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1e-3, 1e-2, 1e-1, 1.0],
        }
        lgbm_cv = RandomizedSearchCV(lgbm, lgbm_params, n_iter=args.n_iter, cv=args.cv,
                                     scoring="neg_mean_absolute_error", n_jobs=-1, random_state=args.random_state)
        lgbm_cv.fit(X_train, y_train)
        models["LGBMRegressor"] = lgbm_cv.best_estimator_
        lgbm_pred = lgbm_cv.predict(X_val)
        lgbm_metrics = evaluate(y_val, lgbm_pred)
        tracker.finish_model("LGBMRegressor", lgbm_metrics,
                             lgbm_cv.best_params_)

    # Evaluate on validation and pick best
    print("\n🏆 Model Comparison:")
    best_name = None
    best_model = None
    best_mae = float("inf")
    val_scores = {}
    for name, model in models.items():
        pred_val = model.predict(X_val)
        metrics = evaluate(y_val, pred_val)
        val_scores[name] = metrics
        print(
            f"  {name:15} - MAE: {metrics['MAE']:8.0f} | RMSE: {metrics['RMSE']:8.0f} | R²: {metrics['R2']:.3f}")
        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_name = name
            best_model = model

    # Test evaluation
    print(f"\n🎯 Best Model: {best_name}")
    pred_test = best_model.predict(X_test)
    test_metrics = evaluate(y_test, pred_test)
    print(f"📊 Test Performance:")
    print(f"  MAE:  {test_metrics['MAE']:8.0f}")
    print(f"  RMSE: {test_metrics['RMSE']:8.0f}")
    print(f"  R²:   {test_metrics['R2']:.3f}")

    # Save model
    best_path = os.path.join(args.out, "best_model.pkl")
    model_metadata = {
        "model": best_model,
        "algo": best_name,
        "feature_names": blob.get("feature_names", []),
        "training_timestamp": datetime.now().isoformat(),
        "device_used": DEVICE,
        "test_metrics": test_metrics
    }
    joblib.dump(model_metadata, best_path)

    # Save metrics
    metrics_payload = {
        "val": val_scores,
        "test": test_metrics,
        "best_model": best_name,
        "training_timestamp": datetime.now().isoformat(),
        "device_used": DEVICE,
        "n_train": blob.get("n_train"),
        "n_val": blob.get("n_val"),
        "n_test": blob.get("n_test"),
    }
    with open(os.path.join(args.metrics_out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Feature importance if available
    feat_imp_path = os.path.join(args.metrics_out, "feature_importance.csv")
    try:
        feature_names = blob.get("feature_names", [])
        importances = None
        if best_name == "RandomForest":
            importances = best_model.feature_importances_
        elif best_name == "XGBRegressor":
            importances = best_model.feature_importances_
        elif best_name == "LGBMRegressor":
            importances = best_model.feature_importances_
        if importances is not None and len(feature_names) == len(importances):
            import pandas as pd
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)
            importance_df.to_csv(feat_imp_path, index=False)
            print(f"📈 Feature importance saved to: {feat_imp_path}")
            print("🔝 Top 10 Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:25} {row['importance']:.4f}")
    except Exception as e:
        print(f"⚠️  Could not save feature importance: {e}")

    # Save training log
    tracker.save_log()

    total_time = time.time() - tracker.start_time
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"💾 Best model saved to: {best_path}")
    print(
        f"📊 Metrics saved to: {os.path.join(args.metrics_out, 'metrics.json')}")

    # GPU memory cleanup if using CUDA
    if DEVICE == "cuda" and HAS_TORCH:
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleaned up")


if __name__ == "__main__":
    main()
