#!/usr/bin/env python3
"""
Vehicle Price Prediction – Enhanced Evaluation

- Loads processed_data.pkl and best_model.pkl
- Comprehensive evaluation with multiple metrics and visualizations
- Price range analysis and detailed error reporting
- Saves enhanced metrics and plots

Usage:
    python evaluate.py --data outputs/processed_data.pkl --model models/best_model.pkl --out outputs/
"""
from __future__ import annotations
import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
try:
    sns.set_palette("husl")
except:
    pass


def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics including price range analysis"""

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Additional metrics
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        # Fallback for older sklearn versions
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    median_ae = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    # Price range analysis
    price_ranges = {
        "Under_5L": (y_true < 500000),
        "5L_to_10L": ((y_true >= 500000) & (y_true < 1000000)),
        "10L_to_20L": ((y_true >= 1000000) & (y_true < 2000000)),
        "Above_20L": (y_true >= 2000000)
    }

    range_metrics = {}
    for range_name, mask in price_ranges.items():
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            range_r2 = r2_score(
                y_true[mask], y_pred[mask]) if np.sum(mask) > 1 else 0
            range_count = int(np.sum(mask))
            range_metrics[range_name] = {
                "count": range_count,
                "MAE": float(range_mae),
                "R2": float(range_r2),
                "percentage": float(range_count / len(y_true) * 100)
            }

    # Error percentiles
    errors = np.abs(y_true - y_pred)
    error_percentiles = {
        "p25": float(np.percentile(errors, 25)),
        "p50": float(np.percentile(errors, 50)),
        "p75": float(np.percentile(errors, 75)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95))
    }

    return {
        "overall": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
            "Median_AE": float(median_ae),
            "Max_Error": float(max_error),
            "n_samples": len(y_true)
        },
        "price_ranges": range_metrics,
        "error_percentiles": error_percentiles,
        "evaluation_timestamp": datetime.now().isoformat()
    }


def create_comprehensive_plots(y_true, y_pred, feature_names, model_name, output_dir):
    """Create comprehensive evaluation plots"""

    # Set up the plot style
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Actual vs Predicted with enhanced styling
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='steelblue',
                edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    lims = [min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, 'r--', alpha=0.8,
             linewidth=2, label='Perfect Prediction')

    # Format axes with Indian currency
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='both')

    plt.xlabel('Actual Price (₹)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
    plt.title(
        f'Actual vs Predicted Prices - {model_name}', fontsize=14, fontweight='bold')
    plt.legend()

    # Add metrics annotation
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = ₹{mae:,.0f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, verticalalignment='top')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_pred_enhanced.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Enhanced Residuals Analysis
    residuals = y_true - y_pred
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=20, color='coral')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Predicted Price (₹)', fontweight='bold')
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
    ax1.set_title('Residuals vs Predicted', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Residuals distribution
    ax2.hist(residuals, bins=50, alpha=0.7,
             color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Residuals (₹)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontweight='bold')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
    ax2.grid(True, alpha=0.3)

    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Absolute residuals vs predicted (for heteroscedasticity check)
    ax4.scatter(y_pred, np.abs(residuals), alpha=0.6, s=20, color='orange')
    ax4.set_xlabel('Predicted Price (₹)', fontweight='bold')
    ax4.set_ylabel('Absolute Residuals', fontweight='bold')
    ax4.set_title('Absolute Residuals vs Predicted', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "residuals_analysis_enhanced.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Error Analysis by Price Ranges
    plt.figure(figsize=(15, 10))

    # Define price ranges
    ranges = [0, 500000, 1000000, 2000000, np.inf]
    range_labels = ['Under ₹5L', '₹5L-10L', '₹10L-20L', 'Above ₹20L']

    errors_by_range = []
    counts_by_range = []
    r2_by_range = []

    for i in range(len(ranges)-1):
        mask = (y_true >= ranges[i]) & (y_true < ranges[i+1])
        if np.sum(mask) > 0:
            range_errors = np.abs(residuals[mask])
            errors_by_range.append(range_errors)
            counts_by_range.append(len(range_errors))
            if len(range_errors) > 1:
                r2_by_range.append(r2_score(y_true[mask], y_pred[mask]))
            else:
                r2_by_range.append(0)
        else:
            errors_by_range.append([])
            counts_by_range.append(0)
            r2_by_range.append(0)

    # Box plot of errors by price range
    plt.subplot(2, 3, 1)
    box_data = [errors for errors in errors_by_range if len(errors) > 0]
    valid_labels = [label for i, label in enumerate(
        range_labels) if len(errors_by_range[i]) > 0]
    if box_data:
        plt.boxplot(box_data, labels=valid_labels)
    plt.ylabel('Absolute Error (₹)', fontweight='bold')
    plt.title('Error Distribution by Price Range', fontweight='bold')
    plt.xticks(rotation=45)

    # R² by price range
    plt.subplot(2, 3, 2)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    bars = plt.bar(range_labels, r2_by_range,
                   color=colors[:len(range_labels)], alpha=0.7)
    plt.ylabel('R² Score', fontweight='bold')
    plt.title('Model Accuracy by Price Range', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts_by_range)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'n={count}', ha='center', va='bottom', fontsize=9)

    # Sample distribution
    plt.subplot(2, 3, 3)
    plt.pie(counts_by_range, labels=range_labels,
            autopct='%1.1f%%', colors=colors)
    plt.title('Sample Distribution by Price Range', fontweight='bold')

    # Error vs actual price
    plt.subplot(2, 3, 4)
    plt.scatter(y_true, np.abs(residuals), alpha=0.6, s=20, color='orange')
    plt.xlabel('Actual Price (₹)', fontweight='bold')
    plt.ylabel('Absolute Error (₹)', fontweight='bold')
    plt.title('Absolute Error vs Actual Price', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Percentage error vs actual price
    plt.subplot(2, 3, 5)
    percentage_errors = np.abs(residuals) / y_true * 100
    plt.scatter(y_true, percentage_errors, alpha=0.6, s=20, color='purple')
    plt.xlabel('Actual Price (₹)', fontweight='bold')
    plt.ylabel('Absolute Percentage Error (%)', fontweight='bold')
    plt.title('Percentage Error vs Actual Price', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Error percentiles
    plt.subplot(2, 3, 6)
    percentiles = [25, 50, 75, 90, 95]
    error_vals = [np.percentile(np.abs(residuals), p) for p in percentiles]
    plt.bar(range(len(percentiles)), error_vals,
            color='lightsteelblue', alpha=0.7)
    plt.xticks(range(len(percentiles)), [f'P{p}' for p in percentiles])
    plt.ylabel('Absolute Error (₹)', fontweight='bold')
    plt.title('Error Percentiles', fontweight='bold')

    # Add value labels on bars
    for i, val in enumerate(error_vals):
        plt.text(i, val + max(error_vals)*0.01, f'₹{val:,.0f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "detailed_error_analysis_enhanced.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Enhanced evaluation of vehicle price prediction model")
    ap.add_argument("--data", default="outputs/processed_data.pkl",
                    help="Path to processed data file")
    ap.add_argument("--model", default="models/best_model.pkl",
                    help="Path to trained model file")
    ap.add_argument("--out", default="outputs",
                    help="Output directory for results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("🔍 Starting Enhanced Model Evaluation...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data and model
    print("📂 Loading data and model...")
    try:
        blob = joblib.load(args.data)
        pack = joblib.load(args.model)
        model = pack["model"]
        model_name = pack.get("algo", "Unknown Model")
        print(f"✅ Successfully loaded {model_name}")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    X_test, y_test = blob["X_test"], blob["y_test"]
    feature_names = blob.get("feature_names", [])

    print(f"📊 Test set: {len(y_test):,} samples")
    print(f"🎯 Model: {model_name}")
    print(f"📈 Features: {len(feature_names)} features")

    # Make predictions
    print("🔮 Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate comprehensive metrics
    print("📈 Calculating comprehensive metrics...")
    metrics = calculate_comprehensive_metrics(y_test, y_pred)

    # Print detailed results
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)

    overall = metrics["overall"]
    print(f"🎯 Overall Performance:")
    print(f"  MAE:          ₹{overall['MAE']:>12,.0f}")
    print(f"  RMSE:         ₹{overall['RMSE']:>12,.0f}")
    print(f"  R²:           {overall['R2']:>13.3f}")
    print(f"  MAPE:         {overall['MAPE']:>12.1f}%")
    print(f"  Median AE:    ₹{overall['Median_AE']:>12,.0f}")
    print(f"  Max Error:    ₹{overall['Max_Error']:>12,.0f}")
    print(f"  Samples:      {overall['n_samples']:>13,}")

    print(f"\n💰 Performance by Price Range:")
    print(f"{'Range':<12} {'Count':<7} {'%':<6} {'MAE':<12} {'R²':<6}")
    print("-" * 50)
    for range_name, range_metrics in metrics["price_ranges"].items():
        range_display = range_name.replace("_", " ")
        print(f"{range_display:<12} {range_metrics['count']:<7} "
              f"{range_metrics['percentage']:<5.1f}% "
              f"₹{range_metrics['MAE']:<10,.0f} {range_metrics['R2']:<6.3f}")

    print(f"\n📊 Error Percentiles:")
    for percentile, value in metrics["error_percentiles"].items():
        print(f"  {percentile.upper():<4}: ₹{value:>10,.0f}")

    # Save detailed metrics
    metrics_file = os.path.join(args.out, "enhanced_test_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Detailed metrics saved to: {metrics_file}")

    # Create comprehensive plots
    print("📊 Creating comprehensive visualization plots...")
    try:
        create_comprehensive_plots(
            y_test, y_pred, feature_names, model_name, args.out)

        print("🎨 Enhanced plots saved:")
        print(f"  📈 {os.path.join(args.out, 'actual_vs_pred_enhanced.png')}")
        print(f"  📊 {os.path.join(args.out, 'residuals_analysis_enhanced.png')}")
        print(
            f"  🔍 {os.path.join(args.out, 'detailed_error_analysis_enhanced.png')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create some plots: {e}")

    # Model performance summary
    print(f"\n" + "="*60)
    print("✅ EVALUATION SUMMARY")
    print("="*60)
    print(f"🏆 Model achieves {overall['R2']:.1%} accuracy (R²)")
    print(f"💰 Average prediction error: ₹{overall['MAE']:,.0f}")
    print(f"📊 Median prediction error: ₹{overall['Median_AE']:,.0f}")

    # Performance interpretation
    if overall['R2'] >= 0.9:
        performance = "Excellent"
        emoji = "🌟"
    elif overall['R2'] >= 0.8:
        performance = "Very Good"
        emoji = "🚀"
    elif overall['R2'] >= 0.7:
        performance = "Good"
        emoji = "👍"
    elif overall['R2'] >= 0.6:
        performance = "Fair"
        emoji = "⚠️"
    else:
        performance = "Needs Improvement"
        emoji = "❌"

    print(f"{emoji} Overall Performance: {performance}")
    print("✅ Enhanced evaluation complete!")


if __name__ == "__main__":
    main()
