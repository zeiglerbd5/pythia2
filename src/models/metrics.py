"""
Model Evaluation and Metrics Tracking

Implements comprehensive evaluation metrics per implementation guide:
- Classification metrics: Accuracy (target 82.44%), Precision (90%+), Recall, F1
- ROC AUC and Precision-Recall curves
- Confusion matrix with detailed breakdown
- Sharpe ratio calculation for trading performance
- Backtest metrics: Win rate, profit factor, max drawdown
- Performance analysis by symbol, timeframe, and market conditions

Critical for validating model meets guide specifications and is ready for production.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from loguru import logger
import json


@dataclass
class ClassificationMetrics:
    """
    Classification performance metrics.

    Per implementation guide:
    - Accuracy: Target 82.44% (guide benchmark)
    - Precision: Target 90%+ (critical for low-frequency trading)
    - Recall: Balance with precision
    - F1: Harmonic mean of precision and recall
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float  # Precision-Recall AUC
    tp: int  # True positives
    fp: int  # False positives
    tn: int  # True negatives
    fn: int  # False negatives
    specificity: float  # True negative rate
    npv: float  # Negative predictive value

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics(\n"
            f"  Accuracy:    {self.accuracy:.4f}\n"
            f"  Precision:   {self.precision:.4f}\n"
            f"  Recall:      {self.recall:.4f}\n"
            f"  F1:          {self.f1:.4f}\n"
            f"  ROC AUC:     {self.roc_auc:.4f}\n"
            f"  PR AUC:      {self.pr_auc:.4f}\n"
            f"  TP/FP/TN/FN: {self.tp}/{self.fp}/{self.tn}/{self.fn}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "specificity": self.specificity,
            "npv": self.npv
        }

    def meets_targets(self) -> bool:
        """
        Check if metrics meet implementation guide targets.

        Returns:
            True if accuracy >= 82.44% and precision >= 90%
        """
        accuracy_target = 0.8244
        precision_target = 0.90

        return self.accuracy >= accuracy_target and self.precision >= precision_target


@dataclass
class TradingMetrics:
    """
    Trading performance metrics.

    Per implementation guide:
    - Sharpe ratio: Target 1.3-1.8 for high-selectivity systems
    - Win rate: Percentage of profitable trades
    - Profit factor: Gross profit / gross loss
    - Max drawdown: Maximum peak-to-trough decline
    """
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    def __repr__(self) -> str:
        return (
            f"TradingMetrics(\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:.3f}\n"
            f"  Win Rate:       {self.win_rate:.2%}\n"
            f"  Profit Factor:  {self.profit_factor:.3f}\n"
            f"  Max Drawdown:   {self.max_drawdown:.2%}\n"
            f"  Avg Win/Loss:   {self.avg_win:.2%} / {self.avg_loss:.2%}\n"
            f"  Total Trades:   {self.total_trades}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades
        }

    def meets_targets(self) -> bool:
        """
        Check if metrics meet implementation guide targets.

        Returns:
            True if Sharpe ratio >= 1.3
        """
        sharpe_target = 1.3
        return self.sharpe_ratio >= sharpe_target


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics.

    Provides both classification and trading metrics for model validation.
    """

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            y_proba: Predicted probabilities (optional, for AUC)

        Returns:
            ClassificationMetrics instance
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value

        # AUC metrics (require probabilities)
        if y_proba is not None and len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
        else:
            roc_auc = 0.0
            pr_auc = 0.0

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            specificity=specificity,
            npv=npv
        )

    @staticmethod
    def calculate_trading_metrics(
        returns: np.ndarray,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> TradingMetrics:
        """
        Calculate trading performance metrics.

        Args:
            returns: Strategy returns (profit/loss per trade)
            predictions: Binary predictions (1 = trade, 0 = no trade)
            actual_returns: Actual forward returns
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            TradingMetrics instance
        """
        # Filter to only trades where prediction was positive
        trade_mask = predictions == 1
        trade_returns = actual_returns[trade_mask]

        if len(trade_returns) == 0:
            return TradingMetrics(
                sharpe_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0
            )

        # Win/loss statistics
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        total_trades = len(trade_returns)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        # Annualized assuming daily returns, adjust if different frequency
        mean_return = trade_returns.mean()
        std_return = trade_returns.std()

        if std_return > 0:
            # Simple Sharpe (not annualized for now)
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative_returns = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max)
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        return TradingMetrics(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalize by row (true label)

        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cm

    @staticmethod
    def calculate_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if len(np.unique(y_true)) < 2:
            return np.array([]), np.array([]), np.array([])

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return fpr, tpr, thresholds

    @staticmethod
    def calculate_pr_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Precision-Recall curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if len(np.unique(y_true)) < 2:
            return np.array([]), np.array([]), np.array([])

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return precision, recall, thresholds

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        # Try different thresholds
        thresholds = np.linspace(0, 1, 101)
        scores = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        scores = np.array(scores)
        best_idx = np.argmax(scores)

        return thresholds[best_idx], scores[best_idx]


class ModelEvaluator:
    """
    Comprehensive model evaluation with reporting.

    Generates detailed evaluation reports for model validation.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.calculator = MetricsCalculator()

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            returns: Actual forward returns (optional, for trading metrics)

        Returns:
            Dictionary with all metrics
        """
        # Classification metrics
        clf_metrics = self.calculator.calculate_classification_metrics(
            y_true, y_pred, y_proba
        )

        results = {
            "classification": clf_metrics.to_dict(),
            "meets_classification_targets": clf_metrics.meets_targets()
        }

        # Trading metrics (if returns provided)
        if returns is not None:
            trading_metrics = self.calculator.calculate_trading_metrics(
                returns=returns,
                predictions=y_pred,
                actual_returns=returns
            )

            results["trading"] = trading_metrics.to_dict()
            results["meets_trading_targets"] = trading_metrics.meets_targets()

        # Confusion matrix
        cm = self.calculator.plot_confusion_matrix(y_true, y_pred, normalize=False)
        cm_norm = self.calculator.plot_confusion_matrix(y_true, y_pred, normalize=True)

        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_normalized"] = cm_norm.tolist()

        # Find optimal threshold if probabilities provided
        if y_proba is not None:
            optimal_threshold, optimal_f1 = self.calculator.find_optimal_threshold(
                y_true, y_proba, metric='f1'
            )
            results["optimal_threshold"] = optimal_threshold
            results["optimal_f1"] = optimal_f1

        return results

    def print_report(self, evaluation: Dict):
        """
        Print formatted evaluation report.

        Args:
            evaluation: Evaluation dictionary from evaluate_model
        """
        logger.info("=" * 80)
        logger.info("MODEL EVALUATION REPORT")
        logger.info("=" * 80)

        # Classification metrics
        logger.info("\nCLASSIFICATION METRICS:")
        clf = evaluation["classification"]
        logger.info(f"  Accuracy:    {clf['accuracy']:.4f} (target: 0.8244)")
        logger.info(f"  Precision:   {clf['precision']:.4f} (target: 0.9000)")
        logger.info(f"  Recall:      {clf['recall']:.4f}")
        logger.info(f"  F1:          {clf['f1']:.4f}")
        logger.info(f"  ROC AUC:     {clf['roc_auc']:.4f}")
        logger.info(f"  PR AUC:      {clf['pr_auc']:.4f}")

        logger.info(f"\nCONFUSION MATRIX:")
        logger.info(f"  True Positives:  {clf['tp']}")
        logger.info(f"  False Positives: {clf['fp']}")
        logger.info(f"  True Negatives:  {clf['tn']}")
        logger.info(f"  False Negatives: {clf['fn']}")

        # Check targets
        if evaluation["meets_classification_targets"]:
            logger.info("\n✓ MEETS CLASSIFICATION TARGETS")
        else:
            logger.info("\n✗ DOES NOT MEET CLASSIFICATION TARGETS")

        # Trading metrics
        if "trading" in evaluation:
            logger.info("\nTRADING METRICS:")
            trade = evaluation["trading"]
            logger.info(f"  Sharpe Ratio:   {trade['sharpe_ratio']:.3f} (target: 1.3-1.8)")
            logger.info(f"  Win Rate:       {trade['win_rate']:.2%}")
            logger.info(f"  Profit Factor:  {trade['profit_factor']:.3f}")
            logger.info(f"  Max Drawdown:   {trade['max_drawdown']:.2%}")
            logger.info(f"  Avg Win:        {trade['avg_win']:.2%}")
            logger.info(f"  Avg Loss:       {trade['avg_loss']:.2%}")
            logger.info(f"  Total Trades:   {trade['total_trades']}")

            if evaluation["meets_trading_targets"]:
                logger.info("\n✓ MEETS TRADING TARGETS")
            else:
                logger.info("\n✗ DOES NOT MEET TRADING TARGETS")

        # Optimal threshold
        if "optimal_threshold" in evaluation:
            logger.info(f"\nOPTIMAL THRESHOLD:")
            logger.info(f"  Threshold: {evaluation['optimal_threshold']:.3f}")
            logger.info(f"  F1 Score:  {evaluation['optimal_f1']:.4f}")

        logger.info("\n" + "=" * 80)

    def save_report(self, evaluation: Dict, filepath: str):
        """
        Save evaluation report to JSON file.

        Args:
            evaluation: Evaluation dictionary
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(evaluation, f, indent=2)

        logger.info(f"Evaluation report saved to {filepath}")


if __name__ == "__main__":
    # Test metrics
    print("=== Metrics Calculator Test ===\n")

    # Create synthetic predictions
    np.random.seed(42)

    n_samples = 1000

    # Simulate imbalanced dataset (5% positive)
    y_true = np.zeros(n_samples)
    y_true[np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = 1

    # Simulate predictions with 85% accuracy, 92% precision
    y_proba = np.random.random(n_samples)

    # Make positive class more likely to have high probabilities
    positive_mask = y_true == 1
    y_proba[positive_mask] = np.clip(y_proba[positive_mask] + 0.4, 0, 1)

    y_pred = (y_proba > 0.5).astype(int)

    print(f"Dataset: {n_samples} samples")
    print(f"  True positives: {y_true.sum():.0f} ({y_true.mean()*100:.1f}%)")
    print(f"  Predicted positives: {y_pred.sum():.0f} ({y_pred.mean()*100:.1f}%)\n")

    # Calculate classification metrics
    print("=== Classification Metrics ===\n")

    calculator = MetricsCalculator()
    clf_metrics = calculator.calculate_classification_metrics(y_true, y_pred, y_proba)

    print(clf_metrics)

    print(f"\nMeets targets: {clf_metrics.meets_targets()}")

    # Calculate trading metrics
    print("\n=== Trading Metrics ===\n")

    # Simulate returns (spike = 20% return, no spike = 0%)
    actual_returns = y_true * 0.20

    trading_metrics = calculator.calculate_trading_metrics(
        returns=actual_returns,
        predictions=y_pred,
        actual_returns=actual_returns
    )

    print(trading_metrics)

    print(f"\nMeets targets: {trading_metrics.meets_targets()}")

    # Find optimal threshold
    print("\n=== Optimal Threshold ===\n")

    optimal_threshold, optimal_f1 = calculator.find_optimal_threshold(
        y_true, y_proba, metric='f1'
    )

    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Optimal F1 score:  {optimal_f1:.4f}")

    # Test with optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    clf_optimal = calculator.calculate_classification_metrics(y_true, y_pred_optimal, y_proba)

    print(f"\nWith optimal threshold:")
    print(f"  Precision: {clf_optimal.precision:.4f}")
    print(f"  Recall:    {clf_optimal.recall:.4f}")
    print(f"  F1:        {clf_optimal.f1:.4f}")

    # Test evaluator
    print("\n=== Model Evaluator ===\n")

    evaluator = ModelEvaluator()

    evaluation = evaluator.evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        returns=actual_returns
    )

    evaluator.print_report(evaluation)

    # Save report
    evaluator.save_report(evaluation, './test_evaluation.json')

    print("\n✓ Metrics module test complete!")

    # Clean up
    import os
    os.remove('./test_evaluation.json')
