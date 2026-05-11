"""
Boruta Feature Selection

Implements Boruta algorithm for feature selection per implementation guide.

Per guide:
- Based on random forest importance
- Reduces features from 100+ candidates to 30-40 selected features
- Improves accuracy and reduces overfitting
- Identifies all relevant features while removing noise

Expected feature composition after selection:
- 5-10 order book features (multiple depths)
- 3-5 volume metrics
- 2-3 microstructure (Roll, VPIN)
- 8-12 price indicators
- 4-6 cross-asset (BTC/ETH)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from scipy.stats import binomtest
from loguru import logger


class BorutaSelector:
    """
    Boruta feature selection algorithm.

    Per implementation guide:
    - Uses random forest for importance calculation
    - Creates shadow features (permuted copies)
    - Iteratively tests feature importance vs shadows
    - Typically reduces 137+ features to 52-84 without sacrificing accuracy

    Attributes:
        n_estimators: Number of trees in random forest (default: 100)
        max_iter: Maximum iterations (default: 100)
        perc: Percentile for shadow importance threshold (default: 100)
        alpha: Significance level for hypothesis tests (default: 0.05)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 100,
        perc: int = 100,
        alpha: float = 0.05,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Boruta selector.

        Args:
            n_estimators: Random forest trees (default: 100 per guide)
            max_iter: Maximum iterations (default: 100 per guide)
            perc: Percentile for importance threshold (default: 100 per guide)
            alpha: Significance level (default: 0.05 per guide)
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state

        # Results
        self.support_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None
        self.importance_history_: List[np.ndarray] = []
        self.selected_features_: Optional[List[str]] = None
        self.rejected_features_: Optional[List[str]] = None
        self.tentative_features_: Optional[List[str]] = None

        logger.info(
            f"BorutaSelector initialized",
            extra={
                "n_estimators": n_estimators,
                "max_iter": max_iter,
                "perc": perc,
                "alpha": alpha
            }
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> "BorutaSelector":
        """
        Run Boruta feature selection.

        Algorithm:
        1. Create shadow features (permuted copies)
        2. Train random forest on augmented dataset
        3. Compute feature importance scores
        4. Compare each feature to max shadow importance
        5. Mark features as confirmed, tentative, or rejected
        6. Repeat until convergence or max_iter

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names (optional, uses DataFrame columns)

        Returns:
            Self
        """
        logger.info(
            f"Starting Boruta feature selection",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1]
            }
        )

        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        else:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        self.feature_names_ = feature_names
        n_features = X.shape[1]

        # Initialize random state
        rng = check_random_state(self.random_state)

        # Initialize feature status
        # 0 = tentative, 1 = confirmed, -1 = rejected
        feature_status = np.zeros(n_features, dtype=int)
        hit_history = np.zeros(n_features, dtype=int)

        # Track importance history
        self.importance_history_ = []

        # Main Boruta loop
        for iteration in range(self.max_iter):
            # Create shadow features (permuted copies)
            X_shadow = self._create_shadow_features(X, rng)

            # Combine original and shadow features
            X_boruta = np.hstack([X, X_shadow])

            # Train random forest
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,  # Limit depth to prevent overfitting
                random_state=rng.randint(0, 10000)
            )

            rf.fit(X_boruta, y)

            # Get feature importances
            importances = rf.feature_importances_

            # Split into real and shadow importances
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # Store importance history
            self.importance_history_.append(real_importances.copy())

            # Calculate threshold (percentile of shadow importances)
            threshold = np.percentile(shadow_importances, self.perc)

            # Test each tentative feature against threshold
            for i in range(n_features):
                if feature_status[i] == 0:  # Tentative
                    if real_importances[i] > threshold:
                        # Feature beats shadow threshold
                        hit_history[i] += 1
                    else:
                        # Feature loses to shadow threshold
                        hit_history[i] -= 1

            # Perform binomial test for tentative features
            for i in range(n_features):
                if feature_status[i] == 0:  # Still tentative
                    # Test if hit rate is significantly different from 0.5
                    n_trials = iteration + 1
                    n_successes = max(0, hit_history[i])

                    # Binomial test: H0 = feature no better than shadow
                    p_value = binomtest(
                        n_successes,
                        n_trials,
                        p=0.5,
                        alternative='greater'
                    ).pvalue

                    if p_value < self.alpha:
                        # Reject H0: feature is significantly better
                        feature_status[i] = 1  # Confirmed
                        logger.debug(f"Confirmed: {feature_names[i]}")

                    elif p_value > (1 - self.alpha):
                        # Feature significantly worse than shadow
                        feature_status[i] = -1  # Rejected
                        logger.debug(f"Rejected: {feature_names[i]}")

            # Check convergence (all features decided)
            n_tentative = np.sum(feature_status == 0)

            logger.debug(
                f"Iteration {iteration + 1}/{self.max_iter}",
                extra={
                    "confirmed": np.sum(feature_status == 1),
                    "rejected": np.sum(feature_status == -1),
                    "tentative": n_tentative,
                    "threshold": threshold
                }
            )

            if n_tentative == 0:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

        # Mark remaining tentative features as rejected (conservative)
        feature_status[feature_status == 0] = -1

        # Store results
        self.support_ = feature_status == 1
        self.ranking_ = np.where(self.support_, 1, 2)  # 1 = selected, 2 = rejected

        # Store feature lists
        self.selected_features_ = [
            feature_names[i] for i in range(n_features)
            if feature_status[i] == 1
        ]

        self.rejected_features_ = [
            feature_names[i] for i in range(n_features)
            if feature_status[i] == -1
        ]

        self.tentative_features_ = [
            feature_names[i] for i in range(n_features)
            if feature_status[i] == 0
        ]

        logger.info(
            f"Boruta selection complete",
            extra={
                "selected": len(self.selected_features_),
                "rejected": len(self.rejected_features_),
                "tentative": len(self.tentative_features_),
                "reduction": f"{n_features} → {len(self.selected_features_)}"
            }
        )

        return self

    def _create_shadow_features(self, X: np.ndarray, rng) -> np.ndarray:
        """
        Create shadow features by permuting each column.

        Shadow features are shuffled versions of real features,
        representing random noise baseline.

        Args:
            X: Feature matrix
            rng: Random number generator

        Returns:
            Shadow feature matrix
        """
        X_shadow = X.copy()

        # Shuffle each column independently
        for i in range(X_shadow.shape[1]):
            rng.shuffle(X_shadow[:, i])

        return X_shadow

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset to include only selected features.

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix
        """
        if self.support_ is None:
            raise ValueError("Boruta selector not fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.support_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit Boruta and transform in one step.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names

        Returns:
            Reduced feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get average feature importance across iterations.

        Returns:
            DataFrame with feature names and average importance
        """
        if not self.importance_history_:
            raise ValueError("Boruta selector not fitted yet.")

        # Calculate average importance
        avg_importance = np.mean(self.importance_history_, axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': avg_importance,
            'selected': self.support_,
            'ranking': self.ranking_
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def get_summary(self) -> Dict:
        """
        Get summary of feature selection results.

        Returns:
            Dictionary with selection statistics
        """
        if self.support_ is None:
            return {"status": "not_fitted"}

        # Calculate feature importance statistics
        importance_df = self.get_feature_importance()

        selected_importance = importance_df[importance_df['selected']]['importance']
        rejected_importance = importance_df[~importance_df['selected']]['importance']

        return {
            "total_features": len(self.feature_names_),
            "selected_features": len(self.selected_features_),
            "rejected_features": len(self.rejected_features_),
            "reduction_pct": (1 - len(self.selected_features_) / len(self.feature_names_)) * 100,
            "selected_importance_mean": selected_importance.mean(),
            "selected_importance_std": selected_importance.std(),
            "rejected_importance_mean": rejected_importance.mean(),
            "rejected_importance_std": rejected_importance.std(),
            "iterations": len(self.importance_history_),
            "selected_features_list": self.selected_features_,
        }


if __name__ == "__main__":
    # Test Boruta selector
    from sklearn.datasets import make_classification

    print("=== Boruta Feature Selector Test ===\n")

    # Generate synthetic dataset with some irrelevant features
    # 20 features: 10 informative, 10 redundant/irrelevant
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )

    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(20)]

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}\n")

    # Initialize Boruta selector
    selector = BorutaSelector(
        n_estimators=100,
        max_iter=100,
        perc=100,
        alpha=0.05,
        random_state=42
    )

    # Run feature selection
    print("Running Boruta selection...\n")
    selector.fit(X_df, y_series)

    # Get summary
    summary = selector.get_summary()
    print(f"\n=== Selection Summary ===")
    print(f"Total features: {summary['total_features']}")
    print(f"Selected: {summary['selected_features']}")
    print(f"Rejected: {summary['rejected_features']}")
    print(f"Reduction: {summary['reduction_pct']:.1f}%")
    print(f"Iterations: {summary['iterations']}")

    print(f"\nSelected features:")
    for feat in summary['selected_features_list']:
        print(f"  ✓ {feat}")

    # Get feature importance
    print(f"\n=== Feature Importance (Top 10) ===")
    importance_df = selector.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # Transform dataset
    print(f"\n=== Transformed Dataset ===")
    X_selected = selector.transform(X_df)
    print(f"Original shape: {X_df.shape}")
    print(f"Selected shape: {X_selected.shape}")
    print(f"Selected columns: {list(X_selected.columns)}")

    # Compare performance (optional - would need actual model training)
    print(f"\n=== Performance Comparison ===")
    print(f"Selected importance (mean ± std): {summary['selected_importance_mean']:.4f} ± {summary['selected_importance_std']:.4f}")
    print(f"Rejected importance (mean ± std): {summary['rejected_importance_mean']:.4f} ± {summary['rejected_importance_std']:.4f}")
