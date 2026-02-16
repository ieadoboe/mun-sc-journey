# Linear Regression

## 1. Ordinary Least Squares (OLS)

**Objective:** Find weights **w** that minimize the sum of squared residuals.

$$\min_{\mathbf{w}} \, \|\mathbf{y} - \mathbf{Xw}\|_2^2 = \min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \mathbf{w})^2$$

**Closed-form solution:**
$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Key assumptions (Gauss-Markov):**
- Linearity in parameters
- Errors have zero mean: $\mathbb{E}[\epsilon_i] = 0$
- Homoscedasticity: $\text{Var}(\epsilon_i) = \sigma^2$
- No autocorrelation: $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$
- Exogeneity: $\mathbb{E}[\epsilon_i \mid \mathbf{X}] = 0$

**Critical vulnerabilities:**
- **Multicollinearity:** When features are highly correlated, $\mathbf{X}^\top \mathbf{X}$ becomes ill-conditioned → unstable, high-variance estimates
- **Overfitting:** With $p \approx n$ or $p > n$, OLS perfectly fits noise
- **No solution when $p > n$:** Matrix $\mathbf{X}^\top \mathbf{X}$ is singular

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5)
y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# OLS
ols = LinearRegression()
ols.fit(X_train, y_train)
print(f"OLS Train R²: {ols.score(X_train, y_train):.4f}")
print(f"OLS Test R²: {ols.score(X_test, y_test):.4f}")
print(f"Coefficients: {ols.coef_}")
```

## 2. Ridge Regression (L2 Regularization)

**Objective:** Add a penalty proportional to the squared magnitude of weights.

$$\min_{\mathbf{w}} \, \|\mathbf{y} - \mathbf{Xw}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

**Closed-form solution:**
$$\mathbf{w}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Mechanics:**
- The term $\alpha \mathbf{I}$ ensures $\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I}$ is always invertible (positive definite)
- Shrinks weights toward zero but *never exactly to zero*
- Equivalent to MAP estimation with Gaussian prior: $\mathbf{w} \sim \mathcal{N}(0, \frac{1}{\alpha}\mathbf{I})$

**When to use:**
- Multicollinear features
- You want to retain all features but reduce their influence
- Prediction accuracy matters more than interpretability

**Tuning $\alpha$:**
- $\alpha = 0$: Reduces to OLS
- $\alpha \to \infty$: All weights shrink to zero
- Select via cross-validation

```python
from sklearn.linear_model import Ridge, RidgeCV

# Ridge with fixed alpha
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge Train R²: {ridge.score(X_train, y_train):.4f}")
print(f"Ridge Test R²: {ridge.score(X_test, y_test):.4f}")
print(f"Coefficients: {ridge.coef_}")

# Ridge with cross-validation for alpha selection
alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_:.4f}")
print(f"RidgeCV Test R²: {ridge_cv.score(X_test, y_test):.4f}")
```

## 3. Lasso Regression (L1 Regularization)

**Objective:** Penalize the absolute magnitude of weights.

$$\min_{\mathbf{w}} \, \|\mathbf{y} - \mathbf{Xw}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

where $\|\mathbf{w}\|_1 = \sum_{j=1}^p |w_j|$

**Key distinction:** No closed-form solution; requires iterative optimization (coordinate descent, proximal gradient).

**Sparsity-inducing property:**
- L1 penalty drives weights *exactly to zero* for sufficiently large $\alpha$
- Performs **automatic feature selection**
- Bayesian interpretation: Laplace prior on weights

**Why sparsity?**
- The L1 constraint creates a diamond-shaped feasible region in weight space
- Contours of the loss function intersect this region at corners (axes) → sparse solutions

**When to use:**
- High-dimensional data with many irrelevant features
- You need interpretability via feature selection
- Underlying true model is sparse

**Limitations:**
- If $p > n$, Lasso selects at most $n$ features
- Among correlated features, Lasso arbitrarily picks one (unstable selection)
- Doesn't perform well when all features are relevant but weak

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso with fixed alpha
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)
print(f"Lasso Train R²: {lasso.score(X_train, y_train):.4f}")
print(f"Lasso Test R²: {lasso.score(X_test, y_test):.4f}")
print(f"Coefficients: {lasso.coef_}")
print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")

# Lasso with cross-validation
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.alpha_:.4f}")
print(f"LassoCV Test R²: {lasso_cv.score(X_test, y_test):.4f}")
```

## 4. Elastic Net

**Objective:** Combine L1 and L2 penalties.

$$\min_{\mathbf{w}} \, \|\mathbf{y} - \mathbf{Xw}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$$

where $\rho \in [0,1]$ controls the L1/L2 mix:
- $\rho = 1$: Pure Lasso
- $\rho = 0$: Pure Ridge

**Advantages:**
- Retains sparsity from L1
- Encourages grouped selection of correlated features (from L2)
- More stable than Lasso when features are correlated

**When to use:**
- Correlated predictors where you want to select groups
- You want both regularization and feature selection

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# ElasticNet with fixed parameters
# l1_ratio is the mixing parameter (equivalent to ρ)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train, y_train)
print(f"ElasticNet Train R²: {elastic.score(X_train, y_train):.4f}")
print(f"ElasticNet Test R²: {elastic.score(X_test, y_test):.4f}")
print(f"Coefficients: {elastic.coef_}")
print(f"Non-zero coefficients: {np.sum(elastic.coef_ != 0)}")

# ElasticNet with cross-validation
elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], 
                           cv=5, max_iter=10000, random_state=42)
elastic_cv.fit(X_train, y_train)
print(f"Best alpha: {elastic_cv.alpha_:.4f}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_:.4f}")
print(f"ElasticNetCV Test R²: {elastic_cv.score(X_test, y_test):.4f}")
```

## 5. Comparative Summary

| Method | Penalty | Sparsity | Closed Form | Best For |
|--------|---------|----------|-------------|----------|
| **OLS** | None | No | Yes | Low $p$, uncorrelated features, theory testing |
| **Ridge** | $\|\|\mathbf{w}\|\|_2^2$ | No | Yes | Multicollinearity, all features relevant |
| **Lasso** | $\|\|\mathbf{w}\|\|_1$ | Yes | No | High $p$, sparse ground truth, interpretability |
| **Elastic Net** | $\rho \|\|\mathbf{w}\|\|_1 + (1-\rho)\|\|\mathbf{w}\|\|_2^2$ | Yes | No | Correlated features, grouped selection |

## 6. Bias-Variance Tradeoff

All regularization methods introduce **bias** (systematically underestimate coefficients) to reduce **variance** (sensitivity to training data fluctuations).

- **OLS:** Unbiased but high variance when $p$ is large or features are correlated
- **Ridge/Lasso:** Biased but lower variance → often better test MSE

The optimal $\alpha$ balances this tradeoff, typically chosen via k-fold cross-validation.

## 7. Geometric Intuition

Picture the constraint regions in 2D weight space:

- **Ridge:** Circle ($w_1^2 + w_2^2 \leq t$) — smooth, no corners
- **Lasso:** Diamond ($|w_1| + |w_2| \leq t$) — sharp corners along axes

OLS contours (ellipses) are more likely to intersect the diamond at a corner (where one weight = 0) than the circle → sparsity.

## 8. Critical Evaluation

**What these models assume you're not questioning:**

1. **Linearity:** The relationship $y = \mathbf{w}^\top \mathbf{x} + \epsilon$ holds. If the true function is nonlinear, regularization won't save you — you need basis expansion, kernels, or nonlinear models.

2. **Feature relevance is binary:** Lasso treats features as either useful or useless. Reality is often continuous relevance.

3. **IID samples:** Violations (time series, spatial data) break standard inference.

4. **No measurement error in $\mathbf{X}$:** Errors-in-variables bias OLS and regularized estimates.

**Alternative perspectives:**

- **Bayesian view:** Ridge/Lasso are just MAP estimates under specific priors. Full Bayesian inference captures uncertainty better.
- **Information theory:** Regularization prevents overfitting by limiting model complexity (minimum description length).
- **Statistical learning theory:** Regularization controls Rademacher complexity, bounding generalization error.

**When regularization fails:**
- Severe nonlinearity or misspecification
- Causal inference (regularization biases causal effect estimates)
- Very high noise-to-signal ratio

## 9. Practical Considerations

**Preprocessing:**
- **Standardize features:** Regularization is scale-dependent. Always use $\mathbf{x}_j \leftarrow \frac{\mathbf{x}_j - \bar{x}_j}{\text{sd}(\mathbf{x}_j)}$
- **Don't regularize intercept:** Sklearn handles this automatically (`fit_intercept=True` by default)

**Hyperparameter tuning:**
- Use cross-validation, not test set
- Ridge: $\alpha \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$ (log scale)
- Lasso: Path algorithms (LARS, coordinate descent) efficiently compute solutions across all $\alpha$

**Key sklearn parameters:**
- `alpha`: Regularization strength (higher = more regularization)
- `l1_ratio`: For ElasticNet, mix between L1 and L2 (0=Ridge, 1=Lasso)
- `max_iter`: Maximum iterations for Lasso/ElasticNet (increase if convergence warnings)
- `cv`: Number of cross-validation folds for CV variants
- `normalize=False`: Deprecated; use `StandardScaler` instead


## Conclusion

Regularization is fundamentally about **constraining hypothesis space** to prevent overfitting. Ridge shrinks continuously; Lasso selects. Elastic Net hedges. The choice depends on your prior beliefs about sparsity, feature correlation, and interpretability needs.

But remember: **regularization cannot fix fundamental model misspecification.** If your features don't contain signal or the relationship is nonlinear, no amount of tuning $\alpha$ will help. Regularization is a bias-variance tool, not a substitute for domain knowledge or exploratory analysis.