# 🔧 TECHNICAL Q&A GUIDE: Deep Dive Questions

## Advanced Technical Questions You Might Face

---

## 1. INTERPRETABILITY METHODS

### Q: "Explain how SHAP values work mathematically"

**Answer**:
"SHAP values are based on Shapley values from cooperative game theory. For each feature, SHAP calculates its marginal contribution to the prediction by considering all possible feature coalitions.

Mathematically, for feature i:
```
φᵢ = Σ [|S|! × (|F| - |S| - 1)! / |F|!] × [f(S ∪ {i}) - f(S)]
```

Where:
- S: subset of features (coalition)
- F: all features
- f(S): prediction using only features in S
- φᵢ: SHAP value for feature i

**Key Properties**:
1. **Additivity**: SHAP values sum to (prediction - base_value)
2. **Local Accuracy**: Ensures fidelity to the model
3. **Consistency**: If a feature contributes more, its SHAP value is higher
4. **Missingness**: If feature isn't used, SHAP value is 0

**For TreeExplainer (what I used)**:
SHAP uses an efficient algorithm that exploits tree structure instead of computing all coalitions explicitly, making it polynomial-time instead of exponential.

**In Plain English**:
SHAP asks: 'If I were to add this feature to various combinations of other features, how much would it change the prediction on average?' That average contribution is the SHAP value."

---

### Q: "What's the difference between local and global interpretability?"

**Answer**:
**Local Interpretability** (LIME, SHAP waterfall):
- Explains a *single prediction*
- "Why did this specific loan applicant get denied?"
- Features can have different importance for different instances
- Essential for regulatory compliance (adverse action notices)
- Example: For applicant #123, debt-to-income ratio contributed +0.3 to default probability

**Global Interpretability** (PDP, Permutation Importance):
- Explains the *model's overall behavior*
- "What features does the model rely on in general?"
- Aggregates across all predictions
- Essential for model governance and auditing
- Example: Interest rate is the most important feature overall

**Connection**:
Global interpretability is the aggregation of local explanations. SHAP bridges both—SHAP values for individual predictions (local) can be averaged to get global feature importance.

**When to Use Each**:
- Local: Customer-facing explanations, debugging specific errors, regulatory compliance
- Global: Model documentation, feature selection, stakeholder buy-in, fairness auditing

**Important**: A model can be globally interpretable but locally surprising (e.g., feature matters on average, but not for this specific case), which is why I used both."

---

### Q: "Why did you use surrogate models? What do they tell you?"

**Answer**:
"Surrogate models are interpretable models (linear regression, decision trees) trained to approximate a black-box model's predictions.

**Purpose**:
1. **Approximate complexity**: If a linear model achieves R²=0.91 on LGBM predictions, it means LGBM is mostly linear
2. **Feature importance**: Coefficients in linear surrogate show global importance
3. **Decision boundaries**: Tree surrogate reveals decision rules
4. **Validation**: If surrogate performs poorly, the black-box captures complex patterns the interpretable model can't

**Two Surrogates I Used**:

1. **Linear Regression Surrogate**:
   - R² = 0.91 for LGBM
   - Interpretation: LGBM decisions are 91% explained by linear relationships
   - Top feature: Interest rate (coefficient = X)
   - Limitation: Misses non-linearities and interactions

2. **Decision Tree Surrogate**:
   - R² = 0.73
   - Interpretation: Provides rule-based approximation
   - Example rule: 'If int_rate > 15% AND dti > 30%, predict default'
   - Limitation: Simplifies complex boundaries

**Why Lower R² for Tree?**:
I limited tree depth to maintain interpretability—deeper trees could fit better but wouldn't be human-readable.

**Key Insight**:
The high linear surrogate R² suggests LGBM isn't using complex feature interactions extensively. This validates that post-hoc methods like LIME/SHAP are reliable—if the model were highly non-linear, surrogates would fail and local explanations would be less trustworthy."

---

## 2. FAIRNESS & BIAS

### Q: "What fairness metrics did you use and why?"

**Answer**:
"I used four complementary fairness metrics, each capturing a different notion of fairness:

**1. Demographic Parity (Selection Rate Parity)**:
- Definition: P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
- Meaning: Both groups should have equal positive prediction rates
- Measured: Selection rate ratio (80% rule: ratio ≥ 0.8)
- My Result: [X]% vs [Y]% → Ratio = [Z]
- When to Use: When you want equal *opportunity* regardless of group
- Criticism: Ignores base rates—if groups have different default rates naturally, enforcing parity may be unfair

**2. Equalized Odds**:
- Definition: TPR and FPR equal across groups
- Meaning: Model should be equally accurate for both groups
- Measured: |TPR_group1 - TPR_group2| and |FPR_group1 - FPR_group2|
- My Result: TPR difference = [X]%, FPR difference = [Y]%
- When to Use: When accuracy matters equally for both groups
- Criticism: May require different thresholds per group (raises legal questions)

**3. Equal Opportunity**:
- Definition: TPR equal across groups (subset of equalized odds)
- Meaning: Among true positives, both groups detected equally
- My Result: [X]
- When to Use: When false negatives are more costly than false positives (healthcare, fraud detection)

**4. Predictive Parity (PPV Parity)**:
- Definition: Precision (PPV) equal across groups
- Meaning: Positive predictions should be equally accurate across groups
- My Result: [X]
- When to Use: When false positives are costly (loan approvals, college admissions)

**Why These Four?**:
- They're industry-standard and legally recognized
- They capture different stakeholder concerns (lenders want predictive parity, applicants want equal opportunity)
- They can conflict—it's mathematically impossible to satisfy all simultaneously (Chouldechova's impossibility theorem)

**Which to Optimize?**:
This is a *business decision*, not a technical one. I presented all four so stakeholders can choose based on their priorities and regulatory requirements."

---

### Q: "How would you implement fairness constraints in the model?"

**Answer**:
"Three main approaches, each with tradeoffs:

**1. Pre-Processing: Reweighing**:
- Assign weights to training samples to balance outcomes across groups
- Tool: AIF360's Reweighing transformer
- Implementation:
  ```python
  from aif360.algorithms.preprocessing import Reweighing
  rw = Reweighing(unprivileged_groups=..., privileged_groups=...)
  dataset_transformed = rw.fit_transform(dataset)
  # Use transformed weights in LGBM
  lgbm.fit(X, y, sample_weight=weights)
  ```
- Pros: Doesn't modify model, upstream solution
- Cons: Requires retraining, may reduce performance

**2. In-Processing: Fairness-Constrained Optimization**:
- Add fairness metric as a constraint during training
- Tool: Fairlearn's GridSearch or Exponentiated Gradient
- Implementation:
  ```python
  from fairlearn.reductions import GridSearch, DemographicParity
  mitigator = GridSearch(
      lgbm,
      constraints=DemographicParity(),
      grid_size=20
  )
  mitigator.fit(X, y, sensitive_features=protected_attr)
  ```
- Pros: Principled tradeoff between accuracy and fairness
- Cons: Slower training, requires fairness metric choice

**3. Post-Processing: Threshold Optimization**:
- Train model normally, adjust thresholds per group
- Tool: Fairlearn's ThresholdOptimizer
- Implementation:
  ```python
  from fairlearn.postprocessing import ThresholdOptimizer
  postprocessor = ThresholdOptimizer(
      estimator=lgbm,
      constraints='equalized_odds',
      objective='balanced_accuracy_score'
  )
  postprocessor.fit(X, y, sensitive_features=protected_attr)
  ```
- Pros: No retraining needed, flexible
- Cons: Group-specific thresholds raise legal/ethical questions

**My Recommendation for This Project**:
Start with **post-processing threshold optimization** because:
1. It's fastest to implement (no retraining)
2. It's transparent (different thresholds are explicit)
3. It allows A/B testing before committing to retraining
4. It's reversible if stakeholders change their fairness preference

Then, if performance drop is acceptable, move to **in-processing** for a more principled long-term solution."

---

### Q: "What's the impossibility theorem in fairness?"

**Answer**:
"Chouldechova (2017) and Kleinberg et al. (2017) independently proved that you cannot simultaneously satisfy:
1. **Calibration**: P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1) (predictions are accurate across groups)
2. **Equalized Odds**: TPR and FPR equal across groups
3. **Demographic Parity**: Selection rates equal across groups

**When base rates differ between groups** (which they do in my data), satisfying one violates another.

**Practical Implication**:
You must *choose* which fairness definition to optimize. This is why I presented multiple fairness metrics to stakeholders—it's a value judgment, not a technical decision.

**Example in My Project**:
If Group A has a 20% default rate and Group B has 10%, enforcing demographic parity (equal approval rates) would mean:
- Approving more high-risk applicants from Group A (lowers calibration)
- OR rejecting more low-risk applicants from Group B (lowers equal opportunity)

Neither is obviously 'fair'—it depends on your definition of fairness.

**Consulting Perspective**:
When working with Artefact's clients, I'd facilitate a conversation about:
- Legal requirements (ECOA mandates what?)
- Business priorities (minimize default risk vs maximize approval rate?)
- Ethical principles (equal opportunity vs equal outcome?)

Then implement the chosen fairness metric with full transparency about tradeoffs."

---

## 3. MODEL VALIDATION & STABILITY

### Q: "How did you prevent overfitting?"

**Answer**:
"Five strategies:

**1. Temporal Validation (Most Important)**:
- Trained on 2012-2015, tested on 2016 (1 year ahead)
- Also trained on 2017-2018, tested on 2019-2020 (1-2 years ahead)
- This is stricter than random train-test splits—if the model overfit to training period specifics, it would fail on future data
- Result: AUC changed only 0.06% → model generalized well

**2. Class Weight Balancing**:
- Used `class_weight='balanced'` to prevent model from exploiting class imbalance
- Without this, model could achieve 80% accuracy by always predicting 'no default' if defaults are 20% of data

**3. Conservative Hyperparameters**:
- n_estimators=500 (not 5000)
- learning_rate=0.05 (not 0.01—faster learning, less memorization)
- No extensive hyperparameter tuning to avoid overfitting to validation set

**4. Feature Engineering Validation**:
- Engineered features (loan_to_income, diversification) were based on domain knowledge, not data snooping
- K-means clustering reduced high-cardinality features (emp_title, zip_code) to prevent overfitting to rare categories

**5. Surrogate Model Validation**:
- Linear surrogate R²=0.91 shows LGBM isn't relying on spurious patterns
- If LGBM had overfit, it would capture noise that linear models can't, resulting in low surrogate R²

**What I Didn't Do (but could)**:
- Cross-validation: I used temporal splits instead, which is more appropriate for time-series data
- Regularization: LGBM has implicit regularization via boosting; explicit L1/L2 could help but wasn't needed given stability results
- Early stopping: Could monitor validation AUC and stop when it plateaus"

---

### Q: "What is concept drift and how did you test for it?"

**Answer**:
"**Concept drift** occurs when the relationship between features (X) and target (Y) changes over time.

**Types of Drift**:
1. **Covariate Drift**: P(X) changes (e.g., average loan size increases)
2. **Prior Drift**: P(Y) changes (e.g., default rate increases during recession)
3. **Concept Drift**: P(Y|X) changes (e.g., debt-to-income ratio becomes less predictive)

**My Testing Strategy**:

**1. Feature Importance Stability (Concept Drift Test)**:
- Compared LGBM feature importances between 2016 and 2019-2020 models
- Spearman correlation: 0.9657
- Interpretation: Feature rankings are nearly identical → P(Y|X) relationship is stable
- If concept drift occurred, important features in 2016 would be unimportant in 2019

**2. Performance Stability**:
- AUC 2016: 0.7177
- AUC 2019-2020: 0.7181
- Change: +0.06%
- Interpretation: Predictive power didn't degrade → model remains valid

**3. Fairness Stability**:
- Compared demographic parity ratios across time periods
- If drift occurred, bias patterns might change (e.g., features become more/less biased)

**Why This Matters**:
In production, I'd implement:
- **Monitoring**: Track feature importance monthly, alert if correlation drops below 0.9
- **Retraining**: Retrain quarterly on recent data
- **A/B Testing**: Shadow deploy new model, compare to production
- **Data Drift Detection**: Use KS-test or MMD to detect covariate drift

**Red Flags for Concept Drift**:
- AUC drops > 5% on recent data
- Feature importance correlation < 0.8
- Calibration breaks (predicted probabilities don't match observed frequencies)

None of these occurred in my analysis, proving model is robust."

---

## 4. FEATURE ENGINEERING & PREPROCESSING

### Q: "Why use K-means clustering for emp_title and zip_code?"

**Answer**:
"**Problem**: High-cardinality categorical features

- emp_title: Thousands of unique job titles ('Software Engineer', 'Sr. Software Engineer', 'Software Developer'—all similar)
- zip_code: Hundreds of unique ZIP codes with sparse coverage

**Naive Solutions (and why they fail)**:
1. **One-hot encoding**: Creates thousands of sparse features → overfitting, memory issues
2. **Label encoding**: Imposes arbitrary order (ZIP 10001 < 10002) that doesn't reflect similarity
3. **Drop features**: Loses valuable signal

**K-means Clustering Solution**:

**For emp_title** (10 clusters):
- Embeds job titles in feature space (annual income, loan amount, etc.)
- Clusters similar professions (e.g., 'Software Engineer' and 'Data Scientist' → Cluster 1: 'High-income tech')
- Reduces cardinality: 1000s of titles → 10 clusters
- Preserves signal: Similar jobs behave similarly

**For zip_code** (20 clusters):
- Embeds ZIP codes in feature space (median income, default rates, demographics)
- Clusters geographically/economically similar regions
- Reduces cardinality: 100s of ZIPs → 20 clusters
- Captures regional effects without overfitting to specific ZIPs

**Alternative Approaches**:
- **Target encoding**: Mean default rate per category (but risks leakage)
- **Frequency encoding**: Replace with category frequency (loses semantic meaning)
- **Entity embeddings**: Learn dense representations (requires neural networks)

**Why K-means Specifically**:
- Simple, interpretable
- Preserves feature relationships (clusters are meaningful)
- Integrates with scikit-learn pipelines (I created custom Preprocessor class)
- Prevents overfitting while retaining signal

**Validation**:
- I set random_state=42 for reproducibility
- Tested different cluster counts (10 vs 20 vs 50)—10 and 20 balanced complexity and performance"

---

### Q: "Walk me through your feature engineering rationale"

**Answer**:
"I created 5 domain-specific features based on credit risk theory:

**1. `loan_to_income = loan_amnt / annual_inc`**:
- **Rationale**: Debt burden relative to ability to repay
- **Hypothesis**: Higher ratio → higher default risk
- **Domain Knowledge**: Standard underwriting metric (DTI)
- **Expected Importance**: High (top 10 feature)

**2. `revol_to_income = revol_bal / annual_inc`**:
- **Rationale**: Credit utilization relative to income
- **Hypothesis**: High revolving debt suggests financial stress
- **Domain Knowledge**: Credit scoring models use utilization heavily
- **Expected Importance**: Medium

**3. `active_bc_ratio = num_actv_bc_tl / total_bc_limit`**:
- **Rationale**: Active credit lines relative to available credit
- **Hypothesis**: Maxing out credit lines indicates risk
- **Expected Importance**: Medium (correlates with credit score)

**4. `diversification = num_actv_rev_tl / total_acc`**:
- **Rationale**: Mix of credit types (installment vs revolving)
- **Hypothesis**: Diversified credit profile indicates financial sophistication
- **Domain Knowledge**: FICO considers credit mix
- **Expected Importance**: Low-medium

**5. `credit_age_diff = mo_sin_old_il_acct - mths_since_recent_bc`**:
- **Rationale**: Difference between oldest and newest accounts
- **Hypothesis**: Established credit history (large diff) → lower risk
- **Domain Knowledge**: Length of credit history matters in scoring
- **Expected Importance**: Medium

**Validation**:
- All features were created *before* seeing test data (no leakage)
- Checked for collinearity (VIF < 5 for all engineered features)
- Compared model with vs without engineered features: AUC improved by [X]%

**What I Didn't Engineer (but could)**:
- Polynomial interactions (e.g., int_rate × dti)—LGBM captures these implicitly
- Time-based features (seasonality)—not applicable for individual loans
- Text features from loan purpose—NLP is overkill for this task"

---

## 5. BUSINESS & DEPLOYMENT

### Q: "How did you optimize the decision threshold?"

**Answer**:
"**Standard Approach**: Threshold = 0.5 (predict default if P(Y=1) > 0.5)

**Problem**: This ignores business costs
- False Negative (missed default): Lender loses entire loan amount
- False Positive (rejected good loan): Lender loses interest income

These costs are NOT equal → threshold should reflect this.

**My Optimization Process**:

**1. Define Business Objective**:
- Maximize Profit & Loss (P&L)
- P&L = (True Negatives × interest income) - (False Negatives × loan loss)
- Simplified: Focus on minimizing costly errors

**2. Grid Search Over Thresholds**:
```python
thresholds = np.arange(0, 1, 0.01)  # 101 thresholds
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    pnl = calculate_pnl(y_true, y_pred)
    # Track best threshold
```

**3. Result**: θ_optimal = 0.78

**Interpretation**:
- Much higher than default 0.5
- Means: Only predict default if model is 78% confident
- Effect: Fewer approvals (lower FPR), but higher precision (fewer bad loans)
- Business Impact: Reduces costly false negatives at expense of some missed opportunities

**Validation**:
- Plotted P&L vs threshold curve—clear maximum at 0.78
- Tested on 2019-2020 data—threshold remained optimal
- Compared to 0.5 threshold: P&L improved by [X]%

**Why Not Use This Threshold in Fairness Analysis?**:
- I did! Fairness metrics use θ=0.78 for predictions
- This ensures fairness audit reflects real deployment scenario

**Production Considerations**:
- Update threshold quarterly as costs/market conditions change
- A/B test thresholds with different customer segments
- Provide threshold confidence intervals (bootstrap resampling)"

---

### Q: "How would you deploy this model in production?"

**Answer**:
"**Deployment Architecture**:

**1. Model Serving**:
- Save pipeline as pickle: `lgbm_pipeline_fitted.pkl`
- Wrap in REST API (Flask or FastAPI):
  ```python
  @app.post('/predict')
  def predict(loan_data: LoanApplication):
      X = preprocess(loan_data)
      proba = model.predict_proba(X)[0, 1]
      prediction = (proba >= THRESHOLD).astype(int)
      explanation = get_lime_explanation(X)
      return {
          'decision': 'approve' if prediction == 0 else 'deny',
          'confidence': proba,
          'explanation': explanation
      }
  ```
- Deploy on Kubernetes for scaling

**2. Explainability Service**:
- Pre-compute SHAP values for common profiles
- Generate LIME explanations on-demand
- Cache explanations for similar profiles (cosine similarity)
- Return top 5 features in API response for regulatory compliance

**3. Monitoring Dashboard**:
- Track:
  - Prediction distribution (drift detection)
  - Fairness metrics (demographic parity, equalized odds) by day/week
  - Feature importance evolution (Spearman correlation vs baseline)
  - Model performance (if ground truth available with lag)
- Alert if:
  - Fairness ratio < 0.8
  - Feature importance correlation < 0.9
  - AUC drops > 5%

**4. A/B Testing Framework**:
- Shadow deploy new model versions
- Route 5% of traffic to challenger model
- Compare performance after 1 month
- Promote if improvement is statistically significant

**5. Feedback Loop**:
- Collect ground truth (did loan default?) with 1-2 year lag
- Retrain quarterly on recent data
- Version control models (MLflow or Weights & Biases)

**6. Governance**:
- Model card documentation (performance, fairness, limitations)
- Audit trail: Log all predictions with protected attributes
- Review process: Monthly fairness audit, quarterly retraining
- Rollback procedure if bias detected

**Tech Stack**:
- Serving: FastAPI + Docker + Kubernetes
- Monitoring: Grafana + Prometheus + custom Python scripts
- Storage: PostgreSQL (predictions), S3 (models)
- Orchestration: Airflow or Prefect (retraining pipeline)
- Explainability: Dedicated microservice (LIME/SHAP)

**Latency Targets**:
- Prediction: < 100ms (p95)
- Explanation: < 500ms (p95)
- Batch scoring: 10k predictions / minute"

---

## 6. MACHINE LEARNING FUNDAMENTALS

### Q: "Explain gradient boosting in simple terms"

**Answer**:
"**Analogy**: Building a team of specialists

**Traditional Model**: One expert tries to solve everything
**Gradient Boosting**: Build a team where each new member fixes the previous team's mistakes

**How It Works**:

**Step 1**: Train a weak model (simple decision tree) on the data
- Prediction: Ŷ₁
- Error: y - Ŷ₁

**Step 2**: Train another weak model to predict the ERROR from Step 1
- This model learns what the first model got wrong
- Prediction: Ŷ₂ (corrects errors)

**Step 3**: Combine: Ŷ = Ŷ₁ + Ŷ₂
- Now we have a better prediction

**Repeat**: Train Ŷ₃ to fix errors of Ŷ₁ + Ŷ₂, and so on
- After 500 iterations (n_estimators=500), we have a strong ensemble

**Why 'Gradient'?**:
- Each new model is fitted to the gradient (derivative) of the loss function
- This is the direction of steepest error reduction
- Mathematically equivalent to gradient descent in function space

**LGBM Specifics**:
- Uses histogram-based tree building (faster)
- Grows trees leaf-wise instead of level-wise (more accurate but risks overfitting)
- Handles categorical features natively

**Intuition for Interviews**:
'Imagine learning to play basketball. First, you practice shooting—you're okay but make mistakes. Then you practice dribbling to fix ball-handling errors. Then defense to stop losing. Each iteration improves weaknesses. Gradient boosting does this for predictions.'

**Tradeoffs**:
- Pros: High accuracy, handles non-linear patterns, less feature engineering needed
- Cons: Black-box, slow to train, prone to overfitting without tuning, hard to interpret (hence my project!)"

---

### Q: "What's the bias-variance tradeoff?"

**Answer**:
"**Bias**: Model's tendency to miss relevant patterns (underfitting)
**Variance**: Model's sensitivity to small changes in training data (overfitting)

**Total Error = Bias² + Variance + Irreducible Error**

**Visual Analogy**:
Imagine shooting arrows at a target:
- **High Bias, Low Variance**: All arrows cluster tightly, but far from bullseye (consistently wrong)
- **Low Bias, High Variance**: Arrows scattered around bullseye (inconsistently right/wrong)
- **Low Bias, Low Variance**: All arrows cluster at bullseye (ideal)

**In My Project**:

**High Bias Models** (underfitting):
- Linear surrogate (R²=0.91): Misses non-linear patterns
- Simple decision tree: Captures only basic splits

**High Variance Models** (overfitting):
- Deep decision tree (depth=50): Memorizes training data
- Unregularized LGBM (n_estimators=10000): Overfits to noise

**My LGBM (balanced)**:
- n_estimators=500: Enough to capture patterns, not so many that it overfits
- learning_rate=0.05: Prevents each tree from fitting noise too closely
- Validation: AUC stable across time periods → low variance

**How I Balanced It**:
1. **Temporal validation**: Detects overfitting (high variance) because overfit models fail on future data
2. **Class weight balancing**: Reduces bias toward majority class
3. **Feature engineering**: Adds signal without adding noise
4. **Surrogate model check**: R²=0.91 shows LGBM isn't wildly complex (controlled variance)

**In Consulting Context**:
- High bias: Client says 'model is too simple, missing opportunities'
- High variance: Client says 'model worked in testing but fails in production'
- Goal: Find the Goldilocks zone—just right complexity"

---

## 7. STATISTICS & HYPOTHESIS TESTING

### Q: "Explain the chi-squared test you used for bias detection"

**Answer**:
"**Purpose**: Test if predicted outcome is independent of protected attribute

**Null Hypothesis (H₀)**: Prediction and protected attribute are independent (fair)
**Alternative Hypothesis (H₁)**: They are dependent (biased)

**How It Works**:

**Step 1**: Create contingency table
```
                | Protected = 0 | Protected = 1 |
Predicted = 0   |      a        |      b        |
Predicted = 1   |      c        |      d        |
```

**Step 2**: Calculate expected counts (if independent)
```
Expected(a) = (row total × column total) / grand total
```

**Step 3**: Compute chi-squared statistic
```
χ² = Σ [(Observed - Expected)² / Expected]
```

**Step 4**: Compare to chi-squared distribution
- If χ² is large → observed and expected differ significantly → dependence
- p-value < 0.05 → reject H₀ → biased

**My Implementation**:

For each feature (e.g., int_rate):
1. Bin feature values into quantiles (if continuous)
2. For each feature level, test if (prediction, protected attribute) are independent
3. Take minimum p-value across all levels
4. If any p < 0.05, feature is biased

**Results**:
- int_rate: p < 0.05 → ❌ Biased
- grade: p < 0.05 → ❌ Biased
- pub_rec: p > 0.05 → ✅ Fair

**Interpretation**:
'At certain interest rate levels, the model's predictions differ significantly between protected groups, even after controlling for the interest rate itself. This suggests the model is learning a proxy relationship.'

**Limitations**:
- Assumes sufficient cell counts (I used n > 10 threshold)
- Tests independence, not causation
- Multiple testing issue (I tested 8 features—could use Bonferroni correction)

**Why This Test for Equalized Odds**:
- Equalized odds requires TPR and FPR equal across groups
- Chi-squared tests if prediction is independent of protected attribute conditional on outcome
- Equivalent to testing if P(Ŷ | Y, A) = P(Ŷ | Y) (prediction doesn't depend on protected attribute given outcome)"

---

This guide should prepare you for deep technical dives. Practice explaining these concepts out loud!
