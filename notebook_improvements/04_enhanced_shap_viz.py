# ============================================================================
# ENHANCED SHAP VISUALIZATIONS FOR INTERVIEW
# Insert this cell after Cell 55 (after existing SHAP runs)
# ============================================================================

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (16, 12)
shap.initjs()  # Initialize JavaScript for interactive plots

print("="*100)
print("SHAP ANALYSIS: COMPREHENSIVE FEATURE ATTRIBUTION")
print("="*100)
print()

# ============================================================================
# 1. CALCULATE SHAP VALUES FOR TEST SET
# ============================================================================

print("📊 Computing SHAP values for 2016 validation set...")
print("⏳ This may take a few minutes for large datasets...")

# Get processed test data
X_val_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_val_2016)
feature_names = lgbm_pipeline_fitted_2016[:-1].get_feature_names_out()

# Extract the LGBM model from pipeline
lgbm_model = lgbm_pipeline_fitted_2016.named_steps['lgbmclassifier']

# Create SHAP explainer
explainer = shap.TreeExplainer(lgbm_model)

# Calculate SHAP values for a sample (use subset for efficiency)
# For interview: use 1000-2000 samples for good coverage without excessive compute time
sample_size = min(2000, len(X_val_processed))
np.random.seed(42)
sample_indices = np.random.choice(len(X_val_processed), sample_size, replace=False)

X_sample = X_val_processed[sample_indices]
y_sample = y_val_2016[sample_indices]

shap_values = explainer.shap_values(X_sample)

# For binary classification, get SHAP values for positive class (Default = 1)
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # Class 1 (Default)
else:
    shap_values_positive = shap_values

print(f"✅ SHAP values computed for {sample_size} instances")
print(f"   Shape: {shap_values_positive.shape}")
print()

# ============================================================================
# 2. SHAP SUMMARY PLOT (BAR) - GLOBAL FEATURE IMPORTANCE
# ============================================================================

print("="*100)
print("1️⃣  SHAP SUMMARY PLOT (BAR): Global Feature Importance")
print("="*100)
print("Shows the average absolute SHAP value for each feature")
print("Higher value = more important for model predictions")
print()

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_positive,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title('SHAP Feature Importance: Top 20 Features\n(Average Absolute Impact on Predictions)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. SHAP SUMMARY PLOT (BEE SWARM) - FEATURE IMPACT DISTRIBUTION
# ============================================================================

print("\n" + "="*100)
print("2️⃣  SHAP SUMMARY PLOT (BEE SWARM): Feature Impact Distribution")
print("="*100)
print("Shows how feature values affect predictions:")
print("  • Red points: High feature value")
print("  • Blue points: Low feature value")
print("  • X-axis: SHAP value (positive = increases default risk)")
print()

plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values_positive,
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_size=(12, 10)
)
plt.title('SHAP Value Distribution: How Feature Values Impact Predictions\n(Red = High Feature Value, Blue = Low Feature Value)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impact on Default Prediction)', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('shap_feature_distribution_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. SHAP WATERFALL PLOTS - INDIVIDUAL PREDICTIONS
# ============================================================================

print("\n" + "="*100)
print("3️⃣  SHAP WATERFALL PLOTS: Individual Prediction Explanations")
print("="*100)
print("Shows how each feature contributes to a specific prediction")
print()

# Select same representative cases as LIME for comparison
# Case 1: True Positive (High Confidence)
tp_high_conf_local = np.where((y_sample == 1) &
                               (lgbm_model.predict_proba(X_sample)[:, 1] > 0.90))[0]
case1_local = tp_high_conf_local[0] if len(tp_high_conf_local) > 0 else 0

# Case 2: False Negative
fn_local = np.where((y_sample == 1) &
                     (lgbm_model.predict_proba(X_sample)[:, 1] < 0.78))[0]
case2_local = fn_local[0] if len(fn_local) > 0 else 1

# Case 3: True Negative (High Confidence)
tn_high_conf_local = np.where((y_sample == 0) &
                               (lgbm_model.predict_proba(X_sample)[:, 1] < 0.10))[0]
case3_local = tn_high_conf_local[0] if len(tn_high_conf_local) > 0 else 2

# Case 4: False Positive
fp_local = np.where((y_sample == 0) &
                     (lgbm_model.predict_proba(X_sample)[:, 1] >= 0.78))[0]
case4_local = fp_local[0] if len(fp_local) > 0 else 3

representative_cases = [
    (case1_local, "True Positive (High Confidence)", "green"),
    (case2_local, "False Negative (Missed Default)", "red"),
    (case3_local, "True Negative (High Confidence)", "blue"),
    (case4_local, "False Positive (Incorrect Alert)", "orange")
]

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('SHAP Waterfall Plots: Explaining Individual Predictions',
             fontsize=18, fontweight='bold', y=0.995)

for idx, (case_idx, case_name, color) in enumerate(representative_cases):
    ax = axes[idx // 2, idx % 2]

    # Get prediction details
    pred_proba = lgbm_model.predict_proba(X_sample[case_idx:case_idx+1])[:, 1][0]
    true_label = 'Default' if y_sample[case_idx] == 1 else 'No Default'
    pred_label = 'Default' if pred_proba >= 0.78 else 'No Default'

    # Create explanation object for this instance
    explanation = shap.Explanation(
        values=shap_values_positive[case_idx],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                     else explainer.expected_value,
        data=X_sample[case_idx],
        feature_names=feature_names
    )

    # Plot waterfall
    shap.waterfall_plot(explanation, max_display=12, show=False)

    # Customize title
    plt.sca(ax)
    ax.set_title(
        f"{case_name}\n"
        f"True: {true_label} | Predicted: {pred_label} (p={pred_proba:.3f})",
        fontweight='bold', fontsize=12, color=color, pad=10
    )

plt.tight_layout()
plt.savefig('shap_waterfall_plots_representative_cases.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. SHAP DEPENDENCE PLOTS - FEATURE INTERACTIONS
# ============================================================================

print("\n" + "="*100)
print("4️⃣  SHAP DEPENDENCE PLOTS: Feature Interactions")
print("="*100)
print("Shows how a feature's value affects its SHAP contribution")
print("Color represents interaction with another feature")
print()

# Plot dependence for top 4 features
shap_importance = np.abs(shap_values_positive).mean(axis=0)
top_features_idx = np.argsort(shap_importance)[-4:][::-1]
top_features = [feature_names[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHAP Dependence Plots: How Feature Values Drive Predictions',
             fontsize=16, fontweight='bold', y=0.995)

for idx, feature_idx in enumerate(top_features_idx):
    ax = axes[idx // 2, idx % 2]

    shap.dependence_plot(
        feature_idx,
        shap_values_positive,
        X_sample,
        feature_names=feature_names,
        show=False,
        ax=ax
    )

    ax.set_title(f'Impact of {feature_names[feature_idx]}',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel(f'{feature_names[feature_idx]} Value', fontweight='bold')
    ax.set_ylabel('SHAP Value\n(Impact on Default Prediction)', fontweight='bold')

plt.tight_layout()
plt.savefig('shap_dependence_plots_top_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. SHAP FORCE PLOT - ALTERNATIVE VISUALIZATION
# ============================================================================

print("\n" + "="*100)
print("5️⃣  SHAP FORCE PLOT: Visualizing Prediction Drivers")
print("="*100)
print("Shows how features push prediction from base value to final prediction")
print()

# Create force plot for a single instance (True Positive case)
instance_idx = case1_local

plt.figure(figsize=(20, 3))
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
    shap_values_positive[instance_idx],
    X_sample[instance_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f'SHAP Force Plot: True Positive Case (p={lgbm_model.predict_proba(X_sample[instance_idx:instance_idx+1])[:, 1][0]:.3f})',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_force_plot_example.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. SHAP FEATURE IMPORTANCE TABLE
# ============================================================================

print("\n" + "="*100)
print("SHAP FEATURE IMPORTANCE RANKING")
print("="*100)

# Calculate mean absolute SHAP values
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': np.abs(shap_values_positive).mean(axis=0),
    'Mean SHAP': shap_values_positive.mean(axis=0),
    'Std SHAP': shap_values_positive.std(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

print("\nTop 20 Features by SHAP Importance:")
print("-"*100)
print(shap_importance_df.head(20).to_string(index=False))
print()

# ============================================================================
# 8. SAVE SHAP VALUES FOR FURTHER ANALYSIS
# ============================================================================

# Save SHAP values as numpy array
np.save('shap_values_2016_sample.npy', shap_values_positive)
shap_importance_df.to_csv('shap_feature_importance_ranking.csv', index=False)

print("\n" + "="*100)
print("KEY INSIGHTS FROM SHAP ANALYSIS")
print("="*100)
print(f"✅ SHAP provides theoretically grounded feature attributions (based on Shapley values)")
print(f"✅ Top feature: {shap_importance_df.iloc[0]['Feature']} "
      f"(Mean |SHAP| = {shap_importance_df.iloc[0]['Mean |SHAP|']:.4f})")
print(f"✅ Feature interactions visible in dependence plots")
print(f"✅ Individual predictions fully decomposed (waterfall plots)")
print(f"✅ SHAP values sum to (prediction - base_value), ensuring consistency")
print("="*100)

print("\n✅ All SHAP visualizations saved successfully!")
print("✅ Files created:")
print("   - shap_feature_importance_bar.png")
print("   - shap_feature_distribution_beeswarm.png")
print("   - shap_waterfall_plots_representative_cases.png")
print("   - shap_dependence_plots_top_features.png")
print("   - shap_force_plot_example.png")
print("   - shap_values_2016_sample.npy")
print("   - shap_feature_importance_ranking.csv")
