# ============================================================================
# FEATURE IMPORTANCE CONSENSUS ANALYSIS
# Insert this cell after Step 8 (Permutation Importance)
# This compares ALL feature importance methods used in the project
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

print("="*100)
print("FEATURE IMPORTANCE CONSENSUS ANALYSIS")
print("Comparing: LGBM Native | Permutation | SHAP | Surrogate Models")
print("="*100)
print()

# ============================================================================
# 1. COLLECT FEATURE IMPORTANCES FROM ALL METHODS
# ============================================================================

# Get feature names
feature_names = lgbm_pipeline_fitted_2016[:-1].get_feature_names_out()

# METHOD 1: LGBM Native Feature Importance (Gain-based)
lgbm_model = lgbm_pipeline_fitted_2016.named_steps['lgbmclassifier']
lgbm_importance = lgbm_model.feature_importances_

# METHOD 2: Permutation Importance (if already computed)
# Load from saved CSV or recompute
try:
    perm_importance_df = pd.read_csv('step8_permutation_importance_2016.csv')
    perm_importance = perm_importance_df.set_index('feature')['importance'].reindex(feature_names).fillna(0).values
except:
    print("⚠️  Permutation importance not found. Run Step 8 first or computing now...")
    from sklearn.inspection import permutation_importance
    X_val_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_val_2016)
    perm_result = permutation_importance(lgbm_model, X_val_processed, y_val_2016,
                                          n_repeats=10, random_state=42, scoring='roc_auc')
    perm_importance = perm_result.importances_mean

# METHOD 3: SHAP Values (if already computed)
try:
    shap_importance_df = pd.read_csv('shap_feature_importance_ranking.csv')
    shap_importance = shap_importance_df.set_index('Feature')['Mean |SHAP|'].reindex(feature_names).fillna(0).values
except:
    print("⚠️  SHAP importance not found. Computing now...")
    import shap
    X_val_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_val_2016)
    explainer = shap.TreeExplainer(lgbm_model)

    # Use smaller sample for speed
    sample_size = min(1000, len(X_val_processed))
    sample_indices = np.random.choice(len(X_val_processed), sample_size, replace=False)
    shap_values = explainer.shap_values(X_val_processed[sample_indices])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    shap_importance = np.abs(shap_values).mean(axis=0)

# METHOD 4: Surrogate Model (Linear Regression Coefficients)
# From Step 4 - Global surrogate on LGBM
from sklearn.linear_model import LinearRegression

X_val_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_val_2016)
y_proba_lgbm = lgbm_model.predict_proba(X_val_processed)[:, 1]

surrogate_lr = LinearRegression()
surrogate_lr.fit(X_val_processed, y_proba_lgbm)
surrogate_importance = np.abs(surrogate_lr.coef_)

# ============================================================================
# 2. CREATE COMPARISON DATAFRAME
# ============================================================================

# Normalize all importances to [0, 1] for fair comparison
def normalize(arr):
    if arr.max() == 0:
        return arr
    return arr / arr.max()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'LGBM Native': normalize(lgbm_importance),
    'Permutation': normalize(perm_importance),
    'SHAP': normalize(shap_importance),
    'Surrogate (LR)': normalize(surrogate_importance)
})

# Add mean importance across all methods
importance_df['Mean Importance'] = importance_df[['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']].mean(axis=1)

# Add standard deviation to see consensus
importance_df['Std Dev'] = importance_df[['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']].std(axis=1)

# Add rank for each method
for col in ['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']:
    importance_df[f'{col} Rank'] = importance_df[col].rank(ascending=False, method='min').astype(int)

# Sort by mean importance
importance_df = importance_df.sort_values('Mean Importance', ascending=False)

# ============================================================================
# 3. DISPLAY TOP FEATURES WITH CONSENSUS
# ============================================================================

print("="*100)
print("TOP 20 FEATURES: CONSENSUS ACROSS ALL METHODS")
print("="*100)
print()

display_df = importance_df.head(20)[['Feature', 'LGBM Native', 'Permutation', 'SHAP',
                                      'Surrogate (LR)', 'Mean Importance', 'Std Dev']]
print(display_df.to_string(index=False))
print()

# ============================================================================
# 4. IDENTIFY HIGH-CONSENSUS FEATURES
# ============================================================================

# High consensus = low std dev and high mean importance
high_consensus = importance_df[
    (importance_df['Mean Importance'] > importance_df['Mean Importance'].quantile(0.75)) &
    (importance_df['Std Dev'] < importance_df['Std Dev'].quantile(0.5))
].head(10)

print("="*100)
print("HIGH CONSENSUS FEATURES (High Importance + Low Disagreement)")
print("="*100)
print(high_consensus[['Feature', 'Mean Importance', 'Std Dev']].to_string(index=False))
print()

# ============================================================================
# 5. VISUALIZE FEATURE IMPORTANCE COMPARISON
# ============================================================================

# Plot 1: Top 20 features across all methods
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Feature Importance Comparison: Top 20 Features Across All Methods',
             fontsize=18, fontweight='bold', y=0.995)

methods = ['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (method, color) in enumerate(zip(methods, colors)):
    ax = axes[idx // 2, idx % 2]

    # Get top 20 for this method
    top_features = importance_df.nlargest(20, method)

    y_pos = np.arange(len(top_features))
    values = top_features[method].values

    bars = ax.barh(y_pos, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
    ax.set_xlabel('Normalized Importance', fontweight='bold', fontsize=11)
    ax.set_title(f'{method}', fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison_by_method.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. HEATMAP: FEATURE IMPORTANCE ACROSS METHODS
# ============================================================================

# Select top 25 features by mean importance
top_25 = importance_df.head(25)

# Create matrix for heatmap
heatmap_data = top_25[['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']].values

fig, ax = plt.subplots(figsize=(12, 14))

sns.heatmap(
    heatmap_data,
    xticklabels=['LGBM\nNative', 'Permutation', 'SHAP', 'Surrogate\n(LR)'],
    yticklabels=top_25['Feature'].values,
    cmap='YlOrRd',
    annot=True,
    fmt='.3f',
    linewidths=0.5,
    cbar_kws={'label': 'Normalized Importance'},
    ax=ax
)

ax.set_title('Feature Importance Heatmap: Top 25 Features Across All Methods',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Method', fontweight='bold', fontsize=12)
ax.set_ylabel('Feature', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. RANK CORRELATION ANALYSIS
# ============================================================================

print("="*100)
print("RANK CORRELATION ANALYSIS (Spearman's ρ)")
print("Measures agreement between different importance methods")
print("="*100)
print()

# Compute rank correlation matrix
methods = ['LGBM Native', 'Permutation', 'SHAP', 'Surrogate (LR)']
correlation_matrix = np.zeros((len(methods), len(methods)))

for i, method1 in enumerate(methods):
    for j, method2 in enumerate(methods):
        rho, _ = spearmanr(importance_df[method1], importance_df[method2])
        correlation_matrix[i, j] = rho

correlation_df = pd.DataFrame(correlation_matrix, index=methods, columns=methods)
print(correlation_df.round(4))
print()

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    correlation_df,
    annot=True,
    fmt='.4f',
    cmap='coolwarm',
    center=0.5,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=2,
    cbar_kws={'label': "Spearman's ρ"},
    ax=ax
)

ax.set_title("Feature Importance Rank Correlation\n(How Well Do Methods Agree?)",
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('feature_importance_rank_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. CONSENSUS RANKING
# ============================================================================

print("="*100)
print("FINAL CONSENSUS RANKING: TOP 15 FEATURES")
print("Based on mean importance across all methods")
print("="*100)
print()

consensus_top_15 = importance_df.head(15)[['Feature', 'Mean Importance', 'Std Dev',
                                             'LGBM Native Rank', 'Permutation Rank',
                                             'SHAP Rank', 'Surrogate (LR) Rank']]

for idx, row in consensus_top_15.iterrows():
    print(f"{consensus_top_15.index.get_loc(idx) + 1:2d}. {row['Feature']:35s} "
          f"| Mean: {row['Mean Importance']:.4f} | Std: {row['Std Dev']:.4f}")
    print(f"    Ranks: LGBM={row['LGBM Native Rank']:3.0f} | "
          f"Perm={row['Permutation Rank']:3.0f} | "
          f"SHAP={row['SHAP Rank']:3.0f} | "
          f"Surrogate={row['Surrogate (LR) Rank']:3.0f}")
    print()

# ============================================================================
# 9. VISUALIZE CONSENSUS WITH ERROR BARS
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

top_20_consensus = importance_df.head(20)
y_pos = np.arange(len(top_20_consensus))

# Plot mean importance with std dev as error bars
ax.barh(y_pos, top_20_consensus['Mean Importance'].values,
        xerr=top_20_consensus['Std Dev'].values,
        color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=2,
        capsize=5, error_kw={'linewidth': 2})

# Add value labels
for i, (mean, std) in enumerate(zip(top_20_consensus['Mean Importance'].values,
                                     top_20_consensus['Std Dev'].values)):
    ax.text(mean + std + 0.02, i, f'{mean:.3f} ± {std:.3f}',
            ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_20_consensus['Feature'].values, fontsize=11)
ax.set_xlabel('Mean Importance Across All Methods', fontweight='bold', fontsize=12)
ax.set_title('Feature Importance Consensus: Top 20 Features\n(Mean ± Std Dev Across 4 Methods)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_consensus_with_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

importance_df.to_csv('feature_importance_comparison_all_methods.csv', index=False)
correlation_df.to_csv('feature_importance_rank_correlation.csv')

print("="*100)
print("KEY INSIGHTS:")
print("="*100)
print(f"✅ Strongest correlation: {correlation_df.values[np.triu_indices_from(correlation_df.values, k=1)].max():.4f}")
print(f"✅ Weakest correlation: {correlation_df.values[np.triu_indices_from(correlation_df.values, k=1)].min():.4f}")
print(f"✅ Top consensus feature: {importance_df.iloc[0]['Feature']}")
print(f"✅ Most controversial feature: {importance_df.nlargest(20, 'Std Dev').iloc[0]['Feature']} (high std dev)")
print("="*100)

print("\n✅ All feature importance comparison files saved!")
print("Files created:")
print("  • feature_importance_comparison_all_methods.csv")
print("  • feature_importance_rank_correlation.csv")
print("  • feature_importance_comparison_by_method.png")
print("  • feature_importance_heatmap.png")
print("  • feature_importance_rank_correlation.png")
print("  • feature_importance_consensus_with_uncertainty.png")
