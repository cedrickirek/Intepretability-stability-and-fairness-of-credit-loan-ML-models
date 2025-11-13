# ============================================================================
# ENHANCED FAIRNESS ANALYSIS WITH ACTIONABLE RECOMMENDATIONS
# Insert this cell after Cell 63 (after existing fairness analysis)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*100)
print("COMPREHENSIVE FAIRNESS AUDIT: BIAS DETECTION & MITIGATION STRATEGIES")
print("="*100)
print()

# ============================================================================
# 1. COMPUTE STANDARD FAIRNESS METRICS
# ============================================================================

def compute_fairness_metrics(y_true, y_pred, y_proba, protected_attr, threshold=0.78):
    """
    Compute comprehensive fairness metrics for binary classification

    Metrics computed:
    - Demographic Parity (Selection Rate Parity)
    - Equalized Odds (TPR and FPR parity)
    - Equal Opportunity (TPR parity only)
    - Predictive Parity (PPV parity)
    - Calibration (by group)
    """

    # Binarize protected attribute (assume it's continuous - pct_afro_american)
    protected_median = np.median(protected_attr)
    protected_binary = (protected_attr >= protected_median).astype(int)
    group_names = {0: 'Low pct_afro_american', 1: 'High pct_afro_american'}

    results = {}

    for group in [0, 1]:
        mask = (protected_binary == group)

        if mask.sum() == 0:
            continue

        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        y_proba_group = y_proba[mask]

        # Compute metrics
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()

        results[group_names[group]] = {
            'n': mask.sum(),
            'base_rate': y_true_group.mean(),
            'selection_rate': y_pred_group.mean(),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True Positive Rate (Recall)
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate (Specificity)
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value (Precision)
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'mean_predicted_prob': y_proba_group.mean()
        }

    return pd.DataFrame(results).T, protected_binary, group_names

# Compute fairness metrics for 2016
fairness_2016, protected_binary_2016, group_names = compute_fairness_metrics(
    y_true=y_val_2016,
    y_pred=y_pred_lgbm_2016_adapted,
    y_proba=y_proba_lgbm_2016,
    protected_attr=full_df_val_1['pct_afro_american'].values,  # Adjust based on your data
    threshold=0.78
)

# Compute fairness metrics for 2019-2020
fairness_201920, protected_binary_201920, _ = compute_fairness_metrics(
    y_true=y_val_201920,
    y_pred=y_pred_lgbm_201920_adapted,
    y_proba=y_proba_lgbm_201920,
    protected_attr=full_df_val_2['pct_afro_american'].values,
    threshold=0.78
)

# ============================================================================
# 2. DISPLAY COMPREHENSIVE FAIRNESS METRICS
# ============================================================================

print("="*100)
print("FAIRNESS METRICS: 2016 VALIDATION")
print("="*100)
print(fairness_2016.round(4))
print()

print("="*100)
print("FAIRNESS METRICS: 2019-2020 FUTURE TEST")
print("="*100)
print(fairness_201920.round(4))
print()

# ============================================================================
# 3. CALCULATE FAIRNESS RATIOS AND DISPARITIES
# ============================================================================

def calculate_disparity_metrics(fairness_df):
    """Calculate disparity ratios between groups"""

    groups = fairness_df.index.tolist()
    if len(groups) < 2:
        return None

    group1, group2 = groups[0], groups[1]

    disparities = {}

    # Demographic Parity Ratio (80% rule)
    sr_ratio = min(fairness_df.loc[group1, 'selection_rate'],
                   fairness_df.loc[group2, 'selection_rate']) / \
               max(fairness_df.loc[group1, 'selection_rate'],
                   fairness_df.loc[group2, 'selection_rate'])

    disparities['Demographic Parity Ratio'] = sr_ratio
    disparities['Demographic Parity (80% Rule)'] = '✅ PASS' if sr_ratio >= 0.8 else '❌ FAIL'

    # Equalized Odds (TPR and FPR difference)
    tpr_diff = abs(fairness_df.loc[group1, 'tpr'] - fairness_df.loc[group2, 'tpr'])
    fpr_diff = abs(fairness_df.loc[group1, 'fpr'] - fairness_df.loc[group2, 'fpr'])

    disparities['TPR Difference'] = tpr_diff
    disparities['FPR Difference'] = fpr_diff
    disparities['Equalized Odds'] = '✅ PASS' if (tpr_diff < 0.05 and fpr_diff < 0.05) else '❌ FAIL'

    # Equal Opportunity (TPR difference only)
    disparities['Equal Opportunity'] = '✅ PASS' if tpr_diff < 0.05 else '❌ FAIL'

    # Predictive Parity (PPV difference)
    ppv_diff = abs(fairness_df.loc[group1, 'ppv'] - fairness_df.loc[group2, 'ppv'])
    disparities['PPV Difference'] = ppv_diff
    disparities['Predictive Parity'] = '✅ PASS' if ppv_diff < 0.05 else '❌ FAIL'

    return disparities

disparity_2016 = calculate_disparity_metrics(fairness_2016)
disparity_201920 = calculate_disparity_metrics(fairness_201920)

print("="*100)
print("DISPARITY ANALYSIS: 2016 VALIDATION")
print("="*100)
for metric, value in disparity_2016.items():
    print(f"{metric:35s}: {value}")
print()

print("="*100)
print("DISPARITY ANALYSIS: 2019-2020 FUTURE TEST")
print("="*100)
for metric, value in disparity_201920.items():
    print(f"{metric:35s}: {value}")
print()

# ============================================================================
# 4. VISUALIZE FAIRNESS METRICS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fairness Audit: Group Comparison (Low vs High pct_afro_american)',
             fontsize=16, fontweight='bold', y=1.00)

metrics_to_plot = [
    ('selection_rate', 'Selection Rate', 'Demographic Parity'),
    ('tpr', 'True Positive Rate', 'Equal Opportunity'),
    ('fpr', 'False Positive Rate', 'Equalized Odds'),
    ('ppv', 'Precision (PPV)', 'Predictive Parity'),
    ('accuracy', 'Accuracy', 'Overall Performance'),
    ('mean_predicted_prob', 'Mean Predicted Probability', 'Calibration')
]

for idx, (metric, title, fairness_type) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]

    # Data for 2016
    groups = fairness_2016.index.tolist()
    values_2016 = [fairness_2016.loc[g, metric] for g in groups]

    # Data for 2019-2020
    values_201920 = [fairness_201920.loc[g, metric] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width/2, values_2016, width, label='2016', alpha=0.8, color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, values_201920, width, label='2019-2020', alpha=0.8, color='#2ecc71', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'{title}\n({fairness_type})', fontweight='bold', fontsize=11)
    ax.set_ylabel(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('pct_afro_american', 'AA%') for g in groups], fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(values_2016), max(values_201920)) * 1.2])

plt.tight_layout()
plt.savefig('fairness_audit_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. CHI-SQUARED BIAS TEST (FROM ORIGINAL ANALYSIS)
# ============================================================================

print("\n" + "="*100)
print("CHI-SQUARED STATISTICAL BIAS TESTING (EQUALIZED ODDS)")
print("="*100)
print("Tests if predicted outcome is independent of protected attribute")
print("H0: Independent (fair) | H1: Dependent (biased)")
print("Significance level: α = 0.05")
print()

# Test key features for bias
features_to_test = ['int_rate', 'grade_encoded', 'sub_grade_encoded', 'annual_inc',
                     'dti', 'pub_rec', 'tax_liens', 'open_acc']

bias_results = []

for feature in features_to_test:
    try:
        # Get feature values
        feature_values = full_df_val_1[feature].values
        protected = protected_binary_2016
        predictions = y_pred_lgbm_2016_adapted

        # Bin continuous features
        if len(np.unique(feature_values)) > 20:
            feature_binned = pd.qcut(feature_values, q=10, duplicates='drop')
        else:
            feature_binned = feature_values

        # For each feature level, test independence of (prediction, protected)
        p_values = []
        for level in np.unique(feature_binned):
            mask = (feature_binned == level)
            if mask.sum() < 10:  # Skip levels with too few samples
                continue

            contingency_table = pd.crosstab(predictions[mask], protected[mask])

            if contingency_table.shape == (2, 2):  # Valid 2x2 table
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                p_values.append(p_value)

        if p_values:
            min_p_value = min(p_values)
            is_biased = min_p_value < 0.05

            bias_results.append({
                'Feature': feature,
                'Min P-value': min_p_value,
                'Biased?': '❌ YES' if is_biased else '✅ NO',
                'Interpretation': 'Violates equalized odds' if is_biased else 'Passes fairness test'
            })
    except Exception as e:
        print(f"⚠️  Could not test {feature}: {e}")

bias_df = pd.DataFrame(bias_results)
print(bias_df.to_string(index=False))
print()

# ============================================================================
# 6. IDENTIFY BIASED FEATURES
# ============================================================================

biased_features = bias_df[bias_df['Biased?'] == '❌ YES']['Feature'].tolist()
fair_features = bias_df[bias_df['Biased?'] == '✅ NO']['Feature'].tolist()

print("="*100)
print("BIAS DETECTION SUMMARY")
print("="*100)
print(f"🚨 BIASED FEATURES (p < 0.05): {len(biased_features)}")
for feat in biased_features:
    print(f"   ❌ {feat}")
print()
print(f"✅ FAIR FEATURES (p >= 0.05): {len(fair_features)}")
for feat in fair_features:
    print(f"   ✅ {feat}")
print()

# ============================================================================
# 7. ACTIONABLE RECOMMENDATIONS FOR BIAS MITIGATION
# ============================================================================

print("="*100)
print("ACTIONABLE RECOMMENDATIONS FOR BIAS MITIGATION")
print("="*100)
print()

print("🎯 STRATEGY 1: FEATURE ENGINEERING")
print("-"*100)
print("Action: Remove or replace biased features")
print("Pros: Simple to implement, reduces direct bias")
print("Cons: May reduce model performance")
print()
print("Recommendations:")
for feat in biased_features:
    print(f"  • Consider removing '{feat}' or replacing with less correlated proxy")
print()

print("🎯 STRATEGY 2: FAIRNESS-AWARE MODELING")
print("-"*100)
print("Action: Apply fairness constraints during training")
print("Pros: Balances performance and fairness")
print("Cons: Requires specialized algorithms")
print()
print("Recommended Libraries:")
print("  • Fairlearn (Microsoft): Implements demographic parity, equalized odds constraints")
print("  • AIF360 (IBM): Comprehensive fairness toolkit")
print("  • Themis-ML: Fair classification with reweighting")
print()
print("Implementation:")
print("  1. Use GridSearch with fairness constraint (e.g., demographic_parity_difference < 0.05)")
print("  2. Apply threshold optimizer to equalize TPR/FPR across groups")
print("  3. Use adversarial debiasing to remove protected attribute influence")
print()

print("🎯 STRATEGY 3: POST-PROCESSING CALIBRATION")
print("-"*100)
print("Action: Adjust decision thresholds per group")
print("Pros: Maintains model performance, flexible")
print("Cons: Requires group-specific thresholds")
print()
print("Recommendations:")
print(f"  • Current threshold: 0.78 (optimized for P&L)")
print(f"  • Group 1 (Low AA%): Selection rate = {fairness_2016.iloc[0]['selection_rate']:.3f}")
print(f"  • Group 2 (High AA%): Selection rate = {fairness_2016.iloc[1]['selection_rate']:.3f}")
print(f"  • Consider adjusting thresholds to equalize selection rates")
print()

print("🎯 STRATEGY 4: PREPROCESSING - DATA REWEIGHTING")
print("-"*100)
print("Action: Reweigh training samples to balance outcomes across groups")
print("Pros: Upstream solution, doesn't modify model architecture")
print("Cons: May require retraining")
print()
print("Implementation:")
print("  1. Compute reweighting factors using AIF360's Reweighing transformer")
print("  2. Apply sample weights during LGBM training")
print("  3. Validate fairness metrics after retraining")
print()

print("🎯 STRATEGY 5: MONITORING & CONTINUOUS AUDITING")
print("-"*100)
print("Action: Implement real-time fairness monitoring in production")
print("Pros: Detects drift, ensures ongoing compliance")
print("Cons: Requires infrastructure")
print()
print("Recommendations:")
print("  • Set up alerts for demographic parity ratio < 0.8")
print("  • Monitor TPR/FPR disparity monthly")
print("  • Log all predictions with protected attribute for audit trail")
print("  • Create fairness dashboard for stakeholders")
print()

# ============================================================================
# 8. ESTIMATED IMPACT ANALYSIS
# ============================================================================

print("="*100)
print("ESTIMATED IMPACT OF BIAS")
print("="*100)

# Calculate how many individuals are affected
total_samples = len(y_val_2016)
group1_samples = (protected_binary_2016 == 0).sum()
group2_samples = (protected_binary_2016 == 1).sum()

sr_diff = abs(fairness_2016.iloc[0]['selection_rate'] - fairness_2016.iloc[1]['selection_rate'])
affected_individuals = int(sr_diff * min(group1_samples, group2_samples))

print(f"Total samples: {total_samples:,}")
print(f"Group 1 (Low AA%): {group1_samples:,} ({group1_samples/total_samples*100:.1f}%)")
print(f"Group 2 (High AA%): {group2_samples:,} ({group2_samples/total_samples*100:.1f}%)")
print()
print(f"Selection Rate Difference: {sr_diff:.3f} ({sr_diff*100:.1f}%)")
print(f"Estimated Affected Individuals: ~{affected_individuals:,} loans")
print()

if disparity_2016['Demographic Parity (80% Rule)'] == '❌ FAIL':
    print("⚠️  MODEL FAILS 80% RULE: Selection rates differ by more than 20%")
    print("⚠️  REGULATORY RISK: May violate fair lending laws (ECOA, FHA)")
    print("⚠️  BUSINESS RISK: Potential litigation, reputational damage")
else:
    print("✅ MODEL PASSES 80% RULE: Selection rates are sufficiently similar")

print()

# ============================================================================
# 9. SAVE FAIRNESS AUDIT RESULTS
# ============================================================================

fairness_2016.to_csv('fairness_audit_2016_comprehensive.csv')
fairness_201920.to_csv('fairness_audit_201920_comprehensive.csv')
bias_df.to_csv('bias_test_results_chi_squared.csv', index=False)

pd.DataFrame([disparity_2016]).to_csv('disparity_metrics_2016.csv', index=False)
pd.DataFrame([disparity_201920]).to_csv('disparity_metrics_201920.csv', index=False)

print("="*100)
print("✅ FAIRNESS AUDIT COMPLETE")
print("="*100)
print("Files saved:")
print("  • fairness_audit_2016_comprehensive.csv")
print("  • fairness_audit_201920_comprehensive.csv")
print("  • bias_test_results_chi_squared.csv")
print("  • disparity_metrics_2016.csv")
print("  • disparity_metrics_201920.csv")
print("  • fairness_audit_comprehensive.png")
print("="*100)
