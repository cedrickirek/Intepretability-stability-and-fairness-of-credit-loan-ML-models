# ============================================================================
# ENHANCED LIME VISUALIZATIONS FOR INTERVIEW
# Insert this cell after Cell 52 (after existing LIME runs)
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from lime import lime_tabular

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# 1. SELECT REPRESENTATIVE CASES FOR DETAILED LIME ANALYSIS
# ============================================================================

print("="*100)
print("LIME ANALYSIS: REPRESENTATIVE CASES FOR INTERVIEW")
print("="*100)
print()

# Select interesting cases:
# 1. True Positive (correctly predicted default) - HIGH CONFIDENCE
# 2. True Positive (correctly predicted default) - LOW CONFIDENCE
# 3. False Negative (missed default) - for learning
# 4. True Negative (correctly predicted non-default) - HIGH CONFIDENCE
# 5. False Positive (incorrectly predicted default) - for understanding errors

# Find cases
y_true_2016 = y_val_2016
y_pred_2016 = y_pred_lgbm_2016_adapted
y_proba_2016 = y_proba_lgbm_2016

# Case 1: True Positive - High Confidence (predicted probability > 0.90)
tp_high_conf_idx = np.where((y_true_2016 == 1) & (y_pred_2016 == 1) & (y_proba_2016 > 0.90))[0]
if len(tp_high_conf_idx) > 0:
    case1_idx = tp_high_conf_idx[0]
else:
    case1_idx = np.where((y_true_2016 == 1) & (y_pred_2016 == 1))[0][0]

# Case 2: False Negative (missed default)
fn_idx = np.where((y_true_2016 == 1) & (y_pred_2016 == 0))[0]
case2_idx = fn_idx[0] if len(fn_idx) > 0 else case1_idx

# Case 3: True Negative - High Confidence (predicted probability < 0.10)
tn_high_conf_idx = np.where((y_true_2016 == 0) & (y_pred_2016 == 0) & (y_proba_2016 < 0.10))[0]
if len(tn_high_conf_idx) > 0:
    case3_idx = tn_high_conf_idx[0]
else:
    case3_idx = np.where((y_true_2016 == 0) & (y_pred_2016 == 0))[0][0]

# Case 4: False Positive (incorrectly predicted default)
fp_idx = np.where((y_true_2016 == 0) & (y_pred_2016 == 1))[0]
case4_idx = fp_idx[0] if len(fp_idx) > 0 else case1_idx

representative_cases = [
    (case1_idx, "True Positive (High Confidence)", "Correctly predicted DEFAULT"),
    (case2_idx, "False Negative", "MISSED default (should have predicted)"),
    (case3_idx, "True Negative (High Confidence)", "Correctly predicted NON-DEFAULT"),
    (case4_idx, "False Positive", "Incorrectly predicted DEFAULT")
]

# ============================================================================
# 2. RUN LIME ON REPRESENTATIVE CASES
# ============================================================================

# Initialize LIME explainer
X_train_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_train)  # Without LGBM step
X_val_processed = lgbm_pipeline_fitted_2016[:-1].transform(X_val_2016)

feature_names = lgbm_pipeline_fitted_2016[:-1].get_feature_names_out()

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_processed,
    feature_names=feature_names,
    class_names=['No Default', 'Default'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

lime_results = {}

for case_idx, case_name, case_description in representative_cases:
    print(f"\n{'='*80}")
    print(f"Analyzing: {case_name}")
    print(f"Description: {case_description}")
    print(f"Index: {case_idx}")
    print(f"True Label: {'Default' if y_true_2016[case_idx] == 1 else 'No Default'}")
    print(f"Predicted Probability: {y_proba_2016[case_idx]:.4f}")
    print(f"Predicted Label (θ=0.78): {'Default' if y_pred_2016[case_idx] == 1 else 'No Default'}")
    print(f"{'='*80}")

    # Get LIME explanation
    explanation = explainer.explain_instance(
        data_row=X_val_processed[case_idx],
        predict_fn=lgbm_pipeline_fitted_2016.predict_proba,
        num_features=10,
        num_samples=5000
    )

    lime_results[case_name] = {
        'explanation': explanation,
        'index': case_idx,
        'true_label': y_true_2016[case_idx],
        'pred_proba': y_proba_2016[case_idx],
        'pred_label': y_pred_2016[case_idx],
        'description': case_description
    }

# ============================================================================
# 3. CREATE COMPREHENSIVE LIME VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('LIME Explanations: Why Did the Model Make These Predictions?',
             fontsize=18, fontweight='bold', y=0.995)

for idx, (case_name, result) in enumerate(lime_results.items()):
    ax = axes[idx // 2, idx % 2]

    # Get feature contributions
    explanation = result['explanation']
    feature_weights = explanation.as_list()

    # Sort by absolute weight
    feature_weights = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)[:10]

    # Parse feature names and values
    features = []
    contributions = []
    colors = []

    for feat_desc, weight in feature_weights:
        features.append(feat_desc)
        contributions.append(weight)
        colors.append('#2ecc71' if weight > 0 else '#e74c3c')  # Green for positive, red for negative

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, contributions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(label_x, i, f'{val:+.3f}', ha=ha, va='center', fontweight='bold', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Contribution to Prediction (Default)', fontweight='bold', fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Title with case details
    true_label = 'Default' if result['true_label'] == 1 else 'No Default'
    pred_label = 'Default' if result['pred_label'] == 1 else 'No Default'
    pred_proba = result['pred_proba']

    title_color = 'green' if result['true_label'] == result['pred_label'] else 'red'
    ax.set_title(
        f"{case_name}\n"
        f"True: {true_label} | Predicted: {pred_label} (p={pred_proba:.3f})\n"
        f"{result['description']}",
        fontweight='bold', fontsize=11, color=title_color, pad=10
    )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', label='Increases Default Risk'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Decreases Default Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('lime_explanations_representative_cases.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. PRINT DETAILED EXPLANATIONS FOR EACH CASE
# ============================================================================

print("\n" + "="*100)
print("DETAILED LIME EXPLANATIONS")
print("="*100)

for case_name, result in lime_results.items():
    print(f"\n{'─'*100}")
    print(f"📊 {case_name}")
    print(f"{'─'*100}")
    print(f"Description: {result['description']}")
    print(f"True Label: {'Default' if result['true_label'] == 1 else 'No Default'}")
    print(f"Predicted Probability: {result['pred_proba']:.4f}")
    print(f"Predicted Label (θ=0.78): {'Default' if result['pred_label'] == 1 else 'No Default'}")
    print(f"\nTop 10 Feature Contributions:")
    print(f"{'─'*100}")

    feature_weights = result['explanation'].as_list()
    for rank, (feat_desc, weight) in enumerate(feature_weights[:10], 1):
        direction = "➡️  INCREASES" if weight > 0 else "⬅️  DECREASES"
        print(f"{rank:2d}. {feat_desc:60s} | {weight:+.4f} | {direction} default risk")

    print(f"{'─'*100}")

# ============================================================================
# 5. SUMMARY TABLE OF LIME INSIGHTS
# ============================================================================

print("\n" + "="*100)
print("LIME ANALYSIS SUMMARY: KEY INSIGHTS FOR INTERVIEW")
print("="*100)

summary_data = []
for case_name, result in lime_results.items():
    top_positive = max([w for f, w in result['explanation'].as_list() if w > 0], default=0)
    top_negative = min([w for f, w in result['explanation'].as_list() if w < 0], default=0)

    summary_data.append({
        'Case': case_name,
        'True Label': 'Default' if result['true_label'] == 1 else 'No Default',
        'Pred Proba': f"{result['pred_proba']:.4f}",
        'Pred Label': 'Default' if result['pred_label'] == 1 else 'No Default',
        'Correct?': '✅' if result['true_label'] == result['pred_label'] else '❌',
        'Top Pos Contrib': f"{top_positive:+.4f}",
        'Top Neg Contrib': f"{top_negative:+.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*100)
print("KEY TAKEAWAYS:")
print("="*100)
print("✅ LIME provides LOCAL explanations: Why THIS specific loan was flagged")
print("✅ Features can have different impacts for different individuals")
print("✅ Model is transparent: We can see exactly what drives each decision")
print("✅ Error cases (FN, FP) show where model struggles - actionable for improvement")
print("="*100)

# ============================================================================
# 6. SAVE LIME EXPLANATIONS AS HTML (BONUS)
# ============================================================================

for case_name, result in lime_results.items():
    safe_name = case_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    result['explanation'].save_to_file(f'lime_explanation_{safe_name}.html')
    print(f"✅ Saved LIME explanation: lime_explanation_{safe_name}.html")

print("\n✅ All LIME visualizations and explanations saved successfully!")
