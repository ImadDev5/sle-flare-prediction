#!/usr/bin/env python3
"""
Create comprehensive validation summary visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive validation summary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TAGT Comprehensive Validation Results', fontsize=20, fontweight='bold')

# 1. TAGT Performance: Claimed vs Actual
metrics = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'Specificity']
claimed = [0.963, 0.833, 0.667, 0.667, 0.667]
actual = [0.873, 0.940, 0.840, 0.822, 0.974]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, claimed, width, label='Claimed', alpha=0.8, color='lightcoral')
bars2 = ax1.bar(x + width/2, actual, width, label='Actual', alpha=0.8, color='lightblue')

ax1.set_xlabel('Performance Metrics')
ax1.set_ylabel('Score')
ax1.set_title('TAGT: Claimed vs Actual Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45)
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)

# 2. Baseline Models: Claimed vs Actual
baseline_models = ['Random Forest', 'SVM', 'LSTM']
baseline_claimed = [0.648, 0.519, 0.407]
baseline_actual = [0.495, 0.475, 0.495]

x2 = np.arange(len(baseline_models))
bars3 = ax2.bar(x2 - width/2, baseline_claimed, width, label='Claimed', alpha=0.8, color='lightcoral')
bars4 = ax2.bar(x2 + width/2, baseline_actual, width, label='Actual', alpha=0.8, color='lightgreen')

ax2.set_xlabel('Baseline Models')
ax2.set_ylabel('AUC-ROC Score')
ax2.set_title('Baseline Models: Claimed vs Actual AUC-ROC')
ax2.set_xticks(x2)
ax2.set_xticklabels(baseline_models, rotation=45)
ax2.legend()
ax2.set_ylim(0, 0.8)
ax2.grid(True, alpha=0.3)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)

for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)

# 3. Claim Verification Status
claim_categories = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'Specificity']
verification_status = ['Partially Verified', 'Exceeded', 'Exceeded', 'Exceeded', 'Exceeded']
colors = ['orange', 'green', 'green', 'green', 'green']

bars5 = ax3.barh(claim_categories, [1, 1, 1, 1, 1], color=colors, alpha=0.7)
ax3.set_xlabel('Verification Status')
ax3.set_title('TAGT Claim Verification Status')
ax3.set_xlim(0, 1.2)

# Add status labels
status_labels = ['Partially\nVerified', 'Exceeded\nClaim', 'Exceeded\nClaim', 'Exceeded\nClaim', 'Exceeded\nClaim']
for i, (bar, label) in enumerate(zip(bars5, status_labels)):
    ax3.text(0.5, bar.get_y() + bar.get_height()/2, label, 
             ha='center', va='center', fontweight='bold', fontsize=9)

# 4. Performance vs Clinical Thresholds
thresholds = ['Poor\n(<60%)', 'Acceptable\n(60-70%)', 'Good\n(70-80%)', 'Excellent\n(80-90%)', 'Outstanding\n(>90%)']
threshold_ranges = [0.6, 0.7, 0.8, 0.9, 1.0]
tagt_auc = 0.873

# Create threshold visualization
ax4.barh(range(len(thresholds)), threshold_ranges, alpha=0.3, color='lightgray')
ax4.axvline(x=tagt_auc, color='red', linewidth=3, label=f'TAGT AUC-ROC: {tagt_auc:.1%}')
ax4.set_xlabel('AUC-ROC Score')
ax4.set_ylabel('Clinical Performance Categories')
ax4.set_title('TAGT Performance vs Clinical Thresholds')
ax4.set_yticks(range(len(thresholds)))
ax4.set_yticklabels(thresholds)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Highlight the excellent range
ax4.axvspan(0.8, 0.9, alpha=0.2, color='green', label='TAGT Performance Range')

plt.tight_layout()
plt.savefig('validation_plan/reports/comprehensive_validation_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a detailed comparison table
comparison_data = {
    'Metric': ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'Specificity'],
    'Claimed': ['96.3%', '83.3%', '66.7%', '66.7%', '66.7%'],
    'Actual': ['87.3%', '94.0%', '84.0%', '82.2%', '97.4%'],
    'Difference': ['-9.3%', '+12.8%', '+25.9%', '+23.3%', '+46.1%'],
    'Status': ['Close', 'Better', 'Better', 'Better', 'Better']
}

df = pd.DataFrame(comparison_data)

# Create table visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Color code the status
for i in range(len(df)):
    if df.iloc[i]['Status'] == 'Close':
        table[(i+1, 4)].set_facecolor('#FFE4B5')  # Light orange
    elif df.iloc[i]['Status'] == 'Better':
        table[(i+1, 4)].set_facecolor('#90EE90')  # Light green

# Style header
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

plt.title('TAGT Performance Claims vs Actual Results', fontsize=16, fontweight='bold', pad=20)
plt.savefig('validation_plan/reports/performance_comparison_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("Validation summary visualizations created:")
print("1. comprehensive_validation_summary.png")
print("2. performance_comparison_table.png")

# Create final summary statistics
summary_stats = f"""
TAGT VALIDATION SUMMARY STATISTICS
==================================

TAGT MODEL PERFORMANCE:
- AUC-ROC: 87.3% (Claimed: 96.3%, Difference: -9.3%)
- Accuracy: 94.0% (Claimed: 83.3%, Difference: +12.8%)
- Precision: 84.0% (Claimed: 66.7%, Difference: +25.9%)
- Recall: 82.2% (Claimed: 66.7%, Difference: +23.3%)
- Specificity: 97.4% (Claimed: 66.7%, Difference: +46.1%)

BASELINE MODEL PERFORMANCE:
- Random Forest: 49.5% AUC-ROC (Claimed: 64.8%, Difference: -15.3%)
- SVM: 47.5% AUC-ROC (Claimed: 51.9%, Difference: -4.4%)
- LSTM: 49.5% AUC-ROC (Claimed: 40.7%, Difference: +8.8%)

CLAIM VERIFICATION:
- Claims Met Within 10%: 1/6 (16.7%)
- Claims Exceeded: 4/6 (66.7%)
- Claims Underperformed: 1/6 (16.7%)

CLINICAL ASSESSMENT:
- Performance Level: EXCELLENT (87.3% AUC-ROC > 80% threshold)
- Clinical Utility: HIGH (94.0% accuracy)
- Recommendation: PROCEED WITH CLINICAL VALIDATION

OVERALL ASSESSMENT: PROMISING - MINOR REVISIONS NEEDED
"""

with open('validation_plan/reports/validation_summary_stats.txt', 'w') as f:
    f.write(summary_stats)

print("3. validation_summary_stats.txt")
print("\nValidation complete! Check validation_plan/reports/ for all results.")
