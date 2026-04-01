"""
================================================================================
SPEND INTENSITY PREDICTION MODEL — Digital Product Spend per Consumer Unit
================================================================================
Business Goal: Predict monthly digital-product spend per CU to identify
               high-value markets/households and create opportunity scores.

Model:         Gradient Boosting Regressor (scikit-learn)
               → Drop-in XGBoost/LightGBM upgrade included (commented)
Target:        digital_spend_per_CU (monthly aggregation)
Explanations:  Permutation Importance + Partial Dependence
               → SHAP upgrade included (commented)

Author:        ML Development Assistant
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── Color palette ──
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'accent': '#F59E0B',
    'positive': '#10B981',
    'negative': '#EF4444',
    'bg': '#F8FAFC',
    'text': '#1E293B',
    'grid': '#E2E8F0',
}

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("=" * 70)

# ── IMPORTANT: Update this path to your actual CSV file ──
DATA_PATH = "./cleaned_dataset.xlsx"

# Try common file names if default doesn't exist
possible_paths = [
    DATA_PATH,
    "/mnt/user-data/uploads/merged_data.csv",
    "/mnt/user-data/uploads/data.csv",
    "/mnt/user-data/uploads/ce_data.csv",
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_excel(path)
        print(f"✓ Loaded data from: {path}")
        break

if df is None:
    # Generate synthetic data matching the user's schema for demonstration
    print("⚠  Dataset file not found. Generating synthetic data matching your schema...")
    print("   → Update DATA_PATH variable with your actual file path.")

    np.random.seed(42)
    n_cus = 3000
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    years = [2020, 2021, 2022, 2023]
    uccs = [270102, 310243, 310244, 690114, 690210, 690220, 690310, 690320, 690330]
    product_desc_map = {
        270102: ('Cellular phone service', 'Online Service'),
        310243: ('Computers and peripherals', 'Electronics'),
        310244: ('Portable memory devices', 'Electronics'),
        690114: ('Computer information services (internet)', 'Online Service'),
        690210: ('Streaming video services', 'Online Service'),
        690220: ('Streaming audio services', 'Online Service'),
        690310: ('Video game software', 'Software'),
        690320: ('Computer software', 'Software'),
        690330: ('Video game hardware/accessories', 'Electronics'),
    }
    regions = ['West', 'Northeast', 'South', 'Midwest']
    states_by_region = {
        'West': ['California', 'Washington', 'Colorado', 'Arizona', 'Alaska', 'Hawaii'],
        'Northeast': ['New York', 'Pennsylvania', 'New Jersey', 'Massachusetts'],
        'South': ['Florida', 'Georgia', 'Texas', 'Virginia', 'Maryland', 'District of Columbia'],
        'Midwest': ['Illinois', 'Michigan', 'Minnesota', 'Missouri'],
    }
    psu_by_state = {
        'California': ['Los Angeles-Long Beach-Anaheim, CA', 'San Francisco-Oakland-Hayward, CA',
                       'Riverside-San Bernardino-Ontario, CA', 'San Diego-Carlsbad, CA'],
        'New York': ['New York-Newark-Jersey City, NY-NJ-PA'],
        'Florida': ['Miami-Fort Lauderdale-West Palm Beach, FL', 'Tampa-St. Petersburg-Clearwater, FL'],
        'Texas': ['Houston-The Woodlands-Sugar Land, TX', 'Dallas-Fort Worth-Arlington, TX'],
        'Illinois': ['Chicago-Naperville-Elgin, IL-IN-WI'],
        'Washington': ['Seattle-Tacoma-Bellevue, WA'],
        'Colorado': ['Denver-Aurora-Lakewood, CO'],
        'Maryland': ['Baltimore-Columbia-Towson, MD'],
        'Georgia': ['Atlanta-Sandy Springs-Roswell, GA'],
        'Pennsylvania': ['Philadelphia-Camden-Wilmington, PA-NJ-DE-MD'],
        'Michigan': ['Detroit-Warren-Dearborn, MI'],
        'Alaska': ['Anchorage, AK'],
        'Hawaii': ['Honolulu, HI'],
        'Minnesota': ['Minneapolis-St. Paul-Bloomington, MN-WI'],
    }
    division_map = {
        'California': 'Pacific', 'Washington': 'Pacific', 'Alaska': 'Pacific', 'Hawaii': 'Pacific',
        'Colorado': 'Mountain', 'Arizona': 'Mountain',
        'New York': 'Middle Atlantic', 'Pennsylvania': 'Middle Atlantic', 'New Jersey': 'Middle Atlantic',
        'Massachusetts': 'New England',
        'Florida': 'South Central', 'Georgia': 'South Central', 'Virginia': 'South Central',
        'Maryland': 'South Central', 'District of Columbia': 'South Central',
        'Texas': 'West South Central',
        'Illinois': 'East North Central', 'Michigan': 'East North Central',
        'Minnesota': 'West North Central', 'Missouri': 'West North Central',
    }
    pop_sizes = ['100-500 Thousands', '0.5-1.0 Million', '1-5 Million', '5+ Million']

    rows = []
    idx = 0
    for cu_i in range(n_cus):
        newid = 4200000 + cu_i * 10
        region = np.random.choice(regions, p=[0.25, 0.2, 0.35, 0.2])
        state = np.random.choice(states_by_region[region])
        psu_list = psu_by_state.get(state, [f"{state} Metro"])
        psu = np.random.choice(psu_list)
        division = division_map.get(state, 'Other')
        fam_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.3, 0.2, 0.15, 0.07, 0.03])
        income = int(np.clip(np.random.lognormal(11, 0.8), 5000, 1200000))
        salary = int(income * np.random.uniform(0.6, 1.0))
        n_earners = min(fam_size, np.random.choice([0, 1, 2, 3], p=[0.05, 0.45, 0.4, 0.1]))
        sex = np.random.choice(['Male', 'Female'])
        weight = np.random.uniform(5000, 80000)
        pop_size = np.random.choice(pop_sizes, p=[0.15, 0.2, 0.35, 0.3])

        # Each CU active for 3 consecutive months (CE interview structure)
        start_year = np.random.choice(years)
        start_month_idx = np.random.randint(0, 10)  # 0-9 so 3 months fit
        interview_month_idx = min(start_month_idx + 3, 11)

        # Number of UCCs this CU reports (1-4)
        n_uccs = np.random.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.1])
        cu_uccs = np.random.choice(uccs, size=n_uccs, replace=False)

        for ucc in cu_uccs:
            desc, cat = product_desc_map[ucc]
            # Cost varies by income level and category
            base_cost = np.random.lognormal(3.8, 0.6)
            income_factor = 1 + 0.3 * np.log1p(income) / np.log1p(150000)
            cost = round(np.clip(base_cost * income_factor, 1, 600), 2)

            for m_offset in range(3):
                m_idx = start_month_idx + m_offset
                rows.append({
                    'index': idx,
                    'newid': newid,
                    'reference_month': months[m_idx],
                    'reference_year': start_year,
                    'ucc': ucc,
                    'cost': cost,
                    'fam_size': fam_size,
                    'family_income_before_tax_last_12_month': income,
                    'calibration_weight': round(weight, 3),
                    'number_of_earners': n_earners,
                    'interview_month': months[interview_month_idx],
                    'interview_year': start_year,
                    'region': region,
                    'sex_ref': sex,
                    'state': state,
                    'psu': psu,
                    'division': division,
                    'total_salary_income_before_deduction': salary,
                    'product_description': desc,
                    'product_category': cat,
                    'population_size': pop_size,
                })
                idx += 1

    df = pd.DataFrame(rows)
    print(f"   Generated {len(df):,} rows for {n_cus:,} CUs")

print(f"\n  Shape: {df.shape}")
print(f"  CUs:   {df['newid'].nunique():,}")
print(f"  Years: {sorted(df['reference_year'].unique())}")
print(f"  Product categories: {df['product_category'].unique().tolist()}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 70)

# ── 2.1 Month name → numeric ──
month_order = {m: i + 1 for i, m in enumerate([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])}
df['month_num'] = df['reference_month'].map(month_order)

# ── 2.2 Create time key for sorting ──
df['year_month'] = df['reference_year'] * 100 + df['month_num']

# ── 2.3 Spend-mix features at CU-month level BEFORE aggregation ──
# Category spend per CU-month
cat_spend = df.groupby(['newid', 'year_month', 'product_category'])['cost'].sum().unstack(fill_value=0)
cat_spend.columns = [f'spend_{c.lower().replace(" ", "_")}' for c in cat_spend.columns]

# Total spend per CU-month (for ratios)
cat_spend['total_spend'] = cat_spend.sum(axis=1)

# Spend ratios
for col in cat_spend.columns:
    if col.startswith('spend_') and col != 'total_spend':
        ratio_col = col.replace('spend_', 'ratio_')
        cat_spend[ratio_col] = cat_spend[col] / cat_spend['total_spend'].clip(lower=0.01)

# Number of distinct UCCs per CU-month
ucc_count = df.groupby(['newid', 'year_month'])['ucc'].nunique().rename('n_distinct_uccs')

# ── 2.4 Aggregate to CU-month level ──
# Demographics (constant per CU) — take first
demo_cols = [
    'fam_size', 'family_income_before_tax_last_12_month',
    'calibration_weight', 'number_of_earners', 'region', 'sex_ref',
    'state', 'psu', 'division', 'total_salary_income_before_deduction',
    'population_size', 'reference_year', 'month_num'
]

agg_demo = df.groupby(['newid', 'year_month'])[demo_cols].first()

# Target: total digital spend per CU per month
target = df.groupby(['newid', 'year_month'])['cost'].sum().rename('digital_spend')

# Merge everything
cu_monthly = agg_demo.join(target).join(cat_spend).join(ucc_count)
cu_monthly = cu_monthly.reset_index()

print(f"✓ Aggregated to CU-month level: {cu_monthly.shape[0]:,} rows")
print(f"  Target (digital_spend) stats:")
print(f"    Mean:   ${cu_monthly['digital_spend'].mean():.2f}")
print(f"    Median: ${cu_monthly['digital_spend'].median():.2f}")
print(f"    Std:    ${cu_monthly['digital_spend'].std():.2f}")
print(f"    Min:    ${cu_monthly['digital_spend'].min():.2f}")
print(f"    Max:    ${cu_monthly['digital_spend'].max():.2f}")

# ── 2.5 Lag features (per CU, sorted by time) ──
cu_monthly = cu_monthly.sort_values(['newid', 'year_month'])

cu_monthly['spend_lag_1m'] = cu_monthly.groupby('newid')['digital_spend'].shift(1)
cu_monthly['spend_lag_2m'] = cu_monthly.groupby('newid')['digital_spend'].shift(2)
cu_monthly['spend_rolling_3m'] = (
    cu_monthly.groupby('newid')['digital_spend']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# ── 2.6 Income features ──
cu_monthly['log_income'] = np.log1p(
    cu_monthly['family_income_before_tax_last_12_month'].clip(lower=0)
)
cu_monthly['log_salary'] = np.log1p(
    cu_monthly['total_salary_income_before_deduction'].clip(lower=0)
)
cu_monthly['income_per_member'] = (
        cu_monthly['family_income_before_tax_last_12_month'] / cu_monthly['fam_size']
)
cu_monthly['salary_to_income_ratio'] = (
        cu_monthly['total_salary_income_before_deduction'] /
        cu_monthly['family_income_before_tax_last_12_month'].clip(lower=1)
).clip(0, 2)

# ── 2.7 Cyclical time encoding ──
cu_monthly['month_sin'] = np.sin(2 * np.pi * cu_monthly['month_num'] / 12)
cu_monthly['month_cos'] = np.cos(2 * np.pi * cu_monthly['month_num'] / 12)

# ── 2.8 Encode categoricals ──
# Ordinal encoding for population_size
pop_order = ['100-500 Thousands', '0.5-1.0 Million', '1-5 Million', '5+ Million']
pop_map = {v: i for i, v in enumerate(pop_order)}
cu_monthly['pop_size_ord'] = cu_monthly['population_size'].map(pop_map).fillna(0).astype(int)

# Label encoding for low-cardinality categoricals
label_encoders = {}
cat_encode_cols = ['region', 'sex_ref', 'division', 'state', 'psu']
for col in cat_encode_cols:
    le = LabelEncoder()
    cu_monthly[f'{col}_enc'] = le.fit_transform(cu_monthly[col].astype(str))
    label_encoders[col] = le

print(f"\n✓ Feature engineering complete")
print(
    f"  Total features created: {len([c for c in cu_monthly.columns if c not in ['newid', 'year_month', 'digital_spend']])}")

# ── 2.9 Handle NaNs from lag features ──
# Rows without lag data (first month per CU) — fill with 0 or median
lag_cols = ['spend_lag_1m', 'spend_lag_2m', 'spend_rolling_3m']
for col in lag_cols:
    cu_monthly[col] = cu_monthly[col].fillna(0)

# Fill any remaining NaNs in spend columns
spend_cols = [c for c in cu_monthly.columns if c.startswith('spend_') or c.startswith('ratio_')]
for col in spend_cols:
    cu_monthly[col] = cu_monthly[col].fillna(0)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN/TEST SPLIT & MODELING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: MODELING")
print("=" * 70)

# ── 3.1 Define feature set ──
feature_cols = [
    # Demographics
    'fam_size', 'number_of_earners', 'log_income', 'log_salary',
    'income_per_member', 'salary_to_income_ratio',
    # Geography (encoded)
    'region_enc', 'division_enc', 'state_enc', 'psu_enc', 'pop_size_ord',
    # Gender
    'sex_ref_enc',
    # Time
    'reference_year', 'month_num', 'month_sin', 'month_cos',
    # Spend mix
    'n_distinct_uccs',
    # Lag features
    'spend_lag_1m', 'spend_lag_2m', 'spend_rolling_3m',
]

# Add category spend columns dynamically
cat_feature_cols = [c for c in cu_monthly.columns
                    if (c.startswith('spend_') or c.startswith('ratio_'))
                    and c != 'total_spend']
feature_cols.extend(cat_feature_cols)

# Remove duplicates
feature_cols = list(dict.fromkeys(feature_cols))

# Ensure all feature columns exist
feature_cols = [c for c in feature_cols if c in cu_monthly.columns]

X = cu_monthly[feature_cols].copy()
y = cu_monthly['digital_spend'].copy()
weights = cu_monthly['calibration_weight'].copy()

print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {len(X):,}")

# ── 3.2 Time-based train/test split ──
# Use last year (2023) as test, rest as train
split_year_month = 202301  # Jan 2023
train_mask = cu_monthly['year_month'] < split_year_month
test_mask = cu_monthly['year_month'] >= split_year_month

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
w_train, w_test = weights[train_mask], weights[test_mask]

print(f"\n  Train set: {len(X_train):,} rows ({train_mask.sum() / len(X) * 100:.1f}%)")
print(f"  Test set:  {len(X_test):,} rows ({test_mask.sum() / len(X) * 100:.1f}%)")
print(f"  Split:     < {split_year_month} (train) | >= {split_year_month} (test)")

# ── 3.3 Baseline model: Ridge Regression ──
print("\n── Baseline: Ridge Regression ──")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train, sample_weight=w_train)
y_pred_ridge = ridge.predict(X_test)

ridge_metrics = {
    'MAE': mean_absolute_error(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'R2': r2_score(y_test, y_pred_ridge),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_ridge) * 100,
}
for k, v in ridge_metrics.items():
    print(f"  {k}: {v:.4f}" if k == 'R2' else f"  {k}: ${v:.2f}" if k in ['MAE', 'RMSE'] else f"  {k}: {v:.2f}%")

# ── 3.4 Primary model: Gradient Boosting ──
print("\n── Primary: Gradient Boosting Regressor ──")
gbr = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    min_samples_split=30,
    max_features='sqrt',
    random_state=42,
    validation_fraction=0.15,
    n_iter_no_change=30,
    tol=1e-4,
)

print("  Training (with early stopping)...")
gbr.fit(X_train, y_train, sample_weight=w_train)
print(f"  → Stopped at {gbr.n_estimators_} estimators (of {gbr.n_estimators} max)")

y_pred_gbr = gbr.predict(X_test)

gbr_metrics = {
    'MAE': mean_absolute_error(y_test, y_pred_gbr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
    'R2': r2_score(y_test, y_pred_gbr),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_gbr) * 100,
}
for k, v in gbr_metrics.items():
    print(f"  {k}: {v:.4f}" if k == 'R2' else f"  {k}: ${v:.2f}" if k in ['MAE', 'RMSE'] else f"  {k}: {v:.2f}%")

# ── 3.5 Cross-validation (time-series aware) ──
print("\n── Time-Series Cross-Validation (5 folds) ──")
tscv = TimeSeriesSplit(n_splits=5)
# Sort by time for proper CV
sort_idx = cu_monthly[train_mask].sort_values('year_month').index
X_train_sorted = X_train.loc[sort_idx]
y_train_sorted = y_train.loc[sort_idx]

cv_scores = cross_val_score(
    GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    ),
    X_train_sorted, y_train_sorted,
    cv=tscv, scoring='r2', n_jobs=-1
)
print(f"  CV R² scores: {cv_scores.round(4)}")
print(f"  CV R² mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: FEATURE IMPORTANCE & EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: FEATURE IMPORTANCE & EXPLANATIONS")
print("=" * 70)

# ── 4.1 Built-in feature importance (impurity-based) ──
feat_imp = pd.Series(gbr.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features (impurity-based):")
for feat, imp in feat_imp.head(15).items():
    print(f"  {imp:.4f}  {feat}")

# ── 4.2 Permutation importance (more reliable) ──
print("\n  Computing permutation importance (may take a moment)...")
perm_imp = permutation_importance(
    gbr, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1
)
perm_imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std,
}).sort_values('importance_mean', ascending=False)

print("\nTop 15 features (permutation importance):")
for _, row in perm_imp_df.head(15).iterrows():
    print(f"  {row['importance_mean']:.4f} ± {row['importance_std']:.4f}  {row['feature']}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 5: GENERATING VISUALIZATIONS")
print("=" * 70)


# ── Helper for consistent styling ──
def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'], pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=11, color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')


# ── FIGURE 1: Model Comparison ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Model Performance Report', fontsize=18, fontweight='bold',
             color=COLORS['text'], y=1.02)

# 1a. Metrics comparison bar chart
metrics_df = pd.DataFrame({
    'Ridge (Baseline)': ridge_metrics,
    'Gradient Boosting': gbr_metrics,
}).T
ax = axes[0]
x = np.arange(len(metrics_df.columns))
w = 0.35
bars1 = ax.bar(x - w / 2, metrics_df.iloc[0], w, color=COLORS['grid'], label='Ridge', edgecolor='white')
bars2 = ax.bar(x + w / 2, metrics_df.iloc[1], w, color=COLORS['primary'], label='GBR', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.columns, fontsize=10)
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Metrics Comparison', ylabel='Value')

# 1b. Actual vs Predicted scatter
ax = axes[1]
sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
ax.scatter(y_test.iloc[sample_idx], y_pred_gbr[sample_idx],
           alpha=0.3, s=15, c=COLORS['primary'], edgecolors='none')
max_val = max(y_test.max(), y_pred_gbr.max())
ax.plot([0, max_val], [0, max_val], '--', color=COLORS['negative'], lw=2, label='Perfect prediction')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Actual vs Predicted (GBR)', 'Actual Spend ($)', 'Predicted Spend ($)')

# 1c. Residual distribution
ax = axes[2]
residuals = y_test - y_pred_gbr
ax.hist(residuals, bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='white')
ax.axvline(0, color=COLORS['negative'], linestyle='--', lw=2)
ax.axvline(residuals.mean(), color=COLORS['accent'], linestyle='-', lw=2, label=f'Mean: ${residuals.mean():.2f}')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Residual Distribution', 'Residual ($)', 'Count')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/01_model_performance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: 01_model_performance.png")

# ── FIGURE 2: Feature Importance (dual view) ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Feature Importance Analysis', fontsize=18, fontweight='bold',
             color=COLORS['text'], y=1.02)

# 2a. Impurity-based
top_n = 15
top_feats = feat_imp.head(top_n)
ax = axes[0]
bars = ax.barh(range(top_n), top_feats.values[::-1], color=COLORS['primary'],
               edgecolor='white', height=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats.index[::-1], fontsize=10)
for i, (val, bar) in enumerate(zip(top_feats.values[::-1], bars)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9, color=COLORS['text'])
style_ax(ax, 'Impurity-Based Importance', 'Importance Score')

# 2b. Permutation-based
top_perm = perm_imp_df.head(top_n)
ax = axes[1]
bars = ax.barh(range(top_n), top_perm['importance_mean'].values[::-1],
               xerr=top_perm['importance_std'].values[::-1],
               color=COLORS['secondary'], edgecolor='white', height=0.7,
               capsize=3, error_kw={'color': COLORS['text'], 'alpha': 0.5})
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_perm['feature'].values[::-1], fontsize=10)
style_ax(ax, 'Permutation Importance (± std)', 'Importance Score')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/02_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: 02_feature_importance.png")

# ── FIGURE 3: Partial Dependence Plots ──
# Top 4 numeric features by importance
top_numeric_feats = [f for f in feat_imp.head(10).index
                     if f not in [c + '_enc' for c in cat_encode_cols] and 'pop_size' not in f][:4]

if len(top_numeric_feats) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Partial Dependence Plots (Top Features)', fontsize=18,
                 fontweight='bold', color=COLORS['text'], y=1.02)

    for idx, feat in enumerate(top_numeric_feats[:4]):
        ax = axes.flat[idx]
        feat_idx = feature_cols.index(feat)
        display = PartialDependenceDisplay.from_estimator(
            gbr, X_train, [feat_idx], feature_names=feature_cols,
            ax=ax, line_kw={'color': COLORS['primary'], 'lw': 2.5},
            pd_line_kw={'color': COLORS['primary']},
        )
        style_ax(ax, feat, feat, 'Partial Dependence')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(top_numeric_feats), 4):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/03_partial_dependence.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: 03_partial_dependence.png")

# ── FIGURE 4: Prediction Analysis by Segment ──
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Prediction Analysis by Segment', fontsize=18, fontweight='bold',
             color=COLORS['text'], y=1.02)

test_df = cu_monthly[test_mask].copy()
test_df['predicted'] = y_pred_gbr
test_df['residual'] = test_df['digital_spend'] - test_df['predicted']

# 4a. By region
ax = axes[0, 0]
region_perf = test_df.groupby('region').agg(
    actual_mean=('digital_spend', 'mean'),
    pred_mean=('predicted', 'mean'),
).reset_index()
x = np.arange(len(region_perf))
ax.bar(x - 0.2, region_perf['actual_mean'], 0.35, color=COLORS['primary'], label='Actual', edgecolor='white')
ax.bar(x + 0.2, region_perf['pred_mean'], 0.35, color=COLORS['accent'], label='Predicted', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(region_perf['region'], fontsize=10)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Mean Spend by Region', ylabel='Monthly Spend ($)')

# 4b. By family size
ax = axes[0, 1]
fam_perf = test_df.groupby('fam_size').agg(
    actual_mean=('digital_spend', 'mean'),
    pred_mean=('predicted', 'mean'),
    count=('digital_spend', 'size'),
).reset_index()
fam_perf = fam_perf[fam_perf['count'] >= 10]  # filter small groups
ax.plot(fam_perf['fam_size'], fam_perf['actual_mean'], 'o-',
        color=COLORS['primary'], lw=2.5, markersize=8, label='Actual')
ax.plot(fam_perf['fam_size'], fam_perf['pred_mean'], 's--',
        color=COLORS['accent'], lw=2.5, markersize=8, label='Predicted')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Mean Spend by Family Size', 'Family Size', 'Monthly Spend ($)')

# 4c. By income quartile
ax = axes[1, 0]
test_df['income_quartile'] = pd.qcut(
    test_df['family_income_before_tax_last_12_month'], 4,
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
)
inc_perf = test_df.groupby('income_quartile', observed=True).agg(
    actual_mean=('digital_spend', 'mean'),
    pred_mean=('predicted', 'mean'),
).reset_index()
x = np.arange(len(inc_perf))
ax.bar(x - 0.2, inc_perf['actual_mean'], 0.35, color=COLORS['primary'], label='Actual', edgecolor='white')
ax.bar(x + 0.2, inc_perf['pred_mean'], 0.35, color=COLORS['accent'], label='Predicted', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(inc_perf['income_quartile'], fontsize=10)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Mean Spend by Income Quartile', ylabel='Monthly Spend ($)')

# 4d. Monthly trend
ax = axes[1, 1]
monthly_perf = test_df.groupby('month_num').agg(
    actual_mean=('digital_spend', 'mean'),
    pred_mean=('predicted', 'mean'),
).reset_index()
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.plot(monthly_perf['month_num'], monthly_perf['actual_mean'], 'o-',
        color=COLORS['primary'], lw=2.5, markersize=8, label='Actual')
ax.plot(monthly_perf['month_num'], monthly_perf['pred_mean'], 's--',
        color=COLORS['accent'], lw=2.5, markersize=8, label='Predicted')
ax.set_xticks(monthly_perf['month_num'])
ax.set_xticklabels([month_labels[m - 1] for m in monthly_perf['month_num']], fontsize=9)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Monthly Spend Trend (2023)', 'Month', 'Monthly Spend ($)')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/04_segment_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: 04_segment_analysis.png")

# ── FIGURE 5: Learning curve & CV results ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Model Diagnostics', fontsize=18, fontweight='bold',
             color=COLORS['text'], y=1.02)

# 5a. Training deviance (loss curve)
ax = axes[0]
test_score = np.zeros(gbr.n_estimators_, dtype=np.float64)
for i, y_pred_stage in enumerate(gbr.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred_stage)
train_score = gbr.train_score_[:gbr.n_estimators_]

ax.plot(np.arange(1, gbr.n_estimators_ + 1), train_score,
        color=COLORS['primary'], lw=2, label='Train MSE', alpha=0.8)
ax.plot(np.arange(1, gbr.n_estimators_ + 1), test_score,
        color=COLORS['negative'], lw=2, label='Test MSE', alpha=0.8)
ax.axvline(gbr.n_estimators_, color=COLORS['accent'], linestyle='--', lw=1.5,
           label=f'Early stop: {gbr.n_estimators_}')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Training & Test Loss Curve', 'Boosting Iterations', 'MSE')

# 5b. CV fold scores
ax = axes[1]
fold_nums = np.arange(1, len(cv_scores) + 1)
colors = [COLORS['positive'] if s > cv_scores.mean() else COLORS['negative'] for s in cv_scores]
bars = ax.bar(fold_nums, cv_scores, color=colors, edgecolor='white', width=0.6)
ax.axhline(cv_scores.mean(), color=COLORS['primary'], linestyle='--', lw=2,
           label=f'Mean R² = {cv_scores.mean():.4f}')
ax.fill_between([0.5, len(cv_scores) + 0.5],
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color=COLORS['primary'], label=f'± {cv_scores.std():.4f}')
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(fold_nums)
ax.set_xticklabels([f'Fold {i}' for i in fold_nums])
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['grid'])
style_ax(ax, 'Time-Series CV: R² per Fold', ylabel='R² Score')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/05_model_diagnostics.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: 05_model_diagnostics.png")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: MODEL PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 6: MODEL PERFORMANCE REPORT")
print("=" * 70)

report = f"""
{'=' * 70}
 SPEND INTENSITY PREDICTION — MODEL PERFORMANCE REPORT
 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 70}

1. PROBLEM DEFINITION
   Target:       Monthly digital spend per Consumer Unit ($)
   Aggregation:  CU × Month level
   Time span:    {sorted(df['reference_year'].unique())}
   Total CUs:    {df['newid'].nunique():,}
   Total rows:   {len(cu_monthly):,} (after aggregation)

2. DATA SPLIT (Time-based)
   Training:     < Jan 2023  ({train_mask.sum():,} rows)
   Testing:      ≥ Jan 2023  ({test_mask.sum():,} rows)

3. TARGET STATISTICS
   Mean:   ${cu_monthly['digital_spend'].mean():.2f}
   Median: ${cu_monthly['digital_spend'].median():.2f}
   Std:    ${cu_monthly['digital_spend'].std():.2f}
   Min:    ${cu_monthly['digital_spend'].min():.2f}
   Max:    ${cu_monthly['digital_spend'].max():.2f}

4. MODEL COMPARISON
   ┌──────────────────────┬───────────┬───────────────────┐
   │ Metric               │ Ridge     │ Gradient Boosting │
   ├──────────────────────┼───────────┼───────────────────┤
   │ MAE ($)              │ {ridge_metrics['MAE']:>9.2f} │ {gbr_metrics['MAE']:>17.2f} │
   │ RMSE ($)             │ {ridge_metrics['RMSE']:>9.2f} │ {gbr_metrics['RMSE']:>17.2f} │
   │ R²                   │ {ridge_metrics['R2']:>9.4f} │ {gbr_metrics['R2']:>17.4f} │
   │ MAPE (%)             │ {ridge_metrics['MAPE']:>9.2f} │ {gbr_metrics['MAPE']:>17.2f} │
   └──────────────────────┴───────────┴───────────────────┘

5. CROSS-VALIDATION (5-fold Time Series)
   Fold scores: {', '.join(f'{s:.4f}' for s in cv_scores)}
   Mean R²:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

6. TOP 10 FEATURES (Permutation Importance)
"""

for i, (_, row) in enumerate(perm_imp_df.head(10).iterrows()):
    report += f"   {i + 1:>2}. {row['feature']:<35} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}\n"

report += f"""
7. MODEL CONFIGURATION
   Algorithm:        GradientBoostingRegressor (scikit-learn)
   n_estimators:     {gbr.n_estimators} (stopped at {gbr.n_estimators_})
   max_depth:        {gbr.max_depth}
   learning_rate:    {gbr.learning_rate}
   subsample:        {gbr.subsample}
   min_samples_leaf: {gbr.min_samples_leaf}

8. FEATURE SET ({len(feature_cols)} features)
   Demographics:     fam_size, number_of_earners, log_income, log_salary,
                     income_per_member, salary_to_income_ratio
   Geography:        region, division, state, psu, population_size (encoded)
   Time:             reference_year, month_num, month_sin, month_cos
   Spend patterns:   n_distinct_uccs, category spend amounts & ratios
   Lag features:     1-month lag, 2-month lag, 3-month rolling average

9. OUTPUT FILES
   01_model_performance.png   — Metrics, actual vs predicted, residuals
   02_feature_importance.png  — Impurity & permutation importance
   03_partial_dependence.png  — PDP for top numeric features
   04_segment_analysis.png    — Predictions by region/income/family/month
   05_model_diagnostics.png   — Learning curve & CV fold scores

{'=' * 70}
 UPGRADE PATH (when packages are available):
   • Replace GBR with XGBoost/LightGBM for ~5-15% improvement
   • Add SHAP for individual prediction explanations
   • See commented code blocks in the script
{'=' * 70}
"""

print(report)

with open(f'{OUTPUT_DIR}/model_report.txt', 'w') as f:
    f.write(report)
print("✓ Saved: model_report.txt")

# ══════════════════════════════════════════════════════════════════════════════
# APPENDIX: XGBOOST / LIGHTGBM / SHAP UPGRADE CODE (uncomment when available)
# ══════════════════════════════════════════════════════════════════════════════

UPGRADE_CODE = '''
# ═══════════════════════════════════════════════════════════
# UPGRADE 1: XGBoost (drop-in replacement)
# ═══════════════════════════════════════════════════════════
# pip install xgboost

import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,       # L2 regularization
    tree_method='hist',   # fast histogram-based
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

y_pred_xgb = xgb_model.predict(X_test)


# ═══════════════════════════════════════════════════════════
# UPGRADE 2: LightGBM (alternative)
# ═══════════════════════════════════════════════════════════
# pip install lightgbm

import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

lgb_model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
)

y_pred_lgb = lgb_model.predict(X_test)


# ═══════════════════════════════════════════════════════════
# UPGRADE 3: SHAP Explanations
# ═══════════════════════════════════════════════════════════
# pip install shap

import shap

# For tree-based models (XGBoost, LightGBM, or sklearn GBR)
explainer = shap.TreeExplainer(xgb_model)  # or lgb_model or gbr
shap_values = explainer.shap_values(X_test)

# Global feature importance (SHAP bar plot)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Beeswarm plot (detailed feature effects)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# Individual prediction explanation (e.g., first test sample)
fig, ax = plt.subplots(figsize=(12, 4))
shap.force_plot(explainer.expected_value, shap_values[0],
                X_test.iloc[0], feature_names=feature_cols,
                matplotlib=True, show=False)
plt.tight_layout()
plt.savefig('shap_individual.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP dependence plot for top feature
top_feat = feature_cols[np.argmax(np.abs(shap_values).mean(axis=0))]
fig, ax = plt.subplots(figsize=(8, 6))
shap.dependence_plot(top_feat, shap_values, X_test,
                     feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
'''

with open(f'{OUTPUT_DIR}/upgrade_xgboost_shap.py', 'w') as f:
    f.write(UPGRADE_CODE)
print("✓ Saved: upgrade_xgboost_shap.py")

print("\n" + "=" * 70)
print("ALL DONE! Files saved to:", OUTPUT_DIR)
print("=" * 70)