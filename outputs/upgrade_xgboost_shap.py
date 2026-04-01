
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
