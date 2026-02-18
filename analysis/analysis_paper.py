#!/usr/bin/env python3
"""
Winning Strategies in HYROX: A Machine Learning Approach to Race Performance Optimization
==========================================================================================
Comprehensive ML analysis pipeline for HYROX race data.
Generates all figures for the research paper.

Analyses:
  1. PCA - Principal Component Analysis of performance profiles
  2. t-SNE / UMAP - Nonlinear dimensionality reduction
  3. K-Means Clustering - Athlete archetype identification
  4. XGBoost + SHAP - Feature importance & explainability
  5. Elastic Net Regression - Regularized linear model
  6. Pacing Decay Modeling - Exponential fatigue curves
  7. Network Analysis - Workout correlation networks
  8. Quantile Regression - Level-specific performance drivers
  9. Hierarchical Clustering - Workout & nation dendrograms
  10. Bayesian-inspired analysis - Performance distributions
"""

import os
import sys
import glob
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, QuantileRegressor, Ridge
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

import umap

import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ============================================================
# STYLE SETUP
# ============================================================
plt.style.use('default')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'grid.color': '#eeeeee',
    'grid.alpha': 0.6,
    'axes.grid': True,
    'axes.axisbelow': True,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# Colors
C = {
    'blue': '#0d6efd', 'orange': '#e8590c', 'green': '#2f9e44',
    'purple': '#7950f2', 'red': '#e03131', 'cyan': '#1098ad',
    'gold': '#e67700', 'yellow': '#f08c00', 'pink': '#d6336c',
    'bg': '#ffffff', 'card': '#f8f9fa', 'border': '#dee2e6',
    'text': '#212529', 'muted': '#6c757d', 'highlight': '#fd7e14',
}
PALETTE = [C['blue'], C['orange'], C['green'], C['purple'], C['red'],
           C['cyan'], C['gold'], C['yellow'], C['pink']]

OUTPUT_DIR = '/Users/yamanoishuta/HYROX-DATA-ANALYSIS/output/paper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def watermark(fig, text='HYROX Performance Lab'):
    fig.text(0.98, 0.02, text, fontsize=7, color='#cccccc',
             ha='right', va='bottom', style='italic', alpha=0.6)

def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def evaluate_cv_regression(model, X, y, cv_splitter, scale=False):
    """Return fold-aggregated regression metrics for a splitter."""
    r2_list, rmse_list, mae_list = [], [], []
    for train_idx, test_idx in cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler_local = StandardScaler().fit(X_train)
            X_train_fit = scaler_local.transform(X_train)
            X_test_fit = scaler_local.transform(X_test)
        else:
            X_train_fit, X_test_fit = X_train, X_test

        model.fit(X_train_fit, y_train)
        pred = model.predict(X_test_fit)

        r2_list.append(float(r2_score(y_test, pred)))
        rmse_list.append(float(np.sqrt(mean_squared_error(y_test, pred))))
        mae_list.append(float(mean_absolute_error(y_test, pred)))

    return {
        'r2_mean': float(np.mean(r2_list)),
        'r2_std': float(np.std(r2_list)),
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
    }


def evaluate_groupkfold_regression(model, X, y, groups, n_splits=5, scale=False):
    """Event-level holdout evaluation."""
    gkf = GroupKFold(n_splits=n_splits)
    r2_list, rmse_list, mae_list = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler_local = StandardScaler().fit(X_train)
            X_train_fit = scaler_local.transform(X_train)
            X_test_fit = scaler_local.transform(X_test)
        else:
            X_train_fit, X_test_fit = X_train, X_test

        model.fit(X_train_fit, y_train)
        pred = model.predict(X_test_fit)

        r2_list.append(float(r2_score(y_test, pred)))
        rmse_list.append(float(np.sqrt(mean_squared_error(y_test, pred))))
        mae_list.append(float(mean_absolute_error(y_test, pred)))

    return {
        'r2_mean': float(np.mean(r2_list)),
        'r2_std': float(np.std(r2_list)),
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
    }

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

files = glob.glob('/Users/yamanoishuta/HYROX-DATA-ANALYSIS/events/*.csv')
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)
    except:
        pass
data = pd.concat(dfs, ignore_index=True)

data['city'] = data['event'].apply(lambda x: '-'.join(x.split('-')[2:]))
data['year'] = data['event'].apply(lambda x: x.split('-')[1]).astype(int)

workout_cols = ['1000m SkiErg', '50m Sled Push', '50m Sled Pull',
                '80m Burpee Broad Jump', '1000m Row', '200m Farmers Carry',
                '100m Sandbag Lunges', 'Wall Balls']
workout_short = ['SkiErg', 'Sled Push', 'Sled Pull', 'Burpee BJ',
                 'Row', 'Farmers', 'Lunges', 'Wall Balls']
running_cols = [f'Running {i}' for i in range(1, 9)]
all_segment_cols = workout_cols + running_cols

# Focus on men's open division for main analyses
men = data[data['division'] == 'hyrox-men'].copy()
women = data[data['division'] == 'hyrox-women'].copy()
pro_men = data[data['division'] == 'hyrox-pro-men'].copy()

# Drop rows with missing key data
feature_cols = workout_cols + running_cols + ['Roxzone Total']
men = men.dropna(subset=feature_cols + ['Total Time'])

# Remove clearly impossible split values (timing/recording artifacts).
# Thresholds are conservative lower bounds in seconds.
min_plausible_sec = {
    '1000m SkiErg': 60,
    '50m Sled Push': 40,
    '50m Sled Pull': 40,
    '80m Burpee Broad Jump': 40,
    '1000m Row': 60,
    '200m Farmers Carry': 40,
    '100m Sandbag Lunges': 40,
    'Wall Balls': 40,
    'Roxzone Total': 30,
}
for run_col in running_cols:
    min_plausible_sec[run_col] = 60

initial_men_n = len(men)
plausible_mask = np.ones(len(men), dtype=bool)
for col in feature_cols:
    plausible_mask &= men[col].values >= min_plausible_sec[col]
men = men.loc[plausible_mask].copy()
removed_implausible_n = initial_men_n - len(men)

# Tier classification
men['total_min'] = men['Total Time'] / 60
men['tier'] = pd.cut(men['total_min'],
    bins=[0, 60, 70, 80, 90, 100, 120, 999],
    labels=['Sub-60', 'Sub-70', 'Sub-80', 'Sub-90', 'Sub-100', 'Sub-120', '120+'])

print(f"Total records loaded: {len(data):,}")
print(f"Men open (cleaned): {len(men):,}")
print(f"Events: {data['event'].nunique()}")
print(f"Nations: {men['nation'].nunique()}")
print(f"Removed implausible split records: {removed_implausible_n:,}")

# ============================================================
# Prepare standardized features
# ============================================================
scaler = StandardScaler()
X_workout = men[workout_cols].values
X_running = men[running_cols].values
X_all = men[feature_cols].values
X_scaled = scaler.fit_transform(X_all)

# Feature matrix for ML (workout + running + roxzone)
feature_names = [s.replace('Running ', 'Run') for s in feature_cols]

# Store results for paper
results = {}
results['data_quality'] = {
    'men_after_missing_filter': int(initial_men_n),
    'removed_implausible_split_records': int(removed_implausible_n),
    'men_final_n': int(len(men)),
    'min_plausible_sec': min_plausible_sec,
}

# ============================================================
# FIGURE 1: PCA - Variance Explained & Biplot
# ============================================================
print("\n[Fig 1] PCA Analysis...")

pca_full = PCA(n_components=min(len(feature_cols), 10))
X_pca = pca_full.fit_transform(X_scaled)

# 1a: Scree plot
fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
var_exp = pca_full.explained_variance_ratio_ * 100
cum_var = np.cumsum(var_exp)
ax1.bar(range(1, len(var_exp) + 1), var_exp, color=C['blue'], alpha=0.8, edgecolor='none')
ax1.plot(range(1, len(var_exp) + 1), cum_var, 'o-', color=C['orange'], linewidth=2, markersize=6)
ax1.axhline(y=80, color=C['muted'], linestyle='--', alpha=0.5)
ax1.set_xlabel('Principal Component (PC)', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11)
ax1.set_title('Figure 1A', fontsize=13, fontweight='bold', loc='left')
for i, (v, c) in enumerate(zip(var_exp[:5], cum_var[:5])):
    ax1.text(i + 1, v + 1.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
clean_axes(ax1)
watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig01a_pca_scree.png', dpi=200, facecolor='white')
plt.close()

# 1b: PCA biplot (PC1 vs PC2)
fig, ax2 = plt.subplots(figsize=(7.5, 5.5))
tier_colors = {'Sub-60': C['gold'], 'Sub-70': C['orange'], 'Sub-80': C['blue'],
               'Sub-90': C['purple'], 'Sub-100': C['muted'], 'Sub-120': '#adb5bd', '120+': '#ced4da'}
for tier in ['120+', 'Sub-120', 'Sub-100', 'Sub-90', 'Sub-80', 'Sub-70', 'Sub-60']:
    mask = men['tier'] == tier
    if mask.sum() > 0:
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=tier_colors[tier],
                    s=3, alpha=0.4, label=f'{tier} (n={mask.sum():,})')

# Add loading arrows
loadings = pca_full.components_[:2].T
scale_factor = np.abs(X_pca[:, :2]).max() * 0.8 / (np.abs(loadings).max() + 1e-10)
for i, (name, short) in enumerate(zip(feature_cols, feature_names)):
    lx, ly = loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor
    color = C['red'] if 'Running' in name else C['green']
    ax2.annotate('', xy=(lx, ly), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
    ax2.text(lx * 1.12, ly * 1.12, short, fontsize=7, color=color,
             ha='center', va='center', fontweight='bold')

ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontsize=11)
ax2.set_title('Figure 1B', fontsize=13, fontweight='bold', loc='left')
ax2.legend(fontsize=7, loc='upper right', markerscale=3)
clean_axes(ax2)
watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig01b_pca_biplot.png', dpi=200, facecolor='white')
plt.close()

# 1c: Loading heatmap for PC1-PC4
fig, ax3 = plt.subplots(figsize=(7.5, 5.5))
n_pcs = 4
loading_matrix = pca_full.components_[:n_pcs].T
loading_df = pd.DataFrame(loading_matrix, index=feature_names,
                           columns=[f'PC{i+1}' for i in range(n_pcs)])
im = ax3.imshow(loading_df.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax3.set_xticks(range(n_pcs))
ax3.set_xticklabels(loading_df.columns, fontsize=10)
ax3.set_yticks(range(len(feature_names)))
ax3.set_yticklabels(feature_names, fontsize=8)
ax3.set_title('Figure 1C', fontsize=13, fontweight='bold', loc='left')
for i in range(loading_df.shape[0]):
    for j in range(loading_df.shape[1]):
        v = loading_df.values[i, j]
        ax3.text(j, i, f'{v:.2f}', ha='center', va='center',
                 fontsize=6, color='white' if abs(v) > 0.35 else '#333333')
plt.colorbar(im, ax=ax3, shrink=0.6, label='Loading')

results['pca_var_explained'] = var_exp.tolist()
results['pca_cumvar'] = cum_var.tolist()

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig01c_pca_loadings.png', dpi=200, facecolor='white')
plt.close()
print(f"  PC1-3 cumulative: {cum_var[2]:.1f}%")

# ============================================================
# FIGURE 2: UMAP + t-SNE Embedding
# ============================================================
print("\n[Fig 2] UMAP & t-SNE Embeddings...")

# Subsample for speed if needed
n_sample = min(len(men), 15000)
idx = np.random.RandomState(42).choice(len(men), n_sample, replace=False)
X_sub = X_scaled[idx]
tiers_sub = men['tier'].values[idx]
total_sub = men['total_min'].values[idx]

# t-SNE
tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000, learning_rate='auto')
X_tsne = tsne.fit_transform(X_sub)

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_sub)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, embedding, title in [(axes[0], X_tsne, '(a)'),
                               (axes[1], X_umap, '(b)')]:
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=total_sub,
                          cmap='plasma_r', s=3, alpha=0.5, vmin=50, vmax=130)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Dim 1', fontsize=11)
    ax.set_ylabel('Dim 2', fontsize=11)
    clean_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = plt.colorbar(scatter, ax=axes, shrink=0.6, pad=0.02)
cbar.set_label('Finish Time (min)', fontsize=11)

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig02_embeddings.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 3: K-Means Clustering - Athlete Archetypes
# ============================================================
print("\n[Fig 3] K-Means Clustering...")

# Normalize workout and running separately for profiling
X_profile = np.column_stack([
    men[workout_cols].values / men['Workouts Total'].values[:, None],
    men[running_cols].values / men['Running Total'].values[:, None],
])
X_profile = np.nan_to_num(X_profile, nan=0.125)
scaler_profile = StandardScaler()
X_profile_scaled = scaler_profile.fit_transform(X_profile)

# Elbow + Silhouette
from sklearn.metrics import silhouette_score
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_profile_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_profile_scaled, km.labels_, sample_size=5000))

optimal_k = list(K_range)[np.argmax(sil_scores)]
print(f"  Optimal K (silhouette): {optimal_k}")

km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
men['cluster'] = km_final.fit_predict(X_profile_scaled)

# Profile each cluster
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, height_ratios=[1, 1.2])

# 3a: Elbow
ax_elbow = fig.add_subplot(gs[0, 0])
ax_elbow.plot(list(K_range), inertias, 'o-', color=C['blue'], linewidth=2)
ax_elbow.set_xlabel('Number of Clusters (K)', fontsize=11)
ax_elbow.set_ylabel('Inertia', fontsize=11)
ax_elbow.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax_elbow)

# 3b: Silhouette
ax_sil = fig.add_subplot(gs[0, 1])
ax_sil.plot(list(K_range), sil_scores, 'o-', color=C['green'], linewidth=2)
ax_sil.axvline(x=optimal_k, color=C['orange'], linestyle='--', alpha=0.7)
ax_sil.set_xlabel('Number of Clusters (K)', fontsize=11)
ax_sil.set_ylabel('Silhouette Score', fontsize=11)
ax_sil.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax_sil.text(optimal_k + 0.2, max(sil_scores), f'Best K={optimal_k}',
            fontsize=10, color=C['orange'], fontweight='bold')
clean_axes(ax_sil)

# 3c: Cluster sizes & avg total time
ax_dist = fig.add_subplot(gs[0, 2])
cluster_stats = men.groupby('cluster').agg(
    count=('Total Time', 'count'),
    avg_time=('total_min', 'mean'),
    median_time=('total_min', 'median')
).reset_index()
bars = ax_dist.bar(cluster_stats['cluster'], cluster_stats['count'],
                    color=[PALETTE[i] for i in cluster_stats['cluster']], edgecolor='none')
for i, row in cluster_stats.iterrows():
    ax_dist.text(row['cluster'], row['count'] + 50,
                 f'{row["median_time"]:.0f} min', ha='center', fontsize=10, fontweight='bold')
ax_dist.set_xlabel('Cluster', fontsize=11)
ax_dist.set_ylabel('Athletes', fontsize=11)
ax_dist.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax_dist)

# 3d: Radar chart for each cluster (workout Z-scores)
ax_radar = fig.add_subplot(gs[1, :], polar=True)
angles = np.linspace(0, 2 * np.pi, len(workout_cols), endpoint=False).tolist()
angles += angles[:1]

cluster_profiles = {}
archetype_names = {}
for c in sorted(men['cluster'].unique()):
    cluster_men = men[men['cluster'] == c]
    # Z-score relative to overall mean
    profile = []
    for col in workout_cols:
        z = (cluster_men[col].mean() - men[col].mean()) / men[col].std()
        profile.append(z)
    profile += profile[:1]
    cluster_profiles[c] = profile

    # Determine archetype name
    run_avg = cluster_men['Running Total'].mean()
    wo_avg = cluster_men['Workouts Total'].mean()
    run_z = (run_avg - men['Running Total'].mean()) / men['Running Total'].std()
    wo_z = (wo_avg - men['Workouts Total'].mean()) / men['Workouts Total'].std()

    if run_z < -0.3 and wo_z < -0.3:
        name = 'Elite'
    elif run_z < wo_z - 0.3:
        name = 'Runner-Dominant'
    elif wo_z < run_z - 0.3:
        name = 'Strength-Dominant'
    elif cluster_men['total_min'].mean() > 100:
        name = 'Beginner'
    else:
        name = 'Balanced'
    archetype_names[c] = name

    ax_radar.plot(angles, profile, 'o-', color=PALETTE[c], linewidth=2, markersize=4,
                  label=f'C{c}: {name} (n={len(cluster_men):,}, med={cluster_men["total_min"].median():.0f} min)')
    ax_radar.fill(angles, profile, color=PALETTE[c], alpha=0.1)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(workout_short, fontsize=9)
ax_radar.set_title('(d)', fontsize=13, fontweight='bold', loc='left', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

results['n_clusters'] = optimal_k
results['cluster_stats'] = cluster_stats.to_dict()
results['archetype_names'] = archetype_names

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig03_clustering.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 4: XGBoost + SHAP Analysis
# ============================================================
print("\n[Fig 4] XGBoost + SHAP...")

X_ml = men[feature_cols].values
y_ml = men['total_min'].values

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    tree_method='hist'
)
xgb_model.fit(X_ml, y_ml)

# Cross-validation
cv_scores = cross_val_score(xgb_model, X_ml, y_ml, cv=5, scoring='r2')
print(f"  XGBoost CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_ml)

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2)

# 4a: Feature importance (SHAP mean absolute)
ax1 = fig.add_subplot(gs[0, 0])
shap_importance = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(shap_importance)
sorted_names = [feature_names[i] for i in sorted_idx]
colors_imp = [C['red'] if 'Run' in n else C['green'] if 'Rox' in n else C['blue']
              for n in sorted_names]
ax1.barh(range(len(sorted_idx)), shap_importance[sorted_idx], color=colors_imp, height=0.7)
ax1.set_yticks(range(len(sorted_idx)))
ax1.set_yticklabels(sorted_names, fontsize=9)
ax1.set_xlabel('mean |SHAP value| (min)', fontsize=11)
ax1.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
run_patch = mpatches.Patch(color=C['red'], label='Running')
wo_patch = mpatches.Patch(color=C['blue'], label='Workout')
rox_patch = mpatches.Patch(color=C['green'], label='Roxzone')
ax1.legend(handles=[run_patch, wo_patch, rox_patch], fontsize=9, loc='lower right')
clean_axes(ax1)

# 4b: SHAP beeswarm (manual implementation for dark theme)
ax2 = fig.add_subplot(gs[0, 1])
top_n = 12
top_features = np.argsort(shap_importance)[-top_n:][::-1]
for rank, fi in enumerate(top_features):
    shap_vals = shap_values[:, fi]
    feat_vals = X_ml[:, fi]
    # Normalize feature values for coloring
    vmin, vmax = np.percentile(feat_vals, [5, 95])
    colors_norm = np.clip((feat_vals - vmin) / (vmax - vmin + 1e-10), 0, 1)
    # Subsample for readability
    n_show = min(2000, len(shap_vals))
    idx_show = np.random.RandomState(42).choice(len(shap_vals), n_show, replace=False)
    jitter = np.random.RandomState(42).normal(0, 0.15, n_show)
    ax2.scatter(shap_vals[idx_show], rank + jitter,
                c=plt.cm.coolwarm(colors_norm[idx_show]),
                s=2, alpha=0.4, rasterized=True)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([feature_names[i] for i in top_features], fontsize=9)
ax2.set_xlabel('SHAP value (contribution in min)', fontsize=11)
ax2.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax2.axvline(x=0, color=C['muted'], linestyle='--', alpha=0.5)
# Add colorbar
sm_cb = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
sm_cb.set_array([])
cbar = plt.colorbar(sm_cb, ax=ax2, shrink=0.5, pad=0.02)
cbar.set_label('Feature value (low -> high)', fontsize=9)
clean_axes(ax2)

# 4c: SHAP interaction - top 2 features
ax3 = fig.add_subplot(gs[1, 0])
top2 = np.argsort(shap_importance)[-2:]
f1, f2 = top2[1], top2[0]
n_show = min(5000, len(X_ml))
idx_show = np.random.RandomState(42).choice(len(X_ml), n_show, replace=False)
scatter = ax3.scatter(X_ml[idx_show, f1] / 60, X_ml[idx_show, f2] / 60,
                       c=shap_values[idx_show, f1], cmap='RdBu_r',
                       s=5, alpha=0.5, vmin=-5, vmax=5, rasterized=True)
ax3.set_xlabel(f'{feature_names[f1]} (min)', fontsize=11)
ax3.set_ylabel(f'{feature_names[f2]} (min)', fontsize=11)
ax3.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
plt.colorbar(scatter, ax=ax3, shrink=0.7, label='SHAP value')
clean_axes(ax3)

# 4d: Partial dependence-like plot using SHAP
ax4 = fig.add_subplot(gs[1, 1])
most_important = np.argsort(shap_importance)[-1]
feat_val = X_ml[:, most_important] / 60  # convert to minutes
shap_val = shap_values[:, most_important]
# Bin and average
bins = np.linspace(np.percentile(feat_val, 2), np.percentile(feat_val, 98), 30)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_means = []
bin_stds = []
for i in range(len(bins) - 1):
    mask = (feat_val >= bins[i]) & (feat_val < bins[i + 1])
    if mask.sum() > 10:
        bin_means.append(shap_val[mask].mean())
        bin_stds.append(shap_val[mask].std())
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)
bin_means = np.array(bin_means)
bin_stds = np.array(bin_stds)
valid = ~np.isnan(bin_means)
ax4.fill_between(bin_centers[valid], (bin_means - bin_stds)[valid], (bin_means + bin_stds)[valid],
                  alpha=0.2, color=C['blue'])
ax4.plot(bin_centers[valid], bin_means[valid], '-', color=C['blue'], linewidth=2.5)
ax4.axhline(y=0, color=C['muted'], linestyle='--', alpha=0.5)
ax4.set_xlabel(f'{feature_names[most_important]} (min)', fontsize=11)
ax4.set_ylabel('SHAP value (min)', fontsize=11)
ax4.set_title('(d)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax4)

results['xgb_cv_r2'] = float(cv_scores.mean())
results['xgb_cv_std'] = float(cv_scores.std())
results['shap_importance'] = {feature_names[i]: float(shap_importance[i]) for i in range(len(feature_names))}

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig04_xgboost_shap.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 5: Elastic Net + Feature Coefficients
# ============================================================
print("\n[Fig 5] Elastic Net Regression...")

X_en = StandardScaler().fit_transform(X_ml)

en_model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                         cv=5, random_state=42, max_iter=5000)
en_model.fit(X_en, y_ml)
en_pred = en_model.predict(X_en)
en_r2 = r2_score(y_ml, en_pred)
en_cv = cross_val_score(en_model, X_en, y_ml, cv=5, scoring='r2')

print(f"  Elastic Net R²: {en_r2:.4f}, CV: {en_cv.mean():.4f}")
print(f"  Optimal l1_ratio: {en_model.l1_ratio_:.2f}, alpha: {en_model.alpha_:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 5a: Coefficients
coefs = pd.Series(en_model.coef_, index=feature_names)
coefs_sorted = coefs.abs().sort_values(ascending=True)
colors_coef = [C['red'] if coefs[n] > 0 else C['blue'] for n in coefs_sorted.index]
axes[0].barh(range(len(coefs_sorted)), coefs[coefs_sorted.index].values,
              color=colors_coef, height=0.7)
axes[0].set_yticks(range(len(coefs_sorted)))
axes[0].set_yticklabels(coefs_sorted.index, fontsize=9)
axes[0].set_xlabel('Standardized Coefficient', fontsize=11)
axes[0].set_title('(a)', fontsize=13, fontweight='bold', loc='left')
axes[0].axvline(x=0, color=C['muted'], linestyle='--', alpha=0.5)
clean_axes(axes[0])

# 5b: Predicted vs Actual
axes[1].scatter(y_ml, en_pred, s=2, alpha=0.3, color=C['blue'], rasterized=True)
axes[1].plot([40, 160], [40, 160], '--', color=C['red'], linewidth=1.5)
axes[1].set_xlabel('Observed Time (min)', fontsize=11)
axes[1].set_ylabel('Predicted Time (min)', fontsize=11)
axes[1].set_title('(b)', fontsize=13, fontweight='bold', loc='left')
clean_axes(axes[1])

# 5c: Residual distribution
residuals = y_ml - en_pred
axes[2].hist(residuals, bins=50, color=C['purple'], alpha=0.8, edgecolor='white')
axes[2].axvline(x=0, color=C['red'], linestyle='--', linewidth=1.5)
axes[2].set_xlabel('Residual (min)', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('(c)', fontsize=13, fontweight='bold', loc='left')
clean_axes(axes[2])

results['en_r2'] = float(en_r2)
results['en_coefs'] = coefs.to_dict()

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig05_elastic_net.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 6: Pacing Degradation Model
# ============================================================
print("\n[Fig 6] Pacing Decay Model...")

def exp_decay(x, a, b, c):
    """Exponential decay: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def linear_decay(x, a, b):
    return a * x + b

# Compute pacing ratio for each tier
tiers_to_plot = ['Sub-60', 'Sub-70', 'Sub-80', 'Sub-90', 'Sub-100']
segments = np.arange(1, 9)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 6a: Raw pacing curves
ax = axes[0]
decay_params = {}
for tier in tiers_to_plot:
    tier_data = men[men['tier'] == tier]
    if len(tier_data) < 50:
        continue
    splits = [tier_data[f'Running {i}'].mean() for i in range(1, 9)]
    ratio = np.array([s / splits[0] for s in splits])
    ax.plot(segments, ratio, 'o-', color=tier_colors.get(tier, C['muted']),
            linewidth=2, markersize=6, label=f'{tier} (n={len(tier_data):,})')

    # Fit exponential decay
    try:
        popt, _ = curve_fit(exp_decay, segments.astype(float), ratio,
                            p0=[0.01, 0.1, 1.0], maxfev=5000)
        decay_params[tier] = {'a': popt[0], 'b': popt[1], 'c': popt[2],
                              'final_slowdown': (ratio[-1] - 1) * 100}
    except:
        decay_params[tier] = {'final_slowdown': (ratio[-1] - 1) * 100}

ax.axhline(y=1.0, color=C['muted'], linestyle='--', alpha=0.5)
ax.set_xlabel('Running Segment', fontsize=11)
ax.set_ylabel('Pace Ratio vs Run 1', fontsize=11)
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=9)
ax.set_xticks(segments)
clean_axes(ax)

# 6b: Decay rate comparison
ax = axes[1]
decay_rates = []
for tier in tiers_to_plot:
    if tier in decay_params:
        decay_rates.append(decay_params[tier]['final_slowdown'])
    else:
        decay_rates.append(0)
bar_colors = [tier_colors.get(t, C['muted']) for t in tiers_to_plot]
bars = ax.bar(range(len(tiers_to_plot)), decay_rates, color=bar_colors, width=0.6, edgecolor='none')
ax.set_xticks(range(len(tiers_to_plot)))
ax.set_xticklabels(tiers_to_plot, fontsize=10)
ax.set_ylabel('Pace Drop at Run 8 (%)', fontsize=11)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
for i, v in enumerate(decay_rates):
    ax.text(i, v + 0.5, f'+{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
clean_axes(ax)

# 6c: Individual variation in pacing (violin plot)
ax = axes[2]
men_clean = men[men['tier'].isin(['Sub-70', 'Sub-80', 'Sub-90'])].copy()
men_clean['pace_decay'] = (men_clean['Running 8'] / men_clean['Running 1'] - 1) * 100
violin_data = [men_clean[men_clean['tier'] == t]['pace_decay'].dropna().values
               for t in ['Sub-70', 'Sub-80', 'Sub-90']]
parts = ax.violinplot(violin_data, positions=[0, 1, 2], showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([C['orange'], C['blue'], C['purple']][i])
    pc.set_alpha(0.6)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Sub-70', 'Sub-80', 'Sub-90'], fontsize=11)
ax.set_ylabel('Pace Drop (%)', fontsize=11)
ax.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax)

results['decay_params'] = decay_params

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig06_pacing_decay.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 7: Workout Correlation Network
# ============================================================
print("\n[Fig 7] Correlation Network Analysis...")

# Compute correlation matrix for all segments
all_cols = workout_cols + running_cols
corr_matrix = men[all_cols].corr()

# Build network
G = nx.Graph()
node_labels = workout_short + [f'Run{i}' for i in range(1, 9)]
node_types = ['workout'] * 8 + ['running'] * 8

for i, label in enumerate(node_labels):
    G.add_node(label, type=node_types[i])

# Add edges for significant correlations
threshold = 0.3
for i in range(len(all_cols)):
    for j in range(i + 1, len(all_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > threshold:
            G.add_edge(node_labels[i], node_labels[j], weight=abs(r), corr=r)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 7a: Network graph
ax = axes[0]
pos = nx.spring_layout(G, seed=42, k=2)
node_colors = [C['blue'] if G.nodes[n]['type'] == 'workout' else C['red'] for n in G.nodes()]
node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]
edges = G.edges(data=True)
edge_widths = [e[2]['weight'] * 5 for e in edges]
edge_colors = [C['green'] if e[2]['corr'] > 0 else C['red'] for e in edges]
edge_alphas = [min(e[2]['weight'], 0.8) for e in edges]

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='#333333', font_weight='bold')
for (u, v, d), w, c, a in zip(edges, edge_widths, edge_colors, edge_alphas):
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=c, linewidth=w, alpha=a)

wo_patch = mpatches.Patch(color=C['blue'], label='Workout')
run_patch = mpatches.Patch(color=C['red'], label='Running')
ax.legend(handles=[wo_patch, run_patch], fontsize=10, loc='upper left')
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
ax.axis('off')

# 7b: Full correlation heatmap
ax = axes[1]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
display_labels = workout_short + [f'Run{i}' for i in range(1, 9)]
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-0.2, vmax=1.0, aspect='equal')
ax.set_xticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(len(display_labels)))
ax.set_yticklabels(display_labels, fontsize=7)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
# Add correlation values for strong ones
for i in range(len(display_labels)):
    for j in range(len(display_labels)):
        if i != j and abs(corr_matrix.values[i, j]) > 0.4:
            ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=5, color='white' if abs(corr_matrix.values[i, j]) > 0.5 else '#333333')
plt.colorbar(im, ax=ax, shrink=0.7, label='Pearson r')

# Compute network metrics
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
results['network'] = {
    'degree_centrality': degree_centrality,
    'betweenness': betweenness,
    'n_edges': G.number_of_edges(),
    'avg_clustering': nx.average_clustering(G),
}

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig07_network.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 8: Quantile Regression - What matters at each level?
# ============================================================
print("\n[Fig 8] Quantile Regression...")

X_qr = StandardScaler().fit_transform(men[workout_cols + running_cols].values)
y_qr = men['total_min'].values

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
qr_coefs = {}

for q in quantiles:
    try:
        X_qr_sm = sm.add_constant(X_qr)
        qr_model = QuantReg(y_qr, X_qr_sm)
        qr_result = qr_model.fit(q=q, max_iter=5000)
        qr_coefs[q] = qr_result.params[1:]  # exclude intercept
    except Exception as e:
        print(f"  Quantile {q} failed: {e}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 8a: Coefficient evolution across quantiles
ax = axes[0]
qr_feature_names = workout_short + [f'Run{i}' for i in range(1, 9)]
top_features_qr = sorted(range(len(qr_feature_names)),
                           key=lambda i: np.mean([abs(qr_coefs[q][i]) for q in qr_coefs]),
                           reverse=True)[:8]

for fi in top_features_qr:
    vals = [qr_coefs[q][fi] for q in quantiles]
    color = C['red'] if fi >= 8 else C['blue']
    ax.plot(quantiles, vals, 'o-', linewidth=2, markersize=6,
            label=qr_feature_names[fi], alpha=0.8)

ax.axhline(y=0, color=C['muted'], linestyle='--', alpha=0.5)
ax.set_xlabel('Quantile (0.1=fast group, 0.9=slow group)', fontsize=11)
ax.set_ylabel('Standardized Coefficient', fontsize=11)
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=8, ncol=2, loc='upper left')
clean_axes(ax)

# 8b: Heatmap of quantile regression coefficients
ax = axes[1]
qr_matrix = np.array([qr_coefs[q] for q in quantiles]).T
im = ax.imshow(qr_matrix, cmap='RdBu_r', aspect='auto',
                vmin=-np.percentile(np.abs(qr_matrix), 95),
                vmax=np.percentile(np.abs(qr_matrix), 95))
ax.set_xticks(range(len(quantiles)))
ax.set_xticklabels([f'Q{int(q*100)}' for q in quantiles], fontsize=10)
ax.set_yticks(range(len(qr_feature_names)))
ax.set_yticklabels(qr_feature_names, fontsize=8)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax.set_xlabel('Quantile', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.7, label='Standardized Coefficient')

results['quantile_regression'] = {str(q): {qr_feature_names[i]: float(qr_coefs[q][i])
                                            for i in range(len(qr_feature_names))}
                                   for q in qr_coefs}

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig08_quantile_regression.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 9: Hierarchical Clustering Dendrogram
# ============================================================
print("\n[Fig 9] Hierarchical Clustering...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 9a: Workout dendrogram
X_wo = men[workout_cols].values
corr_wo = np.corrcoef(X_wo.T)
dist_wo = 1 - corr_wo
dist_wo = (dist_wo + dist_wo.T) / 2  # force symmetry
np.fill_diagonal(dist_wo, 0)
dist_wo = np.clip(dist_wo, 0, None)
linkage_wo = linkage(squareform(dist_wo), method='ward')

ax = axes[0]
dend = dendrogram(linkage_wo, labels=workout_short, ax=ax, leaf_rotation=45,
                   leaf_font_size=10, color_threshold=0.7,
                   above_threshold_color=C['muted'])
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
ax.set_ylabel('Distance (1 - Pearson r)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9b: Nation dendrogram (top 20 nations)
nation_stats = men.groupby('nation').agg(
    count=('Total Time', 'count'),
    median_time=('total_min', 'median')
).reset_index()
top_nations = nation_stats[nation_stats['count'] >= 100].nlargest(20, 'count')['nation'].values

nation_profiles = []
nation_labels = []
for n in top_nations:
    n_data = men[men['nation'] == n]
    profile = []
    for col in workout_cols:
        z = (n_data[col].mean() - men[col].mean()) / men[col].std()
        profile.append(z)
    for col in running_cols:
        z = (n_data[col].mean() - men[col].mean()) / men[col].std()
        profile.append(z)
    nation_profiles.append(profile)
    nation_labels.append(n)

nation_profiles = np.array(nation_profiles)
linkage_nat = linkage(nation_profiles, method='ward')

ax = axes[1]
dend_n = dendrogram(linkage_nat, labels=nation_labels, ax=ax, leaf_rotation=45,
                     leaf_font_size=10, color_threshold=3.0,
                     above_threshold_color=C['muted'])
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax.set_ylabel('Ward Distance', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig09_dendrograms.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 10: Multi-model comparison + Key findings
# ============================================================
print("\n[Fig 10] Model Comparison & Strategic Insights...")

# Compare multiple models
models = {
    'Elastic Net': ElasticNetCV(cv=5, random_state=42, max_iter=5000),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 random_state=42, tree_method='hist'),
    'Gradient Boost': GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
}

X_comp = StandardScaler().fit_transform(X_ml)
cv_results = {}
for name, model in models.items():
    if name == 'Elastic Net':
        scores = cross_val_score(model, X_comp, y_ml, cv=5, scoring='r2')
    else:
        scores = cross_val_score(model, X_ml, y_ml, cv=5, scoring='r2')
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"  {name}: R²={scores.mean():.4f} ± {scores.std():.4f}")

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3)

# 10a: Model comparison
ax = fig.add_subplot(gs[0, 0])
model_names = list(cv_results.keys())
means = [cv_results[n]['mean'] for n in model_names]
stds = [cv_results[n]['std'] for n in model_names]
colors_m = [C['blue'], C['green'], C['orange'], C['purple']]
bars = ax.bar(range(len(model_names)), means, yerr=stds, color=colors_m,
               width=0.6, edgecolor='none', capsize=5, error_kw={'color': '#333333', 'linewidth': 1.5})
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=9, rotation=20)
ax.set_ylabel('CV R² Score', fontsize=11)
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')
clean_axes(ax)

# 10b: Running vs Workout contribution (stacked area by tier)
ax = fig.add_subplot(gs[0, 1])
tier_breakdown = men.groupby('tier').agg(
    run_pct=('Running Total', lambda x: x.mean()),
    wo_pct=('Workouts Total', lambda x: x.mean()),
    rox_pct=('Roxzone Total', lambda x: x.mean()),
    total=('Total Time', 'mean')
).dropna()

# Convert to percentages
for col in ['run_pct', 'wo_pct', 'rox_pct']:
    tier_breakdown[col] = tier_breakdown[col] / tier_breakdown['total'] * 100

tiers_plot = tier_breakdown.index.tolist()
x = range(len(tiers_plot))
ax.bar(x, tier_breakdown['run_pct'], color=C['blue'], label='Running', edgecolor='none')
ax.bar(x, tier_breakdown['wo_pct'], bottom=tier_breakdown['run_pct'],
        color=C['orange'], label='Workout', edgecolor='none')
ax.bar(x, tier_breakdown['rox_pct'],
        bottom=tier_breakdown['run_pct'] + tier_breakdown['wo_pct'],
        color=C['muted'], label='Roxzone', edgecolor='none')
ax.set_xticks(x)
ax.set_xticklabels(tiers_plot, fontsize=9)
ax.set_ylabel('Time Composition (%)', fontsize=11)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=9, loc='upper right')
clean_axes(ax)

# 10c: Marginal gain analysis
ax = fig.add_subplot(gs[0, 2])
# For each segment, compute: if you improve by 1 std, how many minutes total time improves?
marginal_gains = {}
for i, col in enumerate(feature_cols):
    marginal_gains[feature_names[i]] = float(en_model.coef_[i] * men[col].std() / 60)
    # Convert: 1 std improvement → total minutes saved

mg_series = pd.Series(marginal_gains).sort_values()
is_run = ['Run' in n for n in mg_series.index]
colors_mg = [C['red'] if r else C['blue'] for r in is_run]
ax.barh(range(len(mg_series)), mg_series.values, color=colors_mg, height=0.7)
ax.set_yticks(range(len(mg_series)))
ax.set_yticklabels(mg_series.index, fontsize=8)
ax.set_xlabel('Time Saved by 1σ Improvement (min)', fontsize=11)
ax.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax)

# 10d: Workout difficulty × importance matrix
ax = fig.add_subplot(gs[1, 0:2])
cv_vals = []
importance_vals = []
for i, col in enumerate(workout_cols):
    cv = men[col].std() / men[col].mean() * 100  # Coefficient of variation
    imp = float(shap_importance[i])
    cv_vals.append(cv)
    importance_vals.append(imp)

ax.scatter(cv_vals, importance_vals, s=200, c=PALETTE[:8], alpha=0.8, edgecolors='#333333', linewidth=1.5)
for i, txt in enumerate(workout_short):
    ax.annotate(txt, (cv_vals[i], importance_vals[i]),
                fontsize=10, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 10), textcoords='offset points')

# Add quadrant lines
ax.axhline(y=np.median(importance_vals), color=C['muted'], linestyle='--', alpha=0.5)
ax.axvline(x=np.median(cv_vals), color=C['muted'], linestyle='--', alpha=0.5)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] + (np.median(cv_vals) - xlim[0]) * 0.5,
        ylim[1] * 0.95, 'Stable + Important\n(Build Base)', fontsize=9, color=C['green'],
        ha='center', va='top', fontweight='bold')
ax.text(xlim[1] - (xlim[1] - np.median(cv_vals)) * 0.5,
        ylim[1] * 0.95, 'Unstable + Important\n(Top Priority)', fontsize=9, color=C['red'],
        ha='center', va='top', fontweight='bold')
ax.text(xlim[0] + (np.median(cv_vals) - xlim[0]) * 0.5,
        ylim[0] + (np.median(importance_vals) - ylim[0]) * 0.3,
        'Stable + Low Importance\n(Maintain)', fontsize=9, color=C['muted'],
        ha='center', va='bottom')
ax.text(xlim[1] - (xlim[1] - np.median(cv_vals)) * 0.5,
        ylim[0] + (np.median(importance_vals) - ylim[0]) * 0.3,
        'Unstable + Low Importance\n(Room to Improve)', fontsize=9, color=C['cyan'],
        ha='center', va='bottom')

ax.set_xlabel('Performance Variability (CV%)', fontsize=11)
ax.set_ylabel('SHAP Importance (min)', fontsize=11)
ax.set_title('(d)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax)

# 10e: Summary statistics table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
summary_text = [
    ['Metric', 'Value'],
    ['Population', f'{len(men):,} athletes / {men["event"].nunique()} events'],
    ['XGBoost R²', f'{cv_results["XGBoost"]["mean"]:.3f} ± {cv_results["XGBoost"]["std"]:.3f}'],
    ['Elastic Net R²', f'{en_r2:.3f}'],
    ['PCA PC1-3 Cumulative', f'{cum_var[2]:.1f}%'],
    ['Best Cluster Count', f'{optimal_k}'],
    ['Network Edge Count', f'{G.number_of_edges()}'],
    ['Top Feature', f'{feature_names[np.argmax(shap_importance)]}'],
    ['Median Finish Time', f'{men["total_min"].median():.1f} min'],
]
table = ax.table(cellText=summary_text[1:], colLabels=summary_text[0],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(C['border'])
    if row == 0:
        cell.set_facecolor(C['card'])
        cell.set_text_props(fontweight='bold', color='#212529')
    else:
        cell.set_facecolor(C['bg'])
        cell.set_text_props(color=C['text'])
ax.set_title('(e)', fontsize=13, fontweight='bold', loc='left', pad=20)

results['model_comparison'] = {n: {'r2_mean': float(cv_results[n]['mean']),
                                    'r2_std': float(cv_results[n]['std'])}
                                for n in cv_results}

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_synthesis.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 11: Japan-specific deep dive
# ============================================================
print("\n[Fig 11] Japan Deep Dive...")

jp = men[men['nation'] == 'JP'].copy()
world = men[men['nation'] != 'JP'].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 11a: Distribution comparison
ax = axes[0, 0]
ax.hist(world['total_min'], bins=60, density=True, alpha=0.6, color=C['blue'], label=f'World (n={len(world):,})')
ax.hist(jp['total_min'], bins=30, density=True, alpha=0.7, color=C['orange'], label=f'Japan (n={len(jp):,})')
# KDE
from scipy.stats import gaussian_kde
x_range = np.linspace(40, 160, 200)
if len(jp) > 10:
    kde_jp = gaussian_kde(jp['total_min'].dropna())
    ax.plot(x_range, kde_jp(x_range), color=C['orange'], linewidth=2.5)
kde_world = gaussian_kde(world['total_min'].dropna())
ax.plot(x_range, kde_world(x_range), color=C['blue'], linewidth=2.5)
# KS test
ks_stat, ks_p = stats.ks_2samp(jp['total_min'].dropna(), world['total_min'].dropna())
ax.text(0.95, 0.95, f'KS test: D={ks_stat:.3f}\np={ks_p:.2e}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor=C['card'], edgecolor=C['border']))
ax.set_xlabel('Finish Time (min)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=10)
clean_axes(ax)

# 11b: Radar comparison
ax = axes[0, 1]
ax = fig.add_subplot(2, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(workout_cols), endpoint=False).tolist()
angles += angles[:1]

jp_z = [(jp[col].mean() - men[col].mean()) / men[col].std() for col in workout_cols]
world_z = [(world[col].mean() - men[col].mean()) / men[col].std() for col in workout_cols]
jp_z += jp_z[:1]
world_z += world_z[:1]

ax.plot(angles, jp_z, 'o-', color=C['orange'], linewidth=2, label='Japan', markersize=6)
ax.fill(angles, jp_z, color=C['orange'], alpha=0.15)
ax.plot(angles, world_z, 's-', color=C['blue'], linewidth=2, label='World', markersize=6)
ax.fill(angles, world_z, color=C['blue'], alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(workout_short, fontsize=9)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

# 11c: Japan weakness analysis
ax = axes[1, 0]
jp_gap = {}
for col in workout_cols + running_cols:
    jp_mean = jp[col].mean()
    world_mean = world[col].mean()
    gap_pct = (jp_mean - world_mean) / world_mean * 100
    jp_gap[col] = gap_pct

gap_series = pd.Series(jp_gap).sort_values()
gap_labels = [workout_short[workout_cols.index(c)] if c in workout_cols
              else c.replace('Running ', 'Run') for c in gap_series.index]
colors_gap = [C['green'] if v < 0 else C['red'] for v in gap_series.values]
ax.barh(range(len(gap_series)), gap_series.values, color=colors_gap, height=0.6)
ax.set_yticks(range(len(gap_series)))
ax.set_yticklabels(gap_labels, fontsize=8)
ax.axvline(x=0, color='#333333', linewidth=0.5)
ax.set_xlabel('Gap vs Global Mean (%)', fontsize=11)
ax.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
clean_axes(ax)

# 11d: Japan top athletes vs world top athletes
ax = axes[1, 1]
percentiles = [10, 25, 50, 75, 90]
jp_pcts = [np.percentile(jp['total_min'].dropna(), p) for p in percentiles]
world_pcts = [np.percentile(world['total_min'].dropna(), p) for p in percentiles]
x = np.arange(len(percentiles))
w = 0.35
ax.bar(x - w/2, jp_pcts, w, color=C['orange'], label='Japan', edgecolor='none')
ax.bar(x + w/2, world_pcts, w, color=C['blue'], label='World', edgecolor='none')
ax.set_xticks(x)
ax.set_xticklabels([f'P{p}' for p in percentiles], fontsize=11)
ax.set_ylabel('Time (min)', fontsize=11)
ax.set_title('(d)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=10)
for i in range(len(percentiles)):
    gap = jp_pcts[i] - world_pcts[i]
    ax.text(i, max(jp_pcts[i], world_pcts[i]) + 1, f'+{gap:.0f} min',
            ha='center', fontsize=9, color=C['red'] if gap > 0 else C['green'], fontweight='bold')
clean_axes(ax)

results['japan'] = {
    'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
    'jp_median': float(jp['total_min'].median()),
    'world_median': float(world['total_min'].median()),
    'jp_count': int(len(jp)),
}

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_japan_deep_dive.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# FIGURE 12: Transition (Roxzone) Analysis
# ============================================================
print("\n[Fig 12] Roxzone / Transition Analysis...")

# Estimate per-transition times
# Roxzone Total = Total Time - Running Total - Workouts Total
# Divide by 8 for average per transition
men['avg_roxzone'] = men['Roxzone Total'] / 8
men['roxzone_pct'] = men['Roxzone Total'] / men['Total Time'] * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 12a: Roxzone time by performance tier
ax = axes[0]
rox_by_tier = men.groupby('tier').agg(
    mean_rox=('avg_roxzone', 'mean'),
    std_rox=('avg_roxzone', 'std'),
    pct=('roxzone_pct', 'mean')
).dropna()
valid_tiers = rox_by_tier.index.tolist()
bars = ax.bar(range(len(valid_tiers)), rox_by_tier['mean_rox'].values,
               yerr=rox_by_tier['std_rox'].values,
               color=C['purple'], edgecolor='none', width=0.6,
               capsize=4, error_kw={'color': '#333333', 'linewidth': 1})
ax.set_xticks(range(len(valid_tiers)))
ax.set_xticklabels(valid_tiers, fontsize=10)
ax.set_ylabel('Average Roxzone Time per Segment (s)', fontsize=11)
ax.set_title('(a)', fontsize=13, fontweight='bold', loc='left')
for i, (v, pct) in enumerate(zip(rox_by_tier['mean_rox'], rox_by_tier['pct'])):
    ax.text(i, v + rox_by_tier['std_rox'].values[i] + 1, f'{v:.0f}s\n({pct:.0f}%)',
            ha='center', fontsize=9, fontweight='bold')
clean_axes(ax)

# 12b: Roxzone vs total time scatter
ax = axes[1]
n_show = min(5000, len(men))
idx_show = np.random.RandomState(42).choice(len(men), n_show, replace=False)
ax.scatter(men['Roxzone Total'].values[idx_show] / 60,
            men['total_min'].values[idx_show],
            s=3, alpha=0.3, color=C['blue'], rasterized=True)
# Regression line
rox_vals = men['Roxzone Total'].values / 60
total_vals = men['total_min'].values
slope, intercept = np.polyfit(rox_vals, total_vals, 1)
x_line = np.linspace(np.percentile(rox_vals, 2), np.percentile(rox_vals, 98), 100)
ax.plot(x_line, slope * x_line + intercept, color=C['red'], linewidth=2,
        label=f'y = {slope:.1f}x + {intercept:.1f}\nr = {np.corrcoef(rox_vals, total_vals)[0,1]:.3f}')
ax.set_xlabel('Total Roxzone Time (min)', fontsize=11)
ax.set_ylabel('Finish Time (min)', fontsize=11)
ax.set_title('(b)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=10)
clean_axes(ax)

# 12c: Roxzone as hidden gain
ax = axes[2]
# How much time could you save if your roxzone matched the next tier?
tiers_seq = ['120+', 'Sub-120', 'Sub-100', 'Sub-90', 'Sub-80', 'Sub-70', 'Sub-60']
savings = []
tier_labels = []
for i in range(len(tiers_seq) - 1):
    current = men[men['tier'] == tiers_seq[i]]['Roxzone Total'].mean()
    target = men[men['tier'] == tiers_seq[i + 1]]['Roxzone Total'].mean()
    if not np.isnan(current) and not np.isnan(target):
        savings.append((current - target) / 60)
        tier_labels.append(f'{tiers_seq[i]}→\n{tiers_seq[i+1]}')

ax.bar(range(len(savings)), savings, color=C['green'], edgecolor='none', width=0.6)
ax.set_xticks(range(len(savings)))
ax.set_xticklabels(tier_labels, fontsize=9)
ax.set_ylabel('Time Gain from Roxzone Reduction (min)', fontsize=11)
ax.set_title('(c)', fontsize=13, fontweight='bold', loc='left')
for i, v in enumerate(savings):
    ax.text(i, v + 0.1, f'{v:.1f} min', ha='center', fontsize=11, fontweight='bold')
clean_axes(ax)

watermark(fig)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_roxzone.png', dpi=200, facecolor='white')
plt.close()

# ============================================================
# ADDITIONAL ANALYSES FOR PAPER (Phase 1)
# ============================================================

# --- 1a. Descriptive Statistics ---
print("\n[Additional] Descriptive Statistics...")
from scipy.stats import skew, kurtosis as sp_kurtosis

desc_stats = {}
for col in feature_cols + ['Total Time']:
    vals = men[col].dropna().values
    if col == 'Total Time':
        vals_min = vals / 60
    else:
        vals_min = vals / 60 if col in running_cols else vals
    desc_stats[col] = {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)),
        'median': float(np.median(vals)),
        'q25': float(np.percentile(vals, 25)),
        'q75': float(np.percentile(vals, 75)),
        'iqr': float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'skewness': float(skew(vals)),
        'kurtosis': float(sp_kurtosis(vals)),
    }
results['descriptive_stats'] = desc_stats

# --- 1b. Additional Cluster Validation (Davies-Bouldin & Calinski-Harabasz) ---
print("[Additional] Cluster Validation Indices...")
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

db_scores = []
ch_scores = []
K_range_ext = range(2, 11)
for k in K_range_ext:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = km_temp.fit_predict(X_profile_scaled)
    db_scores.append(float(davies_bouldin_score(X_profile_scaled, labels_temp)))
    ch_scores.append(float(calinski_harabasz_score(X_profile_scaled, labels_temp)))

results['cluster_validation'] = {
    'k_range': list(K_range_ext),
    'davies_bouldin': db_scores,
    'calinski_harabasz': ch_scores,
}
print(f"  DB Index (K=2): {db_scores[0]:.3f}")
print(f"  CH Index (K=2): {ch_scores[0]:.1f}")

# --- 1c. Effect Sizes for Japan Comparison (Cohen's d) ---
print("[Additional] Japan Effect Sizes (Cohen's d)...")

jp = men[men['nation'] == 'JP']
world = men[men['nation'] != 'JP']

cohens_d = {}
segment_pvals = []
segment_keys = []
for col in all_segment_cols:
    jp_vals = jp[col].dropna().values
    world_vals = world[col].dropna().values
    if len(jp_vals) < 10:
        continue
    # Cohen's d
    n1, n2 = len(jp_vals), len(world_vals)
    s1, s2 = np.std(jp_vals, ddof=1), np.std(world_vals, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (np.mean(jp_vals) - np.mean(world_vals)) / pooled_std if pooled_std > 0 else 0.0
    # 95% CI for median difference (bootstrap)
    np.random.seed(42)
    n_boot = 1000
    median_diffs = []
    for _ in range(n_boot):
        jp_boot = np.random.choice(jp_vals, size=len(jp_vals), replace=True)
        world_boot = np.random.choice(world_vals, size=min(len(jp_vals), len(world_vals)), replace=True)
        median_diffs.append(np.median(jp_boot) - np.median(world_boot))
    ci_low, ci_high = np.percentile(median_diffs, [2.5, 97.5])

    short_name = col.replace('Running ', 'Run ')
    cohens_d[short_name] = {
        'cohens_d': float(d),
        'median_diff': float(np.median(jp_vals) - np.median(world_vals)),
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
    }
    _, p_val = stats.mannwhitneyu(jp_vals, world_vals, alternative='two-sided')
    segment_keys.append(short_name)
    segment_pvals.append(float(p_val))

if segment_pvals:
    _, q_vals, _, _ = multipletests(segment_pvals, alpha=0.05, method='fdr_bh')
    for key, p_val, q_val in zip(segment_keys, segment_pvals, q_vals):
        cohens_d[key]['p_value'] = float(p_val)
        cohens_d[key]['q_value'] = float(q_val)

results['japan_effect_sizes'] = cohens_d
print(f"  Computed Cohen's d for {len(cohens_d)} segments")

# --- 1d. Model Comparison with RMSE/MAE + Naive Baseline ---
print("[Additional] Model RMSE/MAE + Baseline...")
from sklearn.dummy import DummyRegressor

# Add baseline model
models_extended = {
    'Naive Baseline': DummyRegressor(strategy='mean'),
    'Elastic Net': ElasticNetCV(cv=5, random_state=42, max_iter=5000),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 random_state=42, tree_method='hist'),
    'Gradient Boost': GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_metrics = {}
for name, model in models_extended.items():
    fold_metrics = evaluate_cv_regression(model, X_ml, y_ml, kf, scale=(name == 'Elastic Net'))
    model_metrics[name] = fold_metrics
    print(
        f"  {name}: R²={fold_metrics['r2_mean']:.4f}, "
        f"RMSE={fold_metrics['rmse_mean']:.2f}, MAE={fold_metrics['mae_mean']:.2f}"
    )

results['model_comparison_extended'] = model_metrics

# --- 1d-2. Alternative Targets with Event-Holdout GroupKFold ---
print("[Additional] Alternative targets with GroupKFold...")
groups_event = men['event'].values
n_group_splits = min(5, men['event'].nunique())
gkf = GroupKFold(n_splits=n_group_splits)

# Share-based predictors (segment / total time)
X_share = (men[feature_cols].values / men['Total Time'].values.reshape(-1, 1)).astype(float)

# Log-ratio predictors relative to Run1 (no total-time denominator)
run1 = men['Running 1'].values.astype(float)
run1_safe = np.clip(run1, 1.0, None)
logratio_cols = [c for c in feature_cols if c != 'Running 1']
X_logratio = np.log(np.clip(men[logratio_cols].values.astype(float), 1.0, None)) - np.log(run1_safe.reshape(-1, 1))

# Targets
y_pct_rank = men.groupby('event')['total_min'].rank(pct=True).values.astype(float)
event_median = men.groupby('event')['total_min'].transform('median').values.astype(float)
y_resid = (men['total_min'].values - event_median).astype(float)

xgb_alt = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist'
)
enet_alt = ElasticNetCV(cv=5, random_state=42, max_iter=5000)

alt_metrics = {
    'share_groupkfold': {
        'pct_rank_xgboost': evaluate_groupkfold_regression(xgb_alt, X_share, y_pct_rank, groups_event, n_splits=n_group_splits, scale=False),
        'pct_rank_elasticnet': evaluate_groupkfold_regression(enet_alt, X_share, y_pct_rank, groups_event, n_splits=n_group_splits, scale=True),
        'resid_xgboost': evaluate_groupkfold_regression(xgb_alt, X_share, y_resid, groups_event, n_splits=n_group_splits, scale=False),
        'resid_elasticnet': evaluate_groupkfold_regression(enet_alt, X_share, y_resid, groups_event, n_splits=n_group_splits, scale=True),
    },
    'logratio_groupkfold': {
        'pct_rank_xgboost': evaluate_groupkfold_regression(xgb_alt, X_logratio, y_pct_rank, groups_event, n_splits=n_group_splits, scale=False),
        'pct_rank_elasticnet': evaluate_groupkfold_regression(enet_alt, X_logratio, y_pct_rank, groups_event, n_splits=n_group_splits, scale=True),
        'resid_xgboost': evaluate_groupkfold_regression(xgb_alt, X_logratio, y_resid, groups_event, n_splits=n_group_splits, scale=False),
        'resid_elasticnet': evaluate_groupkfold_regression(enet_alt, X_logratio, y_resid, groups_event, n_splits=n_group_splits, scale=True),
    },
}
results['alternative_targets'] = alt_metrics
print(
    "  Share/GroupKFold pct-rank XGBoost R²="
    f"{alt_metrics['share_groupkfold']['pct_rank_xgboost']['r2_mean']:.3f}"
)
print(
    "  Logratio/GroupKFold pct-rank XGBoost R²="
    f"{alt_metrics['logratio_groupkfold']['pct_rank_xgboost']['r2_mean']:.3f}"
)

# --- 1d-3. Event Fixed Effects Check (pooled random CV; descriptive) ---
print("[Additional] Event fixed-effects check (pooled random CV)...")
X_base = men[feature_cols].values.astype(float)
event_dummies = pd.get_dummies(men['event'], drop_first=True).values.astype(float)
X_with_fe = np.hstack([X_base, event_dummies])
y_total = men['total_min'].values.astype(float)

ridge_base = Ridge(alpha=1.0, random_state=42)
ridge_fe = Ridge(alpha=1.0, random_state=42)

metrics_base = evaluate_cv_regression(ridge_base, X_base, y_total, kf, scale=True)
metrics_fe = evaluate_cv_regression(ridge_fe, X_with_fe, y_total, kf, scale=True)
results['event_fixed_effects_check'] = {
    'ridge_without_event_dummies': metrics_base,
    'ridge_with_event_dummies': metrics_fe,
}
print(f"  Ridge no FE: R²={metrics_base['r2_mean']:.3f}, MAE={metrics_base['mae_mean']:.2f}")
print(f"  Ridge + FE:  R²={metrics_fe['r2_mean']:.3f}, MAE={metrics_fe['mae_mean']:.2f}")

# --- 1e. VIF (Variance Inflation Factor) ---
print("[Additional] VIF...")
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = StandardScaler().fit_transform(men[feature_cols].values)
X_vif_const = sm.add_constant(X_vif)
vif_data = {}
for i, col_name in enumerate(feature_names):
    vif_val = variance_inflation_factor(X_vif_const, i + 1)  # +1 for constant
    vif_data[col_name] = float(vif_val)
    if vif_val > 10:
        print(f"  WARNING: {col_name} VIF={vif_val:.1f}")

results['vif'] = vif_data
print(f"  Max VIF: {max(vif_data.values()):.1f} ({max(vif_data, key=vif_data.get)})")

# --- 1f. Bootstrap CI for Top SHAP Values ---
print("[Additional] Bootstrap SHAP CI (100 iterations)...")

np.random.seed(42)
n_bootstrap = 100
n_subsample = min(10000, len(X_ml))
top_5_idx = np.argsort(shap_importance)[-5:][::-1]
top_5_names = [feature_names[i] for i in top_5_idx]

bootstrap_shap = {name: [] for name in top_5_names}

for b in range(n_bootstrap):
    boot_idx = np.random.choice(len(X_ml), size=n_subsample, replace=True)
    X_boot = X_ml[boot_idx]
    y_boot = y_ml[boot_idx]
    xgb_boot = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=b,
        tree_method='hist'
    )
    xgb_boot.fit(X_boot, y_boot)
    explainer_boot = shap.TreeExplainer(xgb_boot)
    shap_boot = explainer_boot.shap_values(X_boot)
    shap_imp_boot = np.abs(shap_boot).mean(axis=0)
    for fi, name in zip(top_5_idx, top_5_names):
        bootstrap_shap[name].append(float(shap_imp_boot[fi]))
    if (b + 1) % 20 == 0:
        print(f"  Bootstrap {b + 1}/{n_bootstrap}")

bootstrap_results = {}
for name in top_5_names:
    vals = bootstrap_shap[name]
    bootstrap_results[name] = {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'ci_95_low': float(np.percentile(vals, 2.5)),
        'ci_95_high': float(np.percentile(vals, 97.5)),
    }
    print(f"  {name}: {np.mean(vals):.3f} [{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]")

results['bootstrap_shap'] = bootstrap_results

# --- 1g. Winsorization Sensitivity Check for SHAP Ranking ---
print("[Additional] Winsorization sensitivity check...")
n_sens = min(20000, len(X_ml))
idx_sens = np.random.RandomState(42).choice(len(X_ml), n_sens, replace=False)
X_sens = X_ml[idx_sens].astype(float)
y_sens = y_ml[idx_sens].astype(float)

X_wins = X_sens.copy()
for j in range(X_wins.shape[1]):
    lo, hi = np.percentile(X_wins[:, j], [1, 99])
    X_wins[:, j] = np.clip(X_wins[:, j], lo, hi)

xgb_raw = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist'
)
xgb_win = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist'
)
xgb_raw.fit(X_sens, y_sens)
xgb_win.fit(X_wins, y_sens)

shap_raw = shap.TreeExplainer(xgb_raw).shap_values(X_sens)
shap_win = shap.TreeExplainer(xgb_win).shap_values(X_wins)
imp_raw = np.abs(shap_raw).mean(axis=0)
imp_win = np.abs(shap_win).mean(axis=0)

rho_sens, _ = stats.spearmanr(imp_raw, imp_win)
top5_raw = set(np.argsort(imp_raw)[-5:])
top5_win = set(np.argsort(imp_win)[-5:])
top5_overlap = len(top5_raw.intersection(top5_win))

results['winsorization_sensitivity'] = {
    'n_subsample': int(n_sens),
    'clip_percentiles': [1, 99],
    'spearman_rho_shap_importance': float(rho_sens),
    'top5_overlap': int(top5_overlap),
    'top5_raw_features': [feature_names[i] for i in np.argsort(imp_raw)[-5:][::-1]],
    'top5_winsorized_features': [feature_names[i] for i in np.argsort(imp_win)[-5:][::-1]],
}
print(f"  Spearman rho={rho_sens:.3f}, top-5 overlap={top5_overlap}/5")

# ============================================================
# Save results JSON
# ============================================================
# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

with open(f'{OUTPUT_DIR}/analysis_results.json', 'w') as f:
    json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 60)
