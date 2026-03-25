#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon
import os
import itertools
import math
import warnings
import pickle # 객체 저장을 위해 추가
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm

# [수정] SHAP 임포트 예외 처리
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ 'shap' module not found. Analysis skipped, but artifacts will be saved.")

# 경고 무시
warnings.filterwarnings('ignore')

# ===================================================================
# 1. Configuration
# ===================================================================
MAIN_OUTPUT_DIR = "analysis_result_transformer_final_triangle_learning_1e-4"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

FILE_PATH = "PCR_NGS_Data.xlsx"

TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
CORRELATION_THRESHOLD = 0.8
TARGET_VAR_FOR_CLUSTERING = 'Halitosis'

TRANSFORMER_GRID = {
    'k': [16, 32, 64],
    'heads': [2, 4],
    'depth': [2, 3, 4], 
    'lr': [1e-4],
    'epochs': [100]
}

BATCH_SIZE = 8
OUTER_FOLDS = 3
INNER_FOLDS = 3
RANDOM_STATE = 42
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Transformer Architecture
# ===================================================================
class SelfAttention(nn.Module):
    def __init__(self, k, heads=4):
        super().__init__()
        assert k % heads == 0
        self.heads, self.d_k = heads, k // heads
        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h, d_k = self.heads, self.d_k
        Q = self.toqueries(x).view(b,t,h,d_k).transpose(1,2)
        K = self.tokeys(x).view(b,t,h,d_k).transpose(1,2)
        V = self.tovalues(x).view(b,t,h,d_k).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V).transpose(1,2).contiguous().view(b,t,h*d_k)
        
        return self.unifyheads(out), attn_weights

class TransformerEncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))

    def forward(self, x):
        attended, weights = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x), weights

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, k, heads, depth, dropout=0.1): 
        super().__init__()
        self.embedding_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(1, k), nn.SiLU(), nn.Linear(k, k)) 
            for _ in range(seq_len)
        ])
        self.transformer_encoder = nn.ModuleList([TransformerEncoderBlock(k, heads) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(k, 1)

    def forward(self, x):
        embedded_list = [self.embedding_blocks[i](x[:, i].unsqueeze(-1)) for i in range(x.size(1))]
        x_emb = torch.stack(embedded_list, dim=1)
        
        attn_maps = []
        for layer in self.transformer_encoder:
            x_emb, weights = layer(x_emb)
            attn_maps.append(weights)
            
        x_pooled = x_emb.mean(dim=1)
        return self.fc(self.dropout(x_pooled)).squeeze(-1), attn_maps

# ===================================================================
# 3. Data Processing & Cluster Logic
# ===================================================================
def format_bacterial_name(raw_name):
    clean = raw_name.replace("PCR_", "").replace("PCR", "").replace("__", " ").replace("_", " ").strip()
    parts = clean.split()
    if len(parts) >= 2:
        genus = parts[0][0].upper()
        species = parts[1]
        return f"{genus}. {species}"
    else:
        return clean

def load_and_preprocess_data(file_path):
    print(f"📂 Loading data from '{file_path}'...")
    try:
        if file_path.endswith('.csv'): df_raw = pd.read_csv(file_path, header=None)
        else: df_raw = pd.read_excel(file_path, header=None) 
        
        header_idx = -1
        for i, row in df_raw.iterrows():
            if 'Sex' in str(row.values) and 'Age' in str(row.values): header_idx = i; break
        if header_idx == -1: header_idx = 6
        
        df_raw.columns = df_raw.iloc[header_idx]
        df = df_raw.iloc[header_idx+1:].reset_index(drop=True)
        
        selected = {}
        target_map = {'H2S': 'H2S_ppb', 'CH3SH': 'CH3SH_ppb', 'VSCs': 'VSCs_ppb'}
        pcr_columns_clean = []
        
        for col in df.columns:
            c_str = str(col).strip()
            if "PCR" in c_str:
                formatted_name = format_bacterial_name(c_str)
                selected[formatted_name] = pd.to_numeric(df[col], errors='coerce')
                pcr_columns_clean.append(formatted_name)
            elif 'Halitosis' in c_str:
                selected['Halitosis'] = pd.to_numeric(df[col], errors='coerce')
            else:
                for k, v in target_map.items():
                    if k in c_str: selected[v] = pd.to_numeric(df[col], errors='coerce'); break
        
        df_clean = pd.DataFrame(selected)
        df_final = df_clean.dropna(subset=['Halitosis'])
        
        return df_final, pcr_columns_clean
    except Exception as e:
        print(f"❌ Error: {e}"); return None, None

def get_split_correlation_clusters(df, pcr_features, threshold):
    """Halitosis 유무에 따라 상관관계 클러스터 생성"""
    df_hal = df[df[TARGET_VAR_FOR_CLUSTERING] == 1]
    df_non = df[df[TARGET_VAR_FOR_CLUSTERING] == 0]
    
    def _find(sub, label):
        if len(sub) < 5: return {}
        G = nx.from_pandas_adjacency(sub[pcr_features].corr().abs().fillna(0))
        G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w < threshold])
        clusters = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
        return {f"Cluster_{label}_{i+1}": g for i, g in enumerate(clusters)}

    clusters = {}
    clusters.update(_find(df_hal, "Hal"))    
    clusters.update(_find(df_non, "NonHal")) 
    return clusters

# ===================================================================
# 4. Training Engine
# ===================================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        out, _ = model(xb.to(DEVICE))
        loss = criterion(out, yb.to(DEVICE))
        loss.backward()
        optimizer.step()

def run_inner_grid_search(X, y):
    keys, values = zip(*TRANSFORMER_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_loss = float('inf'); best_config = combinations[0]
    inner_kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for config in combinations:
        if config['k'] % config['heads'] != 0: continue
        fold_losses = []
        for tr, val in inner_kf.split(X):
            sc_x = StandardScaler().fit(X[tr])
            X_tr_s, X_val_s = sc_x.transform(X[tr]), sc_x.transform(X[val])
            sc_y = StandardScaler().fit(y[tr].reshape(-1, 1))
            y_tr_s = sc_y.transform(y[tr].reshape(-1, 1)).flatten()
            y_val_s = sc_y.transform(y[val].reshape(-1, 1)).flatten()
            
            ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            
            model = TransformerRegressor(X.shape[1], config['k'], config['heads'], config['depth']).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            criterion = nn.MSELoss()
            
            for _ in range(config['epochs']): train_epoch(model, loader, optimizer, criterion)
            
            model.eval()
            with torch.no_grad():
                val_out, _ = model(torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE))
                loss = criterion(val_out, torch.tensor(y_val_s, dtype=torch.float32).to(DEVICE)).item()
            fold_losses.append(loss)
            
        if np.mean(fold_losses) < best_loss:
            best_loss = np.mean(fold_losses); best_config = config
    return best_config

def run_nested_cv_transformer(feature_names, X_df, y_full):
    X = X_df[feature_names].values
    if X.ndim == 1: X = X.reshape(-1, 1)
        
    y = y_full
    
    outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_r2s = []
    y_trues_all, y_preds_all = [], []
    
    # Store representative configuration (from the last fold)
    final_config = None
    
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        best_config = run_inner_grid_search(X_tr, y_tr)
        final_config = best_config # Keep update
        
        sc_x = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc_x.transform(X_tr), sc_x.transform(X_te)
        sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).flatten()
        
        ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = TransformerRegressor(X.shape[1], best_config['k'], best_config['heads'], best_config['depth']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])
        criterion = nn.MSELoss()
        
        for _ in range(best_config['epochs']): train_epoch(model, loader, optimizer, criterion)
        
        model.eval()
        with torch.no_grad():
            preds_scaled, _ = model(torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE))
            preds_scaled = preds_scaled.cpu().numpy()
        preds = sc_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        fold_r2 = r2_score(y_te, preds)
        fold_r2s.append(fold_r2)
        y_trues_all.extend(y_te)
        y_preds_all.extend(preds)
        
    return {
        'Mean_R2': np.mean(fold_r2s),
        'y_true': np.array(y_trues_all),
        'y_pred': np.array(y_preds_all),
        'Best_Config': final_config
    }

def train_final_model(X_df, features, y_full, config):
    """
    Train a model on the FULL dataset using the best configuration found.
    Used for SHAP analysis and saving artifacts.
    """
    X = X_df[features].values
    if X.ndim == 1: X = X.reshape(-1, 1)
    
    # 1. Scaler Fitting
    scaler_x = StandardScaler().fit(X)
    X_s = scaler_x.transform(X)
    
    scaler_y = StandardScaler().fit(y_full.reshape(-1, 1))
    y_s = scaler_y.transform(y_full.reshape(-1, 1)).flatten()
    
    # 2. Model Training
    ds = TensorDataset(torch.tensor(X_s, dtype=torch.float32), torch.tensor(y_s, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TransformerRegressor(X.shape[1], config['k'], config['heads'], config['depth']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(config['epochs']):
        train_epoch(model, loader, optimizer, criterion)
        
    model.eval()
    return model, scaler_x, scaler_y

# ===================================================================
# 5. SHAP Analysis & Artifact Saving
# ===================================================================
def save_artifacts_for_later(model, scaler_x, X_df, features, config, model_name, target_name, output_path):
    """
    SHAP 분석 및 모델 재사용을 위한 모든 아티팩트 저장 (pickle & pytorch)
    """
    artifact_dir = os.path.join(output_path, "saved_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Safe filename
    safe_name = model_name.replace(".", "").replace(" ", "_")
    
    # 1. Model State Dict
    torch.save(model.state_dict(), os.path.join(artifact_dir, f"model_{safe_name}.pth"))
    
    # 2. Scaler & Config & Feature Names (Pickle)
    meta_data = {
        'scaler_x': scaler_x,
        'features': features,
        'config': config,
        'model_name': model_name,
        'target': target_name
    }
    
    with open(os.path.join(artifact_dir, f"meta_{safe_name}.pkl"), 'wb') as f:
        pickle.dump(meta_data, f)
        
    # 3. Background Data (Subset for SHAP) - CSV
    # SHAP KernelExplainer needs background data. We save a sample.
    sample_size = min(100, len(X_df))
    bg_data = X_df[features].sample(sample_size, random_state=42)
    bg_data.to_csv(os.path.join(artifact_dir, f"background_data_{safe_name}.csv"), index=False)
    
    print(f"   💾 Artifacts saved in: {artifact_dir}")

def run_shap_analysis(model, X_df, features, scaler, model_name, target_name, output_path):
    if not SHAP_AVAILABLE:
        print(f"   ⚠️ SHAP module missing. Skipping plot for {model_name}.")
        return

    print(f"   🤖 Running SHAP analysis for {model_name}...")
    
    X_val = X_df[features].values
    X_scaled = scaler.transform(X_val)
    
    model.eval()
    
    # Wrapper for SHAP
    def predict_wrapper(data):
        with torch.no_grad():
            t_data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
            out, _ = model(t_data)
            return out.cpu().numpy().flatten()

    # Use kmeans for background summary
    background = shap.kmeans(X_scaled, 10) 
    explainer = shap.KernelExplainer(predict_wrapper, background)
    
    # Calculate SHAP values (subset)
    shap_values = explainer.shap_values(X_scaled[:50], nsamples=50) # Time-consuming, limiting samples
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val[:50], feature_names=features, plot_type="bar", show=False)
    
    plt.title(f"SHAP Feature Importance (Transformer)\n{model_name} - {target_name}", fontsize=14, fontweight='bold')
    plt.xlabel("mean(|SHAP value|)", fontsize=12, fontweight='bold')
    
    safe_name = model_name.replace(".", "").replace(" ", "_")
    save_path = os.path.join(output_path, f"SHAP_Bar_{safe_name}.png")
    plt.savefig(save_path, dpi=700, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 SHAP plot saved to {save_path}")

# ===================================================================
# 6. Statistical Test & Visualization (Lower Triangle)
# ===================================================================
def calculate_pairwise_statistics_and_plot(results_list, target_name, output_path):
    n = len(results_list)
    display_names = []
    for r in results_list:
        name = r['ModelName']
        if "Cluster" in name:
            name = name.replace("Cluster_", "").replace("Halitosis", "Hal").replace("Healthy", "NonHal")
        else:
            name = name.replace("Single_", "") 
        display_names.append(name)

    p_values = np.zeros((n, n))
    
    print(f"   📊 Calculating Wilcoxon Tests for {n} models...")
    
    for i in range(n):
        for j in range(n):
            if i == j: p_values[i, j] = 1.0; continue
            if not np.allclose(results_list[i]['y_true'], results_list[j]['y_true']): p_values[i, j] = 1.0; continue

            se_a = (results_list[i]['y_true'] - results_list[i]['y_pred']) ** 2
            se_b = (results_list[j]['y_true'] - results_list[j]['y_pred']) ** 2
            
            try: _, p = wilcoxon(se_a, se_b); p_values[i, j] = p
            except: p_values[i, j] = 1.0

    # Save P-values CSV
    df_pvals = pd.DataFrame(p_values, index=[r['ModelName'] for r in results_list], columns=[r['ModelName'] for r in results_list])
    df_pvals.to_csv(os.path.join(output_path, f"Stats_Pvalues_{target_name}.csv"))

    # Heatmap
    plt.figure(figsize=(28, 28))
    mask = np.triu(np.ones_like(p_values, dtype=bool), k=0)
    sig_matrix = (p_values < 0.05).astype(int)
    cmap = ListedColormap(['whitesmoke', '#FFF700'])

    df_annot = pd.DataFrame(p_values)
    annot_labels = df_annot.applymap(lambda x: f"{x:.4f}")

    ax = sns.heatmap(sig_matrix, mask=mask, cmap=cmap, annot=annot_labels, fmt="", 
                     cbar=False, linewidths=0.5, linecolor='lightgray', square=True,
                     annot_kws={"size": 14, "color": "black"})

    ax.set_xticklabels(display_names, rotation=45, ha='right', fontweight='bold', fontsize=18)
    ax.set_yticklabels(display_names, rotation=0, ha='right', fontweight='bold', fontsize=18)

    for t in ax.texts:
        try:
            if float(t.get_text()) < 0.05: t.set_weight('bold')
        except: pass

    plt.title(f"Pair-wise Comparison (Transformer) - {target_name}\n(Yellow: P < 0.05)", fontsize=40, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"pairwise_comparison_heatmap_{target_name}.png"), dpi=700)
    plt.close()

# ===================================================================
# 7. Main Execution
# ===================================================================
def run_analysis():
    if not os.path.exists(FILE_PATH): print("❌ File not found."); return

    df, pcr_cols = load_and_preprocess_data(FILE_PATH)
    if df is None: return

    # 1. Models Setup (Unified)
    print("🔍 Generating Models...")
    clusters = get_split_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
    models_to_evaluate = clusters.copy()
    
    for pcr_col in pcr_cols:
        model_name = f"Single_{pcr_col}"
        models_to_evaluate[model_name] = [pcr_col]
        
    print(f"   -> Total models: {len(models_to_evaluate)}")

    all_metrics_summary = []
    
    # 2. Iterate Targets
    for target in TARGET_GASES:
        if target not in df.columns: continue
        
        print(f"\n{'='*50}\n🔥 Analyzing Target: {target} (Transformer)\n{'='*50}")
        
        target_output_dir = os.path.join(MAIN_OUTPUT_DIR, target)
        os.makedirs(target_output_dir, exist_ok=True)

        df_tgt = df.dropna(subset=[target])
        y = df_tgt[target].values
        
        target_results = []
        
        # 3. Train & Evaluate
        for model_name, feats in tqdm(models_to_evaluate.items(), desc=f"Scanning ({target})"):
            res = run_nested_cv_transformer(feats, df_tgt, y)
            
            target_results.append({
                'ModelName': model_name, 'Features': feats,
                'R2': res['Mean_R2'], 'y_true': res['y_true'], 'y_pred': res['y_pred'],
                'Best_Config': res['Best_Config']
            })

        # 4. Save Predictions
        for res in target_results:
            pred_df = pd.DataFrame({'y_true': res['y_true'], 'y_pred': res['y_pred']})
            safe_name = res['ModelName'].replace(".", "").replace(" ", "_")
            pred_df.to_csv(os.path.join(target_output_dir, f"Pred_{safe_name}_Transformer.csv"), index=False)

        # 5. Statistical Test
        if len(target_results) > 1:
            calculate_pairwise_statistics_and_plot(target_results, target, target_output_dir)

        # 6. Final Model Training & SHAP / Artifact Saving (Best Model)
        if len(target_results) > 0:
            sorted_res = sorted(target_results, key=lambda x: x['R2'], reverse=True)
            best_res = sorted_res[0]
            
            print(f"   🏆 Best Model: {best_res['ModelName']} | R2: {best_res['R2']:.3f}")
            
            # Re-train on Full Dataset for SHAP/Saving
            print("   ⚙️ Retraining best model on full dataset for analysis...")
            final_model, scaler_x, _ = train_final_model(df_tgt, best_res['Features'], y, best_res['Best_Config'])
            
            # Save Artifacts for Later
            save_artifacts_for_later(
                final_model, scaler_x, df_tgt, best_res['Features'], best_res['Best_Config'],
                best_res['ModelName'], target, target_output_dir
            )

            # Run SHAP (if Single species -> skip plot, but still save artifacts above)
            if len(best_res['Features']) > 1:
                run_shap_analysis(
                    final_model, df_tgt, best_res['Features'], scaler_x,
                    best_res['ModelName'], target, target_output_dir
                )
            else:
                print("   ℹ️ Single species model: SHAP plot skipped.")

        # 7. Summary Collection
        for res in target_results:
            all_metrics_summary.append({
                'Target': target,
                'ModelName': res['ModelName'],
                'Num_Features': len(res['Features']),
                'R2': res['R2']
            })

    # 8. Save Final Summary
    summary_df = pd.DataFrame(all_metrics_summary)
    summary_path = os.path.join(MAIN_OUTPUT_DIR, "All_Models_Transformer_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Analysis Complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    run_analysis()
#%%