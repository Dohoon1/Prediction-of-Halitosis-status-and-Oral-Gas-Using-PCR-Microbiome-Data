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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix, recall_score
from scipy import stats
from imblearn.over_sampling import SMOTE
import os
import itertools
import warnings
import pickle
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ===================================================================
# 1. Configuration
# ===================================================================
MAIN_OUTPUT_DIR = "Analysis_Result_Halitosis_NewMethod_Combined"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

FILE_PATH = "PCR_NGS_Data.xlsx"
TARGET_VAR = 'Halitosis'  # 0 or 1
CORRELATION_THRESHOLD = 0.4

# Clinical Variables
CLINICAL_VAR_NAMES = ['Sex', 'Age', 'Smoking', 'Oral hygiene']
CONTINUOUS_VARS = ['Age']

# Hyperparameters
LOGISTIC_GRID = {'lr': [0.1, 0.01, 0.001], 'epochs': [500, 1000]}
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
DEVICE = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# Colors for Plotting
DISTINCT_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Data Loading & Preprocessing (Robust)
# ===================================================================
def format_bacterial_name(raw_name):
    clean = raw_name.replace("PCR_", "").replace("PCR", "").replace("__", " ").replace("_", " ").strip()
    parts = clean.split()
    if len(parts) >= 2: return f"{parts[0][0].upper()}. {parts[1]}"
    else: return clean

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
        
        selected = {'Halitosis': pd.to_numeric(df['Halitosis'], errors='coerce')}
        pcr_columns_clean = []
        
        # 1. PCR Bacteria
        for col in df.columns:
            if "PCR" in str(col):
                formatted_name = format_bacterial_name(str(col))
                selected[formatted_name] = pd.to_numeric(df[col], errors='coerce')
                pcr_columns_clean.append(formatted_name)
        
        # 2. Clinical Variables
        clinical_cols_found = []
        for clin_var in CLINICAL_VAR_NAMES:
            found_col = [c for c in df.columns if clin_var.lower() in str(c).lower()]
            if found_col:
                col_name = found_col[0]
                series = df[col_name]
                
                # 연속형 변수 (Age)
                if clin_var in CONTINUOUS_VARS:
                    selected[clin_var] = pd.to_numeric(series, errors='coerce')
                # 범주형 변수
                else:
                    if series.dtype == object:
                        mapping = {}
                        for v in series.dropna().unique():
                            v_str = str(v).lower().strip()
                            if v_str in ['male', 'm', 'no', 'n', 'non-smoker']: mapping[v] = 0
                            elif v_str in ['female', 'f', 'yes', 'y', 'smoker']: mapping[v] = 1
                            else: 
                                try: mapping[v] = float(v)
                                except: pass
                        series_mapped = series.map(mapping)
                        # 매핑 안 된 값(Oral hygiene 등)은 Factorize
                        if series_mapped.isnull().any():
                             if clin_var == 'Oral hygiene' or series_mapped.isnull().sum() > 0:
                                codes, _ = pd.factorize(series)
                                series_mapped = pd.Series(codes, index=series.index)
                        selected[clin_var] = pd.to_numeric(series_mapped, errors='coerce')
                    else:
                        selected[clin_var] = pd.to_numeric(series, errors='coerce')
                clinical_cols_found.append(clin_var)

        df_clean = pd.DataFrame(selected).dropna(subset=['Halitosis'] + clinical_cols_found)
        df_clean['Halitosis'] = df_clean['Halitosis'].astype(int) # Ensure Target is Integer
        
        print(f"   -> Data Shape: {df_clean.shape}")
        return df_clean, pcr_columns_clean, clinical_cols_found

    except Exception as e: print(f"Error: {e}"); return None, None, None

# --- [Changed] Combined Clustering Logic ---
def get_combined_correlation_clusters(df, pcr_features, threshold):
    """전체 환자 대상 상관관계 클러스터링"""
    G = nx.from_pandas_adjacency(df[pcr_features].corr().abs().fillna(0))
    G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w < threshold])
    clusters_list = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
    # Naming: Cluster_Combined_1...
    return {f"Cluster_Combined_{i+1}": g for i, g in enumerate(clusters_list)}

def visualize_cluster(df, pcr_features, threshold, output_dir):
    plt.figure(figsize=(10, 8))
    G = nx.from_pandas_adjacency(df[pcr_features].corr().abs().fillna(0))
    G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w < threshold])
    pos = nx.spring_layout(G, k=0.3, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8)
    plt.title(f"Combined Correlation Cluster (Thresh={threshold})")
    plt.savefig(os.path.join(output_dir, "Cluster_Network_Combined.png"))
    plt.close()

# ===================================================================
# 3. Models (Logistic & Transformer)
# ===================================================================
class LogisticRegressionNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x): return self.linear(x).squeeze(-1)

# Transformer Components
class SelfAttention(nn.Module):
    def __init__(self, k, heads=4):
        super().__init__()
        assert k % heads == 0
        self.heads, self.d_k = heads, k // heads
        self.tokeys = nn.Linear(k, k, bias=False); self.toqueries = nn.Linear(k, k, bias=False); self.tovalues = nn.Linear(k, k, bias=False)
        self.unifyheads = nn.Linear(k, k)
    def forward(self, x):
        b, t, k = x.size(); h, d_k = self.heads, self.d_k
        Q = self.toqueries(x).view(b,t,h,d_k).transpose(1,2); K = self.tokeys(x).view(b,t,h,d_k).transpose(1,2); V = self.tovalues(x).view(b,t,h,d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1))/np.sqrt(d_k); attn = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(b,t,h*d_k)
        return self.unifyheads(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attn = SelfAttention(k, heads); self.norm1=nn.LayerNorm(k); self.norm2=nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))
    def forward(self, x):
        a = self.attn(x); x = self.norm1(a + x); f = self.ff(x); return self.norm2(f + x)

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, k, heads, depth, dropout=0.1): 
        super().__init__()
        self.embedding_blocks = nn.ModuleList([nn.Sequential(nn.Linear(1, k), nn.SiLU(), nn.Linear(k, k)) for _ in range(seq_len)])
        self.enc = nn.ModuleList([TransformerEncoderBlock(k, heads) for _ in range(depth)])
        self.fc = nn.Linear(k, 1); self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        emb = torch.stack([self.embedding_blocks[i](x[:,i].unsqueeze(-1)) for i in range(x.size(1))], dim=1)
        for l in self.enc: emb = l(emb)
        return self.fc(self.dropout(emb.mean(dim=1))).squeeze(-1)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(); out = model(xb.to(DEVICE))
        loss = criterion(out, yb.to(DEVICE)); loss.backward(); optimizer.step()

# ===================================================================
# 4. Nested CV Pipeline (Unified)
# ===================================================================
def run_nested_cv(model_type, features, clin_cols, df, y_full):
    # Combine Features
    cols = list(dict.fromkeys(clin_cols + features)) # Remove duplicates, keep order
    X = df[cols].values
    
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_aucs, y_true_all, y_prob_all = [], [], []
    
    for tr_idx, te_idx in outer_skf.split(X, y_full):
        X_tr, X_te = X[tr_idx], X[te_idx]; y_tr, y_te = y_full[tr_idx], y_full[te_idx]
        
        # Scaling
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        
        # SMOTE
        try: X_tr_res, y_tr_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_s, y_tr)
        except: X_tr_res, y_tr_res = X_tr_s, y_tr
        
        # Inner Loop (Grid Search)
        inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        if model_type == 'Logistic': grid = LOGISTIC_GRID; combos = [{'lr': p} for p in grid['lr']]
        else: keys, values = zip(*TRANSFORMER_GRID.items()); combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
        best_auc = -1; best_config = combos[0]
        
        for config in combos:
            if model_type == 'Transformer' and config['k'] % config['heads'] != 0: continue
            
            val_aucs = []
            for i_tr, i_val in inner_skf.split(X_tr_res, y_tr_res):
                X_it, y_it = X_tr_res[i_tr], y_tr_res[i_tr]
                X_iv, y_iv = X_tr_res[i_val], y_tr_res[i_val]
                
                ds_it = TensorDataset(torch.tensor(X_it, dtype=torch.float32), torch.tensor(y_it, dtype=torch.float32))
                ld_it = DataLoader(ds_it, batch_size=BATCH_SIZE, shuffle=True)
                
                if model_type == 'Logistic':
                    m = LogisticRegressionNN(X.shape[1]).to(DEVICE)
                    opt = optim.Adam(m.parameters(), lr=config['lr'])
                else:
                    m = TransformerClassifier(X.shape[1], config['k'], config['heads'], config['depth']).to(DEVICE)
                    opt = optim.Adam(m.parameters(), lr=config['lr'])
                
                crit = nn.BCEWithLogitsLoss()
                epochs = config['epochs'] if 'epochs' in config else 100
                
                for _ in range(epochs): train_epoch(m, ld_it, opt, crit)
                
                m.eval()
                with torch.no_grad():
                    logits = m(torch.tensor(X_iv, dtype=torch.float32).to(DEVICE))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    try: val_aucs.append(roc_auc_score(y_iv, probs))
                    except: val_aucs.append(0.5)
            
            if np.mean(val_aucs) > best_auc: best_auc = np.mean(val_aucs); best_config = config
            
        # Final Training
        ds_tr = TensorDataset(torch.tensor(X_tr_res, dtype=torch.float32), torch.tensor(y_tr_res, dtype=torch.float32))
        ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
        
        if model_type == 'Logistic':
            fm = LogisticRegressionNN(X.shape[1]).to(DEVICE)
            f_opt = optim.Adam(fm.parameters(), lr=best_config['lr'])
        else:
            fm = TransformerClassifier(X.shape[1], best_config['k'], best_config['heads'], best_config['depth']).to(DEVICE)
            f_opt = optim.Adam(fm.parameters(), lr=best_config['lr'])
            
        f_crit = nn.BCEWithLogitsLoss()
        epochs = best_config['epochs'] if 'epochs' in best_config else 100
        for _ in range(epochs): train_epoch(fm, ld_tr, f_opt, f_crit)
        
        fm.eval()
        with torch.no_grad():
            logits = fm(torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            
        y_true_all.extend(y_te); y_prob_all.extend(probs)
        try: fold_aucs.append(roc_auc_score(y_te, probs))
        except: fold_aucs.append(0.5)
        
    return {
        'Mean_AUC': np.mean(fold_aucs), 
        'Std_AUC': np.std(fold_aucs),
        'y_true': np.array(y_true_all), 
        'y_prob': np.array(y_prob_all),
        'Best_Config': best_config
    }

# ===================================================================
# 5. Helper & Visualization
# ===================================================================
def save_results(name, model_type, res):
    safe_name = name.replace(".", "").replace(" ", "_")
    # Save Prediction
    pd.DataFrame({'y_true': res['y_true'], 'y_prob': res['y_prob']}).to_csv(
        os.path.join(MAIN_OUTPUT_DIR, f"Pred_{safe_name}_{model_type}.csv"), index=False
    )

def plot_roc_top6(summary_df):
    # Cluster Top 3 + Single Top 3
    df = summary_df.copy()
    cluster_mask = df['ModelName'].str.contains("Cluster")
    single_mask = df['ModelName'].str.contains("Single")
    
    top3_cluster = df[cluster_mask].sort_values(by='Mean_AUC', ascending=False).head(3)
    top3_single = df[single_mask].sort_values(by='Mean_AUC', ascending=False).head(3)
    
    top6 = pd.concat([top3_cluster, top3_single])
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    for i, (_, row) in enumerate(top6.iterrows()):
        m_name, m_type = row['ModelName'], row['Type']
        safe_name = m_name.replace(".", "").replace(" ", "_")
        pred_file = os.path.join(MAIN_OUTPUT_DIR, f"Pred_{safe_name}_{m_type}.csv")
        
        if os.path.exists(pred_file):
            dat = pd.read_csv(pred_file)
            fpr, tpr, _ = roc_curve(dat['y_true'], dat['y_prob'])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{m_name} ({m_type}) AUC={auc_val:.3f}")
            
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('Top 6 Halitosis Classification Models (Combined Method)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(MAIN_OUTPUT_DIR, "ROC_Top6_Halitosis.png"), dpi=300)
    plt.close()

# DeLong Test Helpers
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=np.float64); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=np.float64); T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count; n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=np.float64); ty = np.empty([k, n], dtype=np.float64); tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r, :] = compute_midrank(predictions_sorted_transposed[r, m:])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n; v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10); delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(y_true, prob_A, prob_B):
    if len(y_true) != len(prob_A) or len(y_true) != len(prob_B): return 0, 1.0
    mask_pos = y_true == 1; mask_neg = y_true == 0
    if np.sum(mask_pos) == 0 or np.sum(mask_neg) == 0: return 0, 1.0 
    X_pos = np.vstack((prob_A[mask_pos], prob_B[mask_pos]))
    X_neg = np.vstack((prob_A[mask_neg], prob_B[mask_neg]))
    preds_sorted = np.hstack((X_pos, X_neg))
    aucs, delongcov = fast_delong(preds_sorted, np.sum(y_true == 1))
    l = np.array([1, -1]); diff = np.diff(aucs)
    sigma = np.sqrt(np.dot(np.dot(l, delongcov), l.T))
    if sigma == 0: return aucs, 1.0
    z = np.abs(diff) / sigma
    p_value = 2 * (1 - stats.norm.cdf(z))[0]
    return aucs, p_value

def plot_delong_heatmap(summary_df):
    # Cluster Top 3 + Single Top 3
    df = summary_df.copy()
    cluster_mask = df['ModelName'].str.contains("Cluster")
    single_mask = df['ModelName'].str.contains("Single")
    top6 = pd.concat([df[cluster_mask].sort_values(by='Mean_AUC', ascending=False).head(3),
                      df[single_mask].sort_values(by='Mean_AUC', ascending=False).head(3)])
    
    models = []
    for _, row in top6.iterrows():
        safe_name = row['ModelName'].replace(".", "").replace(" ", "_")
        pred_file = os.path.join(MAIN_OUTPUT_DIR, f"Pred_{safe_name}_{row['Type']}.csv")
        if os.path.exists(pred_file):
            d = pd.read_csv(pred_file)
            models.append({'name': f"{row['ModelName']}\n({row['Type']})", 'y_true': d['y_true'].values, 'y_prob': d['y_prob'].values})
            
    n = len(models)
    if n < 2: return
    p_vals = np.zeros((n, n))
    labels = [m['name'] for m in models]
    
    for i in range(n):
        for j in range(n):
            if i == j: p_vals[i, j] = 1.0
            else: 
                _, p = calc_pvalue(models[i]['y_true'], models[i]['y_prob'], models[j]['y_prob'])
                p_vals[i, j] = p
                
    plt.figure(figsize=(10, 8), dpi=300)
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i==j: annot[i,j]="-"
            elif p_vals[i,j]<0.001: annot[i,j]="<.001"
            else: annot[i,j]=f"{p_vals[i,j]:.3f}"
            
    sns.heatmap(p_vals, annot=annot, fmt="", cmap=ListedColormap(['#FFD700', '#F0F0F0']), 
                norm=BoundaryNorm([0, 0.05, 1.0], 2), cbar=False, square=True, linewidths=1, linecolor='white')
    
    plt.xticks(np.arange(n)+0.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(n)+0.5, labels, rotation=0)
    plt.title("Pairwise DeLong Test P-values (Top 6)")
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_OUTPUT_DIR, "DeLong_Heatmap_Top6.png"))
    plt.close()

# ===================================================================
# 6. Main Execution
# ===================================================================
def run_pipeline():
    # 1. Load Data
    if not os.path.exists(FILE_PATH): print("❌ Data Not Found"); return
    load_res = load_and_preprocess_data(FILE_PATH)
    if not load_res or load_res[0] is None: return
    df, pcr_cols, clin_cols = load_res
    print(f"ℹ️ Clinical Vars: {clin_cols}")
    
    y = df['Halitosis'].values
    print(f"🔍 Class Balance: {np.bincount(y)}")
    
    # 2. Visualize Cluster
    visualize_cluster(df, pcr_cols, CORRELATION_THRESHOLD, MAIN_OUTPUT_DIR)
    
    # 3. Define Models (Combined Clusters)
    clusters = get_combined_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
    models_to_run = clusters.copy()
    for col in pcr_cols: models_to_run[f"Single_{col}"] = [col]
    print(f" -> Total {len(models_to_run)} models defined.")
    
    summary_list = []
    
    # 4. Analysis
    for name, feats in tqdm(models_to_run.items(), desc="Analyzing Models"):
        # Logistic
        try:
            res = run_nested_cv('Logistic', feats, clin_cols, df, y)
            save_results(name, 'Logistic', res)
            pooled_auc = roc_auc_score(res['y_true'], res['y_prob'])
            summary_list.append({
                'ModelName': name,
                'Type': 'Logistic',
                'Mean_AUC': pooled_auc,
                'Std_AUC': res['Std_AUC'],
                'Fold_Mean_AUC': res['Mean_AUC'],
                'Pooled_AUC': pooled_auc,
            })
        except Exception as e: pass
        
        # Transformer
        try:
            res = run_nested_cv('Transformer', feats, clin_cols, df, y)
            save_results(name, 'Transformer', res)
            pooled_auc = roc_auc_score(res['y_true'], res['y_prob'])
            summary_list.append({
                'ModelName': name,
                'Type': 'Transformer',
                'Mean_AUC': pooled_auc,
                'Std_AUC': res['Std_AUC'],
                'Fold_Mean_AUC': res['Mean_AUC'],
                'Pooled_AUC': pooled_auc,
            })
        except Exception as e: pass
        
    # 5. Summarize & Visualize
    if not summary_list: print("❌ No results."); return
    
    sum_df = pd.DataFrame(summary_list)
    sum_df.to_csv(os.path.join(MAIN_OUTPUT_DIR, "Classification_Summary.csv"), index=False)
    
    print("\n🎨 Generating Plots...")
    plot_roc_top6(sum_df)
    plot_delong_heatmap(sum_df)
    
    print("\n✨ Halitosis Classification Analysis Completed.")

if __name__ == "__main__":
    run_pipeline()
#%%
