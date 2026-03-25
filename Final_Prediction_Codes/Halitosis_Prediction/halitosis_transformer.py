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
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc, 
                             f1_score, precision_score, recall_score, confusion_matrix)
from scipy import stats
from imblearn.over_sampling import SMOTE
import os
import itertools
import math
import time

# ===================================================================
# 1. Configuration & Hyperparameter Grid
# ===================================================================
FILE_PATH = "PCR_NGS_Data.xlsx" 
OUTPUT_DIR = "analysis_outputs_halitosis_transformer_clusters_1e-4"
CORRELATION_THRESHOLD = 0.8
CLINICAL_VARS = ['Sex', 'Age', 'Smoking', 'Oral hygiene']
TARGET_VAR = 'Halitosis'

# Transformer Grid (조합이 많으면 오래 걸리므로 적절히 조절 필요)
TRANSFORMER_GRID = {
    'k': [16, 32, 64],          # Embedding Dimension
    'heads': [2, 4],        # Attention Heads
    'depth': [2, 3, 4],        # Encoder Layers
    'lr': [1e-4],     # Learning Rate
    'epochs': [100]         # Epochs
}

BATCH_SIZE = 8
OUTER_FOLDS = 3
INNER_FOLDS = 3
RANDOM_STATE = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Helper Functions (Stats, Data, DeLong)
# ===================================================================
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

def delong_roc_test(y_true, prob_a, prob_b):
    y_true = np.array(y_true).reshape(-1); prob_a = np.array(prob_a).reshape(-1); prob_b = np.array(prob_b).reshape(-1)
    order = np.argsort(prob_a); y_true = y_true[order]; prob_a = prob_a[order]; prob_b = prob_b[order]
    if len(np.unique(y_true)) < 2: return np.nan
    preds = np.vstack((prob_a, prob_b)); preds_sorted = np.hstack((preds[:, y_true == 1], preds[:, y_true == 0]))
    aucs, delongcov = fast_delong(preds_sorted, np.sum(y_true == 1))
    l = np.array([1, -1]); diff = np.diff(aucs)
    sigma = np.sqrt(np.dot(np.dot(l, delongcov), l.T))
    if sigma == 0: return 1.0
    z = np.abs(diff) / sigma
    return 2 * (1 - stats.norm.cdf(z))[0]

def shorten(name): 
    if name.startswith("PCR_"):
        clean = name.replace("PCR_", "").replace("_", " ").strip()
        parts = clean.split()
        if len(parts) >= 2:
            genus = parts[0][0].upper()
            species = parts[1]
            return f"{genus}. {species}"
        return clean
    return name.replace("_", " ")

def load_and_preprocess_data(file_path):
    print(f"📂 Loading data from '{file_path}'...")
    try:
        if file_path.endswith('.csv'): df = pd.read_csv(file_path, header=None)
        else: df = pd.read_excel(file_path, header=None)
        
        header_idx = -1
        for i, row in df.iterrows():
            if 'Sex' in str(row.values) and 'Age' in str(row.values): header_idx = i; break
        if header_idx == -1: header_idx = 6
        
        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        
        selected = {}
        for col in df.columns:
            c_str = str(col).strip()
            if "PCR" in c_str:
                clean = "PCR_" + c_str.replace("__", "_").replace(" ", "_").replace("PCR_", "")
                selected[clean] = pd.to_numeric(df[col], errors='coerce')
            elif c_str in CLINICAL_VARS: selected[c_str] = df[col]
            elif TARGET_VAR in c_str: selected[TARGET_VAR] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = pd.DataFrame(selected).dropna(subset=[TARGET_VAR])
        if 'Age' in df_clean: df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
        df_final = pd.get_dummies(df_clean, columns=['Sex', 'Smoking', 'Oral hygiene'], drop_first=True, dtype=float)
        
        pcr = [c for c in df_final.columns if c.startswith("PCR_")]
        clin = [c for c in df_final.columns if any(x in c for x in ['Sex', 'Smoking', 'Oral', 'Age'])]
        return df_final, pcr, clin
    except Exception as e:
        print(f"Error: {e}"); return None, None, None

def get_split_correlation_clusters(df, pcr_features, threshold):
    df_hal = df[df[TARGET_VAR]==1]; df_non = df[df[TARGET_VAR]==0]
    def _find(sub):
        if len(sub)<5: return []
        G=nx.from_pandas_adjacency(sub[pcr_features].corr().abs().fillna(0))
        G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w<threshold])
        return [list(c) for c in nx.connected_components(G) if len(c)>=2]
    return _find(df_hal), _find(df_non)

def compute_metrics_ci(y_true, y_prob):
    n_boot = 1000
    rng = np.random.RandomState(RANDOM_STATE)
    y_pred = (y_prob > 0.5).astype(int)
    
    metric_funcs = {
        'AUROC': lambda yt, yp, ypr: roc_auc_score(yt, ypr) if len(np.unique(yt))>1 else 0.5,
        'Accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp),
        'Sensitivity': lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        'Specificity': lambda yt, yp, ypr: confusion_matrix(yt, yp, labels=[0,1]).ravel()[0] / (confusion_matrix(yt, yp, labels=[0,1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0,1]).ravel()[1]) if (confusion_matrix(yt, yp, labels=[0,1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0,1]).ravel()[1]) > 0 else 0
    }
    
    pt_stats = {k: func(y_true, y_pred, y_prob) for k, func in metric_funcs.items()}
    boot_stats = {k: [] for k in metric_funcs.keys()}
    
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        for k, func in metric_funcs.items():
            boot_stats[k].append(func(yt, yp, ypr))
            
    result = {}
    for k, v in pt_stats.items():
        if len(boot_stats[k]) == 0: result[f'{k}_str'] = "NaN"; continue
        l = np.percentile(boot_stats[k], 2.5)
        u = np.percentile(boot_stats[k], 97.5)
        result[f'{k}_str'] = f"{v:.3f} ({l:.3f}-{u:.3f})"
        result[f'{k}_val'] = v
        
    return result

# ===================================================================
# 3. User Provided Transformer Architecture
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
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V).transpose(1,2).contiguous().view(b,t,h*d_k)
        return self.unifyheads(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1, self.norm2 = nn.LayerNorm(k), nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))
    def forward(self,x):
        return self.norm2(self.ff(self.norm1(x + self.attention(x))) + x)

class Transformer_Encoder(nn.Module):
    def __init__(self, k, heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(k,heads) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class Embedding_Block(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.input_encoder = nn.Sequential(nn.Linear(1, k), nn.SiLU(), nn.Linear(k, k), nn.SiLU(), nn.Linear(k, k), nn.LayerNorm(k))
    def forward(self, x): return self.input_encoder(x)

class Classifier(nn.Module):
    def __init__(self, seq_len, k, heads, depth, dropout=0.1): 
        super().__init__()
        self.embedding_blocks = nn.ModuleList([Embedding_Block(k) for _ in range(seq_len)])
        self.transformer_encoder = Transformer_Encoder(k, heads, depth)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(k, 1)

    def forward(self, x):
        embedded_list = [self.embedding_blocks[i](x[:, i].unsqueeze(-1)) for i in range(x.size(1))]
        x_emb = torch.stack(embedded_list, dim=1)
        x_trans = self.transformer_encoder(x_emb)
        x_pooled = x_trans.mean(dim=1)
        return self.fc(self.dropout(x_pooled)).squeeze(-1)

# ===================================================================
# 4. Training Engine
# ===================================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb.to(DEVICE))
        loss = criterion(out, yb.to(DEVICE))
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            preds.extend(torch.sigmoid(out).cpu().numpy())
            targets.extend(yb.cpu().numpy())
    return np.array(targets), np.array(preds)

# ===================================================================
# 5. Nested CV Pipeline
# ===================================================================
def run_inner_grid_search(X, y):
    keys, values = zip(*TRANSFORMER_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_auc = -1.0; best_config = combinations[0]
    inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for config in combinations:
        if config['k'] % config['heads'] != 0: continue
        
        fold_aucs = []
        for tr_idx, val_idx in inner_skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            sc = StandardScaler().fit(X_tr)
            X_tr_s, X_val_s = sc.transform(X_tr), sc.transform(X_val)
            
            try:
                sm = SMOTE(random_state=RANDOM_STATE)
                X_tr_res, y_tr_res = sm.fit_resample(X_tr_s, y_tr)
            except:
                X_tr_res, y_tr_res = X_tr_s, y_tr
                
            tr_ds = TensorDataset(torch.tensor(X_tr_res, dtype=torch.float32), torch.tensor(y_tr_res, dtype=torch.float32))
            val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
            
            tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=len(X_tr_res)>BATCH_SIZE)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
            
            model = Classifier(seq_len=X.shape[1], k=config['k'], heads=config['heads'], depth=config['depth']).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            criterion = nn.BCEWithLogitsLoss()
            
            for _ in range(config['epochs']): train_epoch(model, tr_loader, optimizer, criterion)
            
            y_true_val, y_prob_val = evaluate(model, val_loader)
            try: fold_aucs.append(roc_auc_score(y_true_val, y_prob_val))
            except: fold_aucs.append(0.5)
            
        if np.mean(fold_aucs) > best_auc:
            best_auc = np.mean(fold_aucs); best_config = config
    return best_config

def run_nested_cv(model_name, features, clin_cols, X_full, y_full):
    cols = clin_cols + features if model_name != 'Base_Clinical' else clin_cols
    cols = list(dict.fromkeys(cols))
    
    X, y = X_full[cols].values, y_full
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_details = []
    y_true_oof, y_prob_oof = [], []
    raw_metrics_list = {'AUROC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}
    
    print(f"   Processing {model_name} (Features: {len(cols)})...")
    
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        best_config = run_inner_grid_search(X_tr, y_tr)
        
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        
        try:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_s, y_tr)
        except:
            X_tr_res, y_tr_res = X_tr_s, y_tr
            
        tr_ds = TensorDataset(torch.tensor(X_tr_res, dtype=torch.float32), torch.tensor(y_tr_res, dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=len(X_tr_res)>BATCH_SIZE)
        
        model = Classifier(seq_len=X.shape[1], k=best_config['k'], heads=best_config['heads'], depth=best_config['depth']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(best_config['epochs']): train_epoch(model, tr_loader, optimizer, criterion)
        
        te_ds = TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32))
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)
        y_true_fold, y_prob_fold = evaluate(model, te_loader)
        
        metrics_fold = compute_metrics_ci(y_true_fold, y_prob_fold)
        
        fold_details.append({
            'Fold': fold_idx + 1,
            'Best_Config': str(best_config),
            'AUROC': metrics_fold['AUROC_str'],
            'Accuracy': metrics_fold['Accuracy_str'],
            'Sensitivity': metrics_fold['Sensitivity_str'],
            'Specificity': metrics_fold['Specificity_str']
        })
        
        raw_metrics_list['AUROC'].append(metrics_fold['AUROC_val'])
        raw_metrics_list['Accuracy'].append(metrics_fold['Accuracy_val'])
        raw_metrics_list['Sensitivity'].append(metrics_fold['Sensitivity_val'])
        raw_metrics_list['Specificity'].append(metrics_fold['Specificity_val'])
        
        y_true_oof.extend(y_true_fold); y_prob_oof.extend(y_prob_fold)
        
    final_stats = {}
    for k, v_list in raw_metrics_list.items():
        final_stats[k] = f"{np.mean(v_list):.3f} ± {np.std(v_list):.3f}"

    # [추가됨] OOF 예측 결과를 CSV로 저장
    oof_df = pd.DataFrame({
        'y_true': y_true_oof,
        'y_prob': y_prob_oof
    })
    # 파일명에 모델 이름과 알고리즘 종류(Logistic/Transformer)를 포함
    save_filename = f"{name}_oof_predictions_Transformer.csv"
    oof_df.to_csv(os.path.join(OUTPUT_DIR, save_filename), index=False)
    print(f"   💾 Saved OOF predictions to {save_filename}")

    return {
        'Model': model_name,
        'Fold_Details': fold_details,
        'Overall_Stats': final_stats,
        'Raw_Mean_AUROC': np.mean(raw_metrics_list['AUROC']),
        'y_true_oof': np.array(y_true_oof),
        'y_prob_oof': np.array(y_prob_oof)
    }

# ===================================================================
# 6. Visualization (Same Style as Logistic)
# ===================================================================
def plot_top3_roc_bootstrap(results):
    print("\n📊 Plotting Top 3 ROC Curves (Bootstrap CI)...")
    sorted_results = sorted(results, key=lambda x: x['Raw_Mean_AUROC'], reverse=True)
    top3 = sorted_results[:3]
    
    plt.figure(figsize=(10, 8))
    colors = ['#d62728', '#2ca02c', '#1f77b4'] 
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, res in enumerate(top3):
        y_true = res['y_true_oof']; y_prob = res['y_prob_oof']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        mean_auc = auc(fpr, tpr)
        
        tprs_boot = []
        rng = np.random.RandomState(RANDOM_STATE)
        for _ in range(1000):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2: continue
            f, t, _ = roc_curve(y_true[idx], y_prob[idx])
            tprs_boot.append(np.interp(mean_fpr, f, t))
            
        mean_tpr = np.mean(tprs_boot, axis=0); mean_tpr[-1] = 1.0
        lower = np.percentile(tprs_boot, 2.5, axis=0)
        upper = np.percentile(tprs_boot, 97.5, axis=0)
        
        plt.plot(mean_fpr, mean_tpr, color=colors[i], lw=2.5, alpha=.9, 
                 label=f"{res['Model']} (AUC={mean_auc:.3f})")
        plt.fill_between(mean_fpr, lower, upper, color=colors[i], alpha=.15)
        
    plt.plot([0,1],[0,1], 'k--', alpha=.6)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Top 3 Halitosis Transformer Models ROC')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "halitosis_transformer_top3_roc.png"), dpi=300)
    plt.show()

# ===================================================================
# 7. Main Execution
# ===================================================================
if __name__ == "__main__":
    df, pcr_cols, clin_cols = load_and_preprocess_data(FILE_PATH)
    
    if df is not None:
        print(f"✅ Data Ready: {len(df)} samples")
        
        # [변경] Group -> Cluster
        print("🔍 Analyzing Correlation Clusters (Split by Class)...")
        clusters_hal, clusters_non = get_split_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
        print(f"   - Halitosis Clusters: {len(clusters_hal)}")
        print(f"   - Non-Halitosis Clusters: {len(clusters_non)}")
        
        models = {'Base_Clinical': []}
        
        for i, g in enumerate(clusters_hal): models[f"Cluster_Hal_{i+1}"] = g
        for i, g in enumerate(clusters_non): models[f"Cluster_NonHal_{i+1}"] = g
        
        # 싱글 마커 모델 추가 (예시: 상관계수 상위 5개)
        corrs = df[pcr_cols + [TARGET_VAR]].corr()[TARGET_VAR].abs().sort_values(ascending=False)
        top5_single = corrs.index[1:6].tolist()
        for p in top5_single: models[f"Single_{shorten(p)}"] = [p]

        print(f"🚀 Starting Nested CV Grid Search for {len(models)} models...")
        
        results = []
        for name, feats in models.items():
            res = run_nested_cv(name, feats, clin_cols, df, df[TARGET_VAR].values)
            results.append(res)
            
        # 1. Save Detailed Report
        detailed_rows = []
        for res in results:
            for fold in res['Fold_Details']:
                detailed_rows.append({'Model': res['Model'], **fold})
            avg_stats = res['Overall_Stats']
            detailed_rows.append({
                'Model': res['Model'], 'Fold': 'Average ± Std', 
                'AUROC': avg_stats['AUROC'], 'Accuracy': avg_stats['Accuracy'],
                'Sensitivity': avg_stats['Sensitivity'], 'Specificity': avg_stats['Specificity']
            })
        pd.DataFrame(detailed_rows).to_csv(os.path.join(OUTPUT_DIR, "detailed_report.csv"), index=False)
        
        # 2. Summary
        summary = pd.DataFrame([{
            'Model': r['Model'], 'Mean AUROC': r['Overall_Stats']['AUROC'], 'Raw AUC': r['Raw_Mean_AUROC']
        } for r in results]).sort_values(by='Raw AUC', ascending=False)
        print("\n[Top 5 Models]")
        print(summary.head(5).to_string(index=False))
        summary.to_csv(os.path.join(OUTPUT_DIR, "summary_report.csv"), index=False)
        
        # 3. DeLong Test (Added back)
        print("\nRunning DeLong's Test on OOF Predictions...")
        n = len(results); p_vals = np.zeros((n, n)); names = [r['Model'] for r in results]
        for i in range(n):
            for j in range(i+1, n):
                p = delong_roc_test(results[i]['y_true_oof'], results[i]['y_prob_oof'], results[j]['y_prob_oof'])
                p_vals[i, j] = p; p_vals[j, i] = p
        pd.DataFrame(p_vals, index=names, columns=names).to_csv(os.path.join(OUTPUT_DIR, "delong_pvalues.csv"))
        
        # 4. Plot
        plot_top3_roc_bootstrap(results)
        print("✅ Process Completed.")