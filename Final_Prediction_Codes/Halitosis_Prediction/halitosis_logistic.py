#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, auc
from scipy import stats
import os
import itertools

# ===================================================================
# 1. Configuration
# ===================================================================
FILE_PATH = "PCR_NGS_Data.xlsx"
OUTPUT_DIR = "analysis_outputs_halitosis_logistic_nested_ci_clusters" # 폴더명도 헷갈리지 않게 변경 권장
CORRELATION_THRESHOLD = 0.8

CLINICAL_VARS_RAW = ['Sex', 'Age', 'Smoking', 'Oral hygiene']
TARGET_VAR = 'Halitosis'

# Hyperparameters
HYPERPARAMS = {
    'lr': [0.1, 0.01, 0.001],
    'epochs': [500, 1000]
}

BATCH_SIZE = 8
OUTER_FOLDS = 3
INNER_FOLDS = 3
RANDOM_STATE = 42
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Helper Functions (Stats & Data)
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

# 세균명 변환 함수 (PCR_ 제거 및 학명 약어 처리)
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
        if file_path.endswith('.csv'): df_raw = pd.read_csv(file_path, header=None)
        else: df_raw = pd.read_excel(file_path, header=None)
        
        header_idx = -1
        for i, row in df_raw.iterrows():
            s = row.astype(str).values
            if 'Sex' in s and 'Age' in s: header_idx = i; break
        if header_idx == -1: header_idx = 6
        
        df_raw.columns = df_raw.iloc[header_idx]
        df = df_raw.iloc[header_idx+1:].reset_index(drop=True)
        
        selected = {}
        for col in df.columns:
            c_str = str(col).strip()
            if "PCR" in c_str: 
                clean_name = c_str.replace("__", "_").replace(" ", "_")
                if not clean_name.startswith("PCR_"): clean_name = "PCR_" + clean_name
                selected[clean_name] = pd.to_numeric(df[col], errors='coerce')
            elif c_str in CLINICAL_VARS_RAW: selected[c_str] = df[col]
            elif TARGET_VAR in c_str: selected[TARGET_VAR] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = pd.DataFrame(selected).dropna(subset=[TARGET_VAR])
        if 'Age' in df_clean: df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
        
        df_final = pd.get_dummies(df_clean, columns=[c for c in ['Sex', 'Smoking', 'Oral hygiene'] if c in df_clean], drop_first=True, dtype=float)
        pcr = [c for c in df_final.columns if c.startswith("PCR_")]
        clin = [c for c in df_final.columns if any(x in c for x in ['Sex', 'Smoking', 'Oral', 'Age'])]
        return df_final, pcr, clin
    except Exception as e:
        print(f"Error: {e}"); return None, None, None

# [변경] 함수 이름 명확화 (Groups -> Clusters)
def get_split_correlation_clusters(df, pcr_features, threshold):
    df_hal = df[df[TARGET_VAR] == 1]; df_non = df[df[TARGET_VAR] == 0]
    def _find(sub):
        if len(sub)<5: return []
        G=nx.from_pandas_adjacency(sub[pcr_features].corr().abs().fillna(0))
        G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w<threshold])
        return [list(c) for c in nx.connected_components(G) if len(c)>=2]
    return _find(df_hal), _find(df_non)

def compute_metrics_with_ci_single_fold(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    n_boot = 1000
    rng = np.random.RandomState(RANDOM_STATE)
    
    metrics = {'AUROC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}
    
    pt_auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pt_acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pt_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    pt_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    for i in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        metrics['AUROC'].append(roc_auc_score(yt, ypr))
        metrics['Accuracy'].append(accuracy_score(yt, yp))
        
        t, f, n, p = confusion_matrix(yt, yp).ravel()
        metrics['Sensitivity'].append(p / (p + n) if (p + n) > 0 else 0)
        metrics['Specificity'].append(t / (t + f) if (t + f) > 0 else 0)
        
    res = {}
    for k, v in metrics.items():
        if not v: res[f'{k}_str'] = "NaN"; continue
        l = np.percentile(v, 2.5); u = np.percentile(v, 97.5)
        pt_val = pt_auroc if k=='AUROC' else (pt_acc if k=='Accuracy' else (pt_sens if k=='Sensitivity' else pt_spec))
        res[f'{k}_str'] = f"{pt_val:.3f} ({l:.3f}-{u:.3f})"
    
    res['AUROC_val'] = pt_auroc
    res['Accuracy_val'] = pt_acc
    res['Sensitivity_val'] = pt_sens
    res['Specificity_val'] = pt_spec
    
    return res

# ===================================================================
# 3. Model: Logistic Regression
# ===================================================================
class LogisticRegressionNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

def train_model(model, X, y, params):
    opt = optim.Adam(model.parameters(), lr=params['lr'])
    crit = nn.BCEWithLogitsLoss()
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE), y_t)
    
    bs = min(BATCH_SIZE, len(X))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(X)>bs))
    
    model.train()
    for _ in range(params['epochs']):
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    return model

def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()

def tune_hyperparams(X, y, input_dim):
    keys, values = zip(*HYPERPARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_auc = -1.0; best_params = combinations[0]
    
    inner_kf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for params in combinations:
        val_aucs = []
        for tr, val in inner_kf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            X_tr_s = sc.transform(X[tr]); X_val_s = sc.transform(X[val])
            
            model = LogisticRegressionNN(input_dim).to(DEVICE)
            model = train_model(model, X_tr_s, y[tr], params)
            probs = predict_model(model, X_val_s)
            
            try: val_aucs.append(roc_auc_score(y[val], probs))
            except: val_aucs.append(0.5)
            
        mean_auc = np.mean(val_aucs)
        if mean_auc > best_auc: best_auc = mean_auc; best_params = params
    return best_params

# ===================================================================
# 4. Nested CV Pipeline
# ===================================================================
def run_nested_cv(name, features, fixed_features, X_full, y_full):
    cols = fixed_features + features
    X, y = X_full[cols].values, y_full 
    in_dim = X.shape[1]
    
    outer_kf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_details = [] 
    y_true_oof, y_prob_oof = [], [] 
    
    raw_metrics_list = {'AUROC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}
    
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_kf.split(X, y)):
        X_tr_out, y_tr_out = X[tr_idx], y[tr_idx]
        X_te_out, y_te_out = X[te_idx], y[te_idx]
        
        best_params = tune_hyperparams(X_tr_out, y_tr_out, in_dim)
        
        sc = StandardScaler().fit(X_tr_out)
        X_tr_s = sc.transform(X_tr_out); X_te_s = sc.transform(X_te_out)
        
        model = LogisticRegressionNN(in_dim).to(DEVICE)
        model = train_model(model, X_tr_s, y_tr_out, best_params)
        probs = predict_model(model, X_te_s)
        
        stats_fold = compute_metrics_with_ci_single_fold(y_te_out, probs)
        
        fold_details.append({
            'Fold': fold_idx + 1,
            'AUROC': stats_fold['AUROC_str'],
            'Accuracy': stats_fold['Accuracy_str'],
            'Sensitivity': stats_fold['Sensitivity_str'],
            'Specificity': stats_fold['Specificity_str'],
            'Best_Params': str(best_params)
        })
        
        raw_metrics_list['AUROC'].append(stats_fold['AUROC_val'])
        raw_metrics_list['Accuracy'].append(stats_fold['Accuracy_val'])
        raw_metrics_list['Sensitivity'].append(stats_fold['Sensitivity_val'])
        raw_metrics_list['Specificity'].append(stats_fold['Specificity_val'])
        
        y_true_oof.extend(y_te_out); y_prob_oof.extend(probs)

    final_stats = {}
    for k, v_list in raw_metrics_list.items():
        final_stats[k] = f"{np.mean(v_list):.3f} ± {np.std(v_list):.3f}"
    
    # [추가됨] OOF 예측 결과를 CSV로 저장
    oof_df = pd.DataFrame({
        'y_true': y_true_oof,
        'y_prob': y_prob_oof
    })
    # 파일명에 모델 이름과 알고리즘 종류(Logistic/Transformer)를 포함
    save_filename = f"{name}_oof_predictions_logistic.csv"
    oof_df.to_csv(os.path.join(OUTPUT_DIR, save_filename), index=False)
    print(f"   💾 Saved OOF predictions to {save_filename}")

    return {
        'Model': name,
        'Fold_Details': fold_details,
        'Overall_Stats': final_stats,
        'Raw_Mean_AUROC': np.mean(raw_metrics_list['AUROC']),
        'y_true_oof': np.array(y_true_oof),
        'y_prob_oof': np.array(y_prob_oof),
        'Features': ", ".join([shorten(f) for f in features])
    }

# ===================================================================
# 5. Visualization (ROC with Bootstrap CI)
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
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Top 3 Halitosis Models ROC (Nested CV)')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "halitosis_top3_roc.png"), dpi=300)
    plt.show()

# ===================================================================
# 6. Main Execution
# ===================================================================
if __name__ == "__main__":
    df, pcr_cols, clin_cols = load_and_preprocess_data(FILE_PATH)
    
    if df is not None:
        print(f"✅ Data Ready: {len(df)} samples")
        y = df[TARGET_VAR].values
        
        # [변경] 출력 메시지 및 변수명 (Groups -> Clusters)
        print("🔍 Analyzing Correlation Clusters (Split by Class)...")
        clusters_hal, clusters_non = get_split_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
        print(f"   - Halitosis Clusters: {len(clusters_hal)}")
        print(f"   - Non-Halitosis Clusters: {len(clusters_non)}")
        
        models = {'Base_Clinical': []}
        
        # Single 모델 이름 설정 (변경 없음)
        for p in pcr_cols: 
            model_name = f"Single_{shorten(p)}" 
            models[model_name] = [p]
            
        # [변경] 모델 이름 설정 (Group -> Cluster)
        # 결과 CSV의 Model 컬럼과 그래프 범례에 'Cluster_Hal_1' 등으로 표시됨
        for i, g in enumerate(clusters_hal): models[f"Cluster_Hal_{i+1}"] = g
        for i, g in enumerate(clusters_non): models[f"Cluster_NonHal_{i+1}"] = g
        
        print(f"\nTraining {len(models)} models (Nested CV)...")
        
        results = []
        for name, feats in models.items():
            res = run_nested_cv(name, feats, clin_cols, df, y)
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
        
        # 3. DeLong Test
        print("\nRunning DeLong's Test on OOF Predictions...")
        n = len(results); p_vals = np.zeros((n, n)); names = [r['Model'] for r in results]
        for i in range(n):
            for j in range(i+1, n):
                p = delong_roc_test(results[i]['y_true_oof'], results[i]['y_prob_oof'], results[j]['y_prob_oof'])
                p_vals[i, j] = p; p_vals[j, i] = p
        pd.DataFrame(p_vals, index=names, columns=names).to_csv(os.path.join(OUTPUT_DIR, "delong_pvalues.csv"))
        
        # 4. Visualization
        plot_top3_roc_bootstrap(results)
        print("✅ All Analysis Completed.")
#%%