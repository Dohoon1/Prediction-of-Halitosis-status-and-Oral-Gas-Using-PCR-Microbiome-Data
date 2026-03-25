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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon
import os
import glob
import itertools
import warnings
import pickle
import shap
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import re

warnings.filterwarnings('ignore')

# ===================================================================
# 1. Configuration
# ===================================================================
# 새로운 방법론 결과 저장 폴더
MAIN_OUTPUT_DIR = "Analysis_Result_NewMethod_Combined_Cluster_0.4_clinicalInculded"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

FILE_PATH = "PCR_NGS_Data.xlsx"
TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
CORRELATION_THRESHOLD = 0.4
TARGET_VAR_FOR_CLUSTERING = 'Halitosis'
CLINICAL_VAR_NAMES  = ['Sex', 'Age', 'Smoking', 'Oral hygiene']
CONTINUOUS_VARS = ['Age']

# Hyperparameters
RIDGE_GRID = {'alpha': [0.01, 0.1, 1.0, 10.0]}
TRANSFORMER_GRID = {
    'k': [32, 64],
    'heads': [2, 4],
    'depth': [2, 3, 4], 
    'lr': [1e-4],
    'epochs': [100]
}

BATCH_SIZE = 8
OUTER_FOLDS = 3
INNER_FOLDS = 3
RANDOM_STATE = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Visualization Colors
DISTINCT_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b'] # 6 colors
COLOR_SIG = '#FFFF00'; COLOR_NOT_SIG = '#F0F0F0'

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Data Loading & New Clustering Logic
# ===================================================================
def format_bacterial_name(raw_name):
    clean = raw_name.replace("PCR_", "").replace("PCR", "").replace("__", " ").replace("_", " ").strip()
    parts = clean.split()
    if len(parts) >= 2: return f"{parts[0][0].upper()}. {parts[1]}"
    else: return clean

# 임상 변수 4개 포함 x
# def load_and_preprocess_data(file_path):
#     print(f"📂 Loading data from '{file_path}'...")
#     try:
#         if file_path.endswith('.csv'): df_raw = pd.read_csv(file_path, header=None)
#         else: df_raw = pd.read_excel(file_path, header=None) 
#         header_idx = -1
#         for i, row in df_raw.iterrows():
#             if 'Sex' in str(row.values) and 'Age' in str(row.values): header_idx = i; break
#         if header_idx == -1: header_idx = 6
#         df_raw.columns = df_raw.iloc[header_idx]
#         df = df_raw.iloc[header_idx+1:].reset_index(drop=True)
#         selected = {}
#         target_map = {'H2S': 'H2S_ppb', 'CH3SH': 'CH3SH_ppb', 'VSCs': 'VSCs_ppb'}
#         pcr_columns_clean = []
#         for col in df.columns:
#             c_str = str(col).strip()
#             if "PCR" in c_str:
#                 formatted_name = format_bacterial_name(c_str)
#                 selected[formatted_name] = pd.to_numeric(df[col], errors='coerce')
#                 pcr_columns_clean.append(formatted_name)
#             elif 'Halitosis' in c_str:
#                 selected['Halitosis'] = pd.to_numeric(df[col], errors='coerce')
#             else:
#                 for k, v in target_map.items():
#                     if k in c_str: selected[v] = pd.to_numeric(df[col], errors='coerce'); break
#         df_clean = pd.DataFrame(selected)
#         df_final = df_clean.dropna(subset=['Halitosis'])
#         return df_final, pcr_columns_clean
#     except Exception as e: print(f"❌ Error: {e}"); return None, None

# 임상 변수 4개 포함 o
def load_and_preprocess_data(file_path):
    print(f"📂 Loading data from '{file_path}'...")
    try:
        # 1. 파일 읽기
        if file_path.endswith('.csv'): df_raw = pd.read_csv(file_path, header=None)
        else: df_raw = pd.read_excel(file_path, header=None) 
        
        # 헤더 찾기
        header_idx = -1
        for i, row in df_raw.iterrows():
            row_str = str(row.values)
            if 'Sex' in row_str and 'Age' in row_str: 
                header_idx = i
                break
        
        if header_idx == -1: 
            print("⚠️ Header not found automatically. Using default index 6.")
            header_idx = 6
        
        df_raw.columns = df_raw.iloc[header_idx]
        df = df_raw.iloc[header_idx+1:].reset_index(drop=True)
        
        selected = {}
        target_map = {'H2S': 'H2S_ppb', 'CH3SH': 'CH3SH_ppb', 'VSCs': 'VSCs_ppb'}
        pcr_columns_clean = []
        
        # [Step A] 박테리아 및 타겟 변수 추출
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
        
        # [Step B] 임상 변수 추출 및 처리
        clinical_cols_found = []
        for clin_var in CLINICAL_VAR_NAMES:
            found_col = [c for c in df.columns if clin_var.lower() in str(c).lower()]
            
            if found_col:
                col_name = found_col[0]
                series = df[col_name]
                
                # 1. 연속형 변수 처리 (Age 등) -> 숫자 그대로 사용
                if clin_var in CONTINUOUS_VARS:
                    # 문자가 섞여있어도 강제로 숫자로 변환 (변환 안되면 NaN)
                    selected[clin_var] = pd.to_numeric(series, errors='coerce')
                    print(f"   ℹ️ Treated '{clin_var}' as Continuous Variable (Numeric).")
                    clinical_cols_found.append(clin_var)
                    continue 

                # 2. 범주형 변수 처리 (Sex, Smoking 등) -> 0, 1로 매핑
                if series.dtype == object:
                    mapping = {}
                    unique_vals = series.dropna().unique()
                    
                    print(f"   ℹ️ Mapping clinical var '{clin_var}': {unique_vals}")
                    for v in unique_vals:
                        v_str = str(v).lower().strip()
                        if v_str in ['male', 'M', 'man']: mapping[v] = 0
                        elif v_str in ['female', 'F', 'woman']: mapping[v] = 1
                        else: 
                            try: mapping[v] = float(v)
                            except: pass 
                    
                    series_mapped = series.map(mapping)
                    
                    # 매핑 안 된 값(Oral hygiene 등)은 Factorize
                    if series_mapped.isnull().any():
                        if clin_var == 'Oral hygiene' or series_mapped.isnull().sum() > 0:
                            codes, uniques = pd.factorize(series)
                            series_mapped = pd.Series(codes, index=series.index)
                            print(f"      -> Auto-factorized '{clin_var}': {list(enumerate(uniques))}")
                    
                    selected[clin_var] = pd.to_numeric(series_mapped, errors='coerce')
                else:
                    selected[clin_var] = pd.to_numeric(series, errors='coerce')
                
                clinical_cols_found.append(clin_var)
            else:
                print(f"⚠️ Clinical variable '{clin_var}' NOT found.")

        df_clean = pd.DataFrame(selected)
        
        # [Step C] 결측치 제거
        subset_cols = ['Halitosis'] + clinical_cols_found
        for gas in TARGET_GASES:
            if gas in df_clean.columns: subset_cols.append(gas)
            
        df_final = df_clean.dropna(subset=subset_cols)
        print(f"   -> Data Shape after preprocessing: {df_final.shape}")
        
        return df_final, pcr_columns_clean, clinical_cols_found

    except Exception as e: 
        print(f"❌ Error in data loading: {e}"); return None, None, None

# --- [Step 1] 클러스터링 비교 및 시각화 함수 ---
def visualize_cluster_changes(df, pcr_features, threshold, output_dir):
    """
    기존 방식(Split)과 새로운 방식(Combined)의 네트워크 그래프를 그려서 차이를 시각화합니다.
    """
    print("🔍 Visualizing Cluster Differences...")
    
    df_hal = df[df[TARGET_VAR_FOR_CLUSTERING] == 1]
    df_non = df[df[TARGET_VAR_FOR_CLUSTERING] == 0]
    
    # 그래프 생성 함수
    def create_graph(sub_df):
        if len(sub_df) < 5: return nx.Graph()
        corr = sub_df[pcr_features].corr().abs().fillna(0)
        G = nx.from_pandas_adjacency(corr)
        G.remove_edges_from([(u, v) for u, v, w in G.edges(data='weight') if w < threshold])
        return G

    G_hal = create_graph(df_hal)
    G_non = create_graph(df_non)
    G_combined = create_graph(df) # ★ New Method
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    pos_layout = nx.spring_layout(G_combined, k=0.3, seed=42) # 레이아웃 고정
    
    def draw_g(G, ax, title):
        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "No Connections", ha='center')
        else:
            # 연결된 노드만 표시
            connected_nodes = [n for n in G.nodes if G.degree(n) > 0]
            sub_G = G.subgraph(connected_nodes)
            if len(sub_G.nodes) > 0:
                pos = nx.spring_layout(sub_G, k=0.5, seed=42)
                nx.draw(sub_G, pos, ax=ax, with_labels=True, 
                        node_color='lightblue', edge_color='gray', 
                        node_size=500, font_size=8, font_weight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

    draw_g(G_hal, axes[0], "Old: Halitosis Group Only")
    draw_g(G_non, axes[1], "Old: Non-Halitosis Group Only")
    draw_g(G_combined, axes[2], "New: All Patients Combined")
    
    plt.suptitle(f"Cluster Network Comparison (Corr Threshold > {threshold})", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Cluster_Method_Comparison.png"), dpi=300)
    plt.close()
    print("   ✅ Cluster comparison plot saved.")

# --- [Step 2] 새로운 방식의 클러스터 추출 함수 ---
def get_combined_correlation_clusters(df, pcr_features, threshold):
    """
    전체 환자를 대상으로 상관관계를 분석하여 클러스터를 생성합니다.
    """
    G = nx.from_pandas_adjacency(df[pcr_features].corr().abs().fillna(0))
    G.remove_edges_from([(u,v) for u,v,w in G.edges(data='weight') if w < threshold])
    
    clusters_list = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
    
    # Naming: Cluster_Combined_1, Cluster_Combined_2 ...
    clusters = {f"Cluster_Combined_{i+1}": g for i, g in enumerate(clusters_list)}
    return clusters

# ===================================================================
# 3. Model Definition (Transformer & Helper) - 기존과 동일
# ===================================================================
class SelfAttention(nn.Module):
    def __init__(self, k, heads=4):
        super().__init__()
        assert k % heads == 0
        self.heads, self.d_k = heads, k // heads
        self.tokeys, self.toqueries, self.tovalues = nn.Linear(k,k,bias=False), nn.Linear(k,k,bias=False), nn.Linear(k,k,bias=False)
        self.unifyheads = nn.Linear(k, k)
    def forward(self, x):
        b, t, k = x.size(); h, d_k = self.heads, self.d_k
        Q = self.toqueries(x).view(b,t,h,d_k).transpose(1,2)
        K = self.tokeys(x).view(b,t,h,d_k).transpose(1,2)
        V = self.tovalues(x).view(b,t,h,d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1))/np.sqrt(d_k)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(b,t,h*d_k)
        return self.unifyheads(out), attn

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, k, heads, depth, dropout=0.1): 
        super().__init__()
        self.embedding_blocks = nn.ModuleList([nn.Sequential(nn.Linear(1, k), nn.SiLU(), nn.Linear(k, k)) for _ in range(seq_len)])
        self.enc = nn.ModuleList([self._make_block(k, heads) for _ in range(depth)])
        self.fc = nn.Linear(k, 1); self.dropout = nn.Dropout(dropout)
    def _make_block(self, k, heads):
        class Block(nn.Module):
            def __init__(self, k, heads):
                super().__init__(); self.attn = SelfAttention(k, heads); self.norm1=nn.LayerNorm(k); self.norm2=nn.LayerNorm(k)
                self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))
            def forward(self, x):
                a, _ = self.attn(x); x = self.norm1(a+x); f = self.ff(x); return self.norm2(f+x), None
        return Block(k, heads)
    def forward(self, x):
        emb = torch.stack([self.embedding_blocks[i](x[:,i].unsqueeze(-1)) for i in range(x.size(1))], dim=1)
        for l in self.enc: emb, _ = l(emb)
        return self.fc(self.dropout(emb.mean(dim=1))).squeeze(-1)

def train_epoch_trans(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(); out = model(xb.to(DEVICE))
        loss = criterion(out, yb.to(DEVICE)); loss.backward(); optimizer.step()

# ===================================================================
# 4. Evaluation & Training Pipeline
# ===================================================================
def run_nested_cv(model_type, feature_names, X_df, y_full):
    X = X_df[feature_names].values; 
    if X.ndim == 1: X = X.reshape(-1, 1)
    
    outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_trues_all, y_preds_all, fold_r2s = [], [], []
    feature_coeffs_sum = np.zeros(len(feature_names))
    final_best_config = None
    
    for tr_idx, te_idx in outer_kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]; y_tr, y_te = y_full[tr_idx], y_full[te_idx]
        sc_x = StandardScaler().fit(X_tr); X_tr_s = sc_x.transform(X_tr); X_te_s = sc_x.transform(X_te)
        sc_y = StandardScaler().fit(y_tr.reshape(-1, 1)); y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).flatten()
        
        inner_kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if model_type == 'Ridge': grid = RIDGE_GRID; combos = [{'alpha': a} for a in grid['alpha']]
        else: keys, values = zip(*TRANSFORMER_GRID.items()); combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
        best_loss = float('inf'); best_config = combos[0]
        for config in combos:
            if model_type == 'Transformer' and config['k']%config['heads']!=0: continue
            losses = []
            for i_tr, i_val in inner_kf.split(X_tr_s):
                if model_type == 'Ridge':
                    m = Ridge(alpha=config['alpha'], random_state=RANDOM_STATE).fit(X_tr_s[i_tr], y_tr_s[i_tr])
                    p = m.predict(X_tr_s[i_val]); losses.append(mean_squared_error(y_tr_s[i_val], p))
                else:
                    ds = TensorDataset(torch.tensor(X_tr_s[i_tr], dtype=torch.float32), torch.tensor(y_tr_s[i_tr], dtype=torch.float32))
                    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                    m = TransformerRegressor(X.shape[1], config['k'], config['heads'], config['depth']).to(DEVICE)
                    opt = optim.Adam(m.parameters(), lr=config['lr']); crit = nn.MSELoss()
                    for _ in range(config['epochs']): train_epoch_trans(m, ld, opt, crit)
                    m.eval()
                    with torch.no_grad(): 
                        out = m(torch.tensor(X_tr_s[i_val], dtype=torch.float32).to(DEVICE))
                        losses.append(crit(out, torch.tensor(y_tr_s[i_val], dtype=torch.float32).to(DEVICE)).item())
            if np.mean(losses) < best_loss: best_loss = np.mean(losses); best_config = config
        
        final_best_config = best_config
        if model_type == 'Ridge':
            fm = Ridge(alpha=best_config['alpha'], random_state=RANDOM_STATE).fit(X_tr_s, y_tr_s)
            pred_s = fm.predict(X_te_s); feature_coeffs_sum += fm.coef_
        else:
            ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32))
            ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            fm = TransformerRegressor(X.shape[1], best_config['k'], best_config['heads'], best_config['depth']).to(DEVICE)
            opt = optim.Adam(fm.parameters(), lr=best_config['lr']); crit = nn.MSELoss()
            for _ in range(best_config['epochs']): train_epoch_trans(fm, ld, opt, crit)
            fm.eval()
            with torch.no_grad(): pred_s = fm(torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            
        pred_final = sc_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
        y_trues_all.extend(y_te); y_preds_all.extend(pred_final); fold_r2s.append(r2_score(y_te, pred_final))

    avg_coeffs = feature_coeffs_sum / OUTER_FOLDS if model_type == 'Ridge' else None
    return {'y_true': np.array(y_trues_all), 'y_pred': np.array(y_preds_all), 
            'Mean_R2': np.mean(fold_r2s), 'Std_R2': np.std(fold_r2s), 
            'Best_Config': final_best_config, 'Avg_Coeffs': avg_coeffs}

def save_artifacts(model_type, model_name, target, res, X_df, features, y, out_dir):
    safe_name = model_name.replace(".", "").replace(" ", "_")
    pd.DataFrame({'y_true': res['y_true'], 'y_pred': res['y_pred']}).to_csv(os.path.join(out_dir, f"Pred_{safe_name}_{model_type}.csv"), index=False)
    if model_type == 'Ridge':
        pd.DataFrame({'Feature': features, 'Coefficient': res['Avg_Coeffs'], 'Abs': np.abs(res['Avg_Coeffs'])}).sort_values(by='Abs', ascending=False).to_csv(os.path.join(out_dir, f"Coef_{safe_name}_{model_type}.csv"), index=False)
    else:
        art_path = os.path.join(out_dir, "artifacts"); os.makedirs(art_path, exist_ok=True)
        # Re-train for saving
        sc_x = StandardScaler().fit(X_df[features]); X_s = sc_x.transform(X_df[features])
        sc_y = StandardScaler().fit(y.reshape(-1, 1)); y_s = sc_y.transform(y.reshape(-1, 1)).flatten()
        cfg = res['Best_Config']
        m = TransformerRegressor(len(features), cfg['k'], cfg['heads'], cfg['depth']).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=cfg['lr']); crit = nn.MSELoss()
        ds = TensorDataset(torch.tensor(X_s, dtype=torch.float32), torch.tensor(y_s, dtype=torch.float32))
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        m.train(); 
        for _ in range(cfg['epochs']): train_epoch_trans(m, ld, opt, crit)
        torch.save(m.state_dict(), os.path.join(art_path, f"model_{safe_name}.pth"))
        with open(os.path.join(art_path, f"meta_{safe_name}.pkl"), 'wb') as f: pickle.dump({'config':cfg, 'features':features, 'scaler_x':sc_x, 'scaler_y':sc_y}, f)
        X_df[features].sample(min(100, len(X_df)), random_state=42).to_csv(os.path.join(art_path, f"background_{safe_name}.csv"), index=False)

# ===================================================================
# 5. Result Visualization (Top 3 Cluster + Top 3 Single)
# ===================================================================
def get_top6_stratified(target_gas, summary_df):
    """
    Cluster 모델 중 Top 3 + Single 모델 중 Top 3 = 총 6개 모델 선정
    """
    df = summary_df[summary_df['Target'] == target_gas].copy()
    
    # 구분
    cluster_mask = df['ModelName'].str.contains("Cluster")
    single_mask = df['ModelName'].str.contains("Single")
    
    # Top 3 Cluster
    top3_cluster = df[cluster_mask].sort_values(by='R2_Mean', ascending=False).head(3)
    # Top 3 Single
    top3_single = df[single_mask].sort_values(by='R2_Mean', ascending=False).head(3)
    
    # Merge
    top6_df = pd.concat([top3_cluster, top3_single])
    
    print(f"   📊 Selected Top 6: {len(top3_cluster)} Clusters + {len(top3_single)} Singles")
    
    loaded_models = []
    for _, row in top6_df.iterrows():
        model_name = row['ModelName']
        model_type = row['Type']
        safe_name = model_name.replace(".", "").replace(" ", "_")
        
        pred_file = os.path.join(MAIN_OUTPUT_DIR, target_gas, f"Pred_{safe_name}_{model_type}.csv")
        if not os.path.exists(pred_file): continue
            
        df_pred = pd.read_csv(pred_file)
        y_true = df_pred['y_true'].values
        y_pred = df_pred['y_pred'].values
        
        loaded_models.append({
            'name': f"{model_name}\n({model_type})",
            'mean_r2': row['R2_Mean'],
            'std_r2': row['R2_Std'],
            'abs_errors': np.abs(y_true - y_pred),
            'sq_errors': (y_true - y_pred) ** 2
        })
    return loaded_models

def plot_stratified_violin(models, target_gas):
    plot_data = []
    for m in models:
        # 이름 줄바꿈
        label = f"{m['name']}\n($R^2$={m['mean_r2']:.3f}±{m['std_r2']:.3f})"
        for err in m['abs_errors']: plot_data.append({'Model': label, 'Absolute Error': err})
    
    df_plot = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    sns.violinplot(x='Model', y='Absolute Error', data=df_plot, palette=DISTINCT_COLORS[:len(models)], inner='quartile', ax=ax)
    sns.pointplot(x='Model', y='Absolute Error', data=df_plot, estimator=np.mean, errorbar='sd', color='black', capsize=0.1, markers='o', scale=0.7, ax=ax)
    
    plt.title(f"Top 6 Models (3 Cluster + 3 Single) Error Distribution - {target_gas}", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold', rotation_mode='anchor')
    dx, dy = 10/72., 0/72.; offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels(): label.set_transform(label.get_transform() + offset)
    plt.ylabel(f"Absolute Error ({target_gas})", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_OUTPUT_DIR, target_gas, f"Violin_Top6_{target_gas}.png"), dpi=300)
    plt.close()

def plot_stratified_heatmap(models, target_gas):
    n = len(models)
    labels = [f"{m['name']}\n($R^2$={m['mean_r2']:.2f})" for m in models]
    p_values = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: p_values[i, j] = 1.0; continue
            try: _, p = stats.wilcoxon(models[i]['sq_errors'], models[j]['sq_errors'])
            except: p = 1.0
            p_values[i, j] = p

    fig, ax = plt.subplots(figsize=(11, 10), dpi=300)
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i==j: annot[i,j]="-"
            elif p_values[i,j]<0.001: annot[i,j]="<.001"
            else: annot[i,j]=f"{p_values[i,j]:.3f}"
            
    cmap = ListedColormap([COLOR_SIG, COLOR_NOT_SIG]); norm = BoundaryNorm([0, 0.05, 1.0], cmap.N)
    sns.heatmap(p_values, annot=annot, fmt="", cmap=cmap, norm=norm, cbar=False, square=True, linewidths=1, linecolor='white', ax=ax)
    
    legend_elements = [Patch(facecolor=COLOR_SIG, edgecolor='lightgray', label='P < 0.05'), Patch(facecolor=COLOR_NOT_SIG, edgecolor='lightgray', label='Not Sig')]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, fontweight='bold', rotation_mode='anchor')
    ax.set_yticklabels(labels, rotation=0, fontsize=9, fontweight='bold')
    dx, dy = 10/72., 0/72.; offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels(): label.set_transform(label.get_transform() + offset)
    
    for t in ax.texts:
        if t.get_text()!="-" and ("<" in t.get_text() or float(t.get_text())<0.05): t.set_weight('bold'); t.set_color('black')
        else: t.set_color('gray')
        
    plt.title(f"Pairwise Wilcoxon Test (Top 6) - {target_gas}", fontsize=14, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_OUTPUT_DIR, target_gas, f"Heatmap_Top6_{target_gas}.png"), dpi=300)
    plt.close()

# ===================================================================
# 6. Main Execution Loop
# ===================================================================
def run_full_pipeline():
    # 1. Load Data
    if not os.path.exists(FILE_PATH): print("❌ Data not found."); return
    df, pcr_cols, clinical_cols = load_and_preprocess_data(FILE_PATH) # [수정]
    if df is None: return

    print(f"ℹ️ Clinical Variables included: {clinical_cols}")

    # 2. Visualize Cluster Changes (Comparison)
    visualize_cluster_changes(df, pcr_cols, CORRELATION_THRESHOLD, MAIN_OUTPUT_DIR)
    
    # 3. Define Models (New Combined Clusters + Singles)
    print("🔍 Defining Models with New Clustering Method...")
    # ★ New: Use Combined Clusters
    clusters = get_combined_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
    models_to_run = clusters.copy()
    for col in pcr_cols: models_to_run[f"Single_{col}"] = [col]
    print(f" -> Total {len(models_to_run)} models defined.")

    summary_list = []

    # 4. Analysis Loop
    for target in TARGET_GASES:
        print(f"\n{'='*60}\n🔥 Analyzing Target: {target}\n{'='*60}")
        tgt_dir = os.path.join(MAIN_OUTPUT_DIR, target); os.makedirs(tgt_dir, exist_ok=True)
        df_tgt = df.dropna(subset=[target]); y = df_tgt[target].values
        
        for name, feats in tqdm(models_to_run.items(), desc=f"Scanning {target}"):
            
            # [수정] ★핵심★: 박테리아 Feature(feats) + 임상 Feature(clinical_cols) 합치기
            # 이제 모델은 박테리아와 나이, 성별 등을 모두 보고 예측합니다.
            combined_features = feats + clinical_cols
            
            # Ridge
            try:
                res = run_nested_cv('Ridge', combined_features, df_tgt, y)
                save_artifacts('Ridge', name, target, res, df_tgt, combined_features, y, tgt_dir)
                summary_list.append({'Target':target, 'ModelName':name, 'Type':'Ridge', 'R2_Mean':res['Mean_R2'], 'R2_Std':res['Std_R2']})
            except: pass
            
            # Transformer
            try:
                res = run_nested_cv('Transformer', combined_features, df_tgt, y)
                save_artifacts('Transformer', name, target, res, df_tgt, combined_features, y, tgt_dir)
                summary_list.append({'Target':target, 'ModelName':name, 'Type':'Transformer', 'R2_Mean':res['Mean_R2'], 'R2_Std':res['Std_R2']})
            except: pass

    # 5. Save Summary & Visualize
    sum_df = pd.DataFrame(summary_list)
    sum_df.to_csv(os.path.join(MAIN_OUTPUT_DIR, "Summary_NewMethod.csv"), index=False)
    
    print("\n🎨 Visualizing Top 6 (3 Cluster + 3 Single)...")
    for target in TARGET_GASES:
        models = get_top6_stratified(target, sum_df)
        if len(models) < 2: continue
        plot_stratified_violin(models, target)
        plot_stratified_heatmap(models, target)
    
    print("\n✨ All Analysis & Visualization Completed.")

if __name__ == "__main__":
    run_full_pipeline()
#%%