#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from sklearn.metrics import r2_score
import matplotlib.cm as cm 
import json
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 설정
# ===================================================================
INPUT_DIR = "Analysis_Result_NewMethod_PatientCombined_Cluster_0.4"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "Analysis_Result_NewMethod_PatientCombined_Cluster_0.4" 

SUMMARY_FILE = os.path.join(INPUT_DIR, "Summary_NewMethod_Clinical_excluded.csv")
VIS_OUTPUT_DIR = "Visualization_NewMethod_0.4_Nonclinical" 
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42

COLOR_SIG = '#FFD700'  # Gold
COLOR_NOT_SIG = '#F0F0F0' # Light Gray
P_GINGIVALIS_SHADES = ['#CC3F3F', '#B53636', '#D85B5B']  # further redder, still toned-down
NON_RED_DISTINCT = ['#1f77b4', '#2ca58d', '#6a4c93', '#17becf', '#bcbd22', '#4c78a8', '#2e8b57', '#7f7f7f']

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ===================================================================
# 2. Transformer Classes (생략 가능하지만 호환성을 위해 유지)
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k); self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))
    def forward(self, x):
        attended, weights = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x), weights

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

# ===================================================================
# 3. 데이터 로드 및 처리
# ===================================================================
def find_prediction_file(model_name, model_type, target_gas):
    safe_name = model_name.replace(".", "").replace(" ", "_")
    base_path = os.path.join(INPUT_DIR, target_gas)
    
    path = os.path.join(base_path, f"Pred_{safe_name}_{model_type}.csv")
    if os.path.exists(path): return path
    
    for t in ['', '_Ridge', '_Transformer']:
        path = os.path.join(base_path, f"Pred_{safe_name}{t}.csv")
        if os.path.exists(path): return path
    return None

def get_top5_with_summary_stats(target_gas):
    """부트스트랩 계산 대신 Summary의 R²를 그대로 사용하는 함수"""
    if not os.path.exists(SUMMARY_FILE):
        print(f"❌ Summary file missing: {SUMMARY_FILE}"); return [], None

    df = pd.read_csv(SUMMARY_FILE)
    if 'R2' in df.columns and 'R2_Mean' not in df.columns:
        df.rename(columns={'R2': 'R2_Mean'}, inplace=True)
        
    df_gas = df[df['Target'] == target_gas].copy()
    if df_gas.empty: return [], None
    
    # 1. Cluster Top 3 + Single Top 3 (또는 필요에 따라 조절)
    cluster_mask = df_gas['ModelName'].str.contains("Cluster")
    single_mask = df_gas['ModelName'].str.contains("Single")
    
    hal_models = df_gas[df_gas['ModelName'].str.contains("Cluster_Hal")]
    nonhal_models = df_gas[df_gas['ModelName'].str.contains("Cluster_NonHal")]
    combined_models = df_gas[df_gas['ModelName'].str.contains("Cluster_Combined")]
    
    # Winner 키워드 탐색 (Hal vs NonHal vs Combined)
    winner_keyword = "Cluster"
    if not combined_models.empty:
        winner_keyword = "Cluster_Combined"
    elif not hal_models.empty and not nonhal_models.empty:
        max_hal = hal_models['R2_Mean'].max()
        max_nonhal = nonhal_models['R2_Mean'].max()
        winner_keyword = "Cluster_Hal" if max_hal > max_nonhal else "Cluster_NonHal"
        
    candidates = df_gas[
        (df_gas['ModelName'].str.contains(winner_keyword)) | 
        (df_gas['ModelName'].str.contains("Single"))
    ].copy()
    
    # R2_Mean 기준으로 정렬하여 Top 5 추출
    top5_df = candidates.sort_values(by='R2_Mean', ascending=False).head(5)
    
    processed_models = []
    
    for _, row in top5_df.iterrows():
        m_name = row['ModelName']
        m_type = row['Type'] if 'Type' in row else 'Unknown'
        
        pred_path = find_prediction_file(m_name, m_type, target_gas)
        if not pred_path: continue
        
        df_pred = pd.read_csv(pred_path)
        y_true = df_pred['y_true'].values
        y_pred = df_pred['y_pred'].values
        
        processed_models.append({
            'name': m_name,
            'type': m_type,
            'mean_r2': row['R2_Mean'], # Summary 값 그대로 사용
            'std_r2': row['R2_Std'] if 'R2_Std' in row else 0.0,
            'y_true': y_true,
            'y_pred': y_pred,
            'abs_errors': np.abs(y_true - y_pred),
            'sq_errors': (y_true - y_pred) ** 2,
            'safe_name': m_name.replace(".", "").replace(" ", "_")
        })
        
    return processed_models, winner_keyword

# ===================================================================
# 4. 시각화 함수 (Violin, Heatmap)
# ===================================================================
def format_model_name_only(name):
    """Cluster_Combined_1 -> Cluster 1 로 변경"""
    if "Cluster" in name: 
        num = name.split('_')[-1] # 마지막 숫자 추출
        return f"Cluster {num}"
    elif "Single" in name: 
        return name.replace("Single_", "Single ")
    else: 
        return name

def format_label_full(m):
    """라벨 출력용 (R² 값 포함)"""
    short_name = format_model_name_only(m['name'])
    return f"{short_name}\n({m['type']})\n($R^2$={m['mean_r2']:.3f}±{m['std_r2']:.3f})"

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def _normalize_text(s):
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

def _is_p_gingivalis_model(name):
    key = _normalize_text(name)
    return ('pgingivalis' in key) or ('porphyromonasgingivalis' in key)

def _calculate_overall_pvalue(models):
    """5개 모델 전체 차이 검정: paired면 Friedman, fallback Kruskal-Wallis."""
    err_arrays = [np.asarray(m['abs_errors'], dtype=float) for m in models]
    if len(err_arrays) < 2:
        return np.nan, "N/A"

    lengths = [len(a) for a in err_arrays]
    if min(lengths) <= 1:
        return np.nan, "N/A"

    same_length = len(set(lengths)) == 1
    try:
        if same_length:
            _, p_val = stats.friedmanchisquare(*err_arrays)
            return p_val, "Friedman"
        _, p_val = stats.kruskal(*err_arrays)
        return p_val, "Kruskal-Wallis"
    except Exception:
        try:
            _, p_val = stats.kruskal(*err_arrays)
            return p_val, "Kruskal-Wallis"
        except Exception:
            return np.nan, "N/A"

def _assign_unified_model_colors(model_keys):
    """P. gingivalis는 톤다운 레드, 나머지는 비-적색 팔레트로 충돌 최소화."""
    color_map = {}
    red_idx = 0
    non_red_idx = 0

    for key in sorted(model_keys):
        model_name = key.rsplit('_', 1)[0]
        if _is_p_gingivalis_model(model_name):
            color_map[key] = P_GINGIVALIS_SHADES[red_idx % len(P_GINGIVALIS_SHADES)]
            red_idx += 1
        else:
            color_map[key] = NON_RED_DISTINCT[non_red_idx % len(NON_RED_DISTINCT)]
            non_red_idx += 1
    return color_map

def plot_violin(models, target_gas, color_map):
    plot_data = []
    model_labels = []
    current_palette = []
    for m in models:
        unique_key = f"{m['name']}_{m['type']}"
        if unique_key in color_map:
            current_palette.append(color_map[unique_key])
        else:
            current_palette.append('#333333')

    for m in models:
        label = format_label_full(m)
        model_labels.append(label)
        for err in m['abs_errors']:
            plot_data.append({'Model': label, 'Absolute Error': err})
            
    df_plot = pd.DataFrame(plot_data)
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=model_labels, ordered=True)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=700)
    palette_map = {label: color for label, color in zip(model_labels, current_palette)}
    
    sns.violinplot(x='Model', y='Absolute Error', data=df_plot,
                   hue='Model', dodge=False, palette=palette_map,
                   inner='quartile', linewidth=1.5, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    sns.pointplot(x='Model', y='Absolute Error', data=df_plot,
                  estimator=np.mean, errorbar='sd', color='black',
                  capsize=0.1, markers='o', scale=0.7, linestyles='-', ax=ax)
    
    legend_elements = [
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, label='Quartiles\n(25%, Median, 75%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Mean'),
        Line2D([0], [0], color='black', linewidth=2, label='Std')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.89, 1.0),
              fontsize=10, frameon=True, framealpha=0.9)
    
    plt.title(f"Top 5 Models Absolute Error - {target_gas}", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel(f"Absolute Error ({target_gas})", fontsize=16, fontweight='bold')
    plt.xlabel("Model (Architecture)", fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold', rotation_mode='anchor')
    plt.yticks(fontsize=12, fontweight='bold')
    
    dx, dy = 10/72., 0/72. 
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels(): label.set_transform(label.get_transform() + offset)

    # 전체 5개 모델 분포 차이 p-value 표기
    p_val, method = _calculate_overall_pvalue(models)
    y_min = float(df_plot['Absolute Error'].min())
    y_max = float(df_plot['Absolute Error'].max())
    y_span = y_max - y_min if y_max > y_min else 1.0
    line_y = y_max + y_span * 0.08
    text_y = y_max + y_span * 0.12
    x_left, x_right = 0, len(models) - 1
    ax.plot([x_left, x_right], [line_y, line_y], color='black', linewidth=1.6)
    if np.isnan(p_val):
        p_text = "Overall p-value unavailable"
    elif p_val < 0.001:
        p_text = f"Overall {method} p < 0.001"
    else:
        p_text = f"Overall {method} p = {p_val:.4f}"
    ax.text((x_left + x_right) / 2, text_y, p_text, ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='black')
    ax.set_ylim(y_min - y_span * 0.05, y_max + y_span * 0.20)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"Violin_{target_gas}.png"), dpi=700)
    plt.close()

def plot_heatmap(models, target_gas):
    n = len(models)
    labels = [format_label_full(m) for m in models]
    p_values = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: p_values[i, j] = 1.0; continue
            try: _, p = stats.wilcoxon(models[i]['sq_errors'], models[j]['sq_errors'])
            except: p = 1.0
            p_values[i, j] = p
            
    fig, ax = plt.subplots(figsize=(11, 11), dpi=700)
    annot_labels = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j: annot_labels[i, j] = "-"
            else: annot_labels[i, j] = f"{p_values[i, j]:.5f}"
            
    cmap = ListedColormap([COLOR_SIG, COLOR_NOT_SIG])
    norm = BoundaryNorm([0, 0.05, 1.0], cmap.N)
    
    sns.heatmap(p_values, annot=annot_labels, fmt="", cmap=cmap, norm=norm,
                cbar=False, square=True, linewidths=1, linecolor='white', 
                annot_kws={"size": 18, "weight": "bold"}, ax=ax)
    
    legend_elements = [Patch(facecolor=COLOR_SIG, edgecolor='lightgray', label='Significant (P < 0.05)'),
                       Patch(facecolor=COLOR_NOT_SIG, edgecolor='lightgray', label='Not Significant')]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.2, 1.00), ncol=2, frameon=False, fontsize=14)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14, fontweight='bold', rotation_mode='anchor')
    ax.set_yticklabels(labels, rotation=0, fontsize=14, fontweight='bold')
    
    dx, dy = 10/72., 0/72. 
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels(): label.set_transform(label.get_transform() + offset)
    
    plt.title(f"Pairwise Wilcoxon Test - {target_gas}", fontsize=18, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"Heatmap_{target_gas}.png"), dpi=700)
    plt.close()

# ===================================================================
# 5. Feature Importance
# ===================================================================
def plot_feature_importance(model_info, target_gas, rank, color_map):
    m_name = model_info['name']
    m_type = model_info['type']
    safe_name = model_info['safe_name']
    
    unique_key = f"{m_name}_{m_type}"
    model_color = color_map.get(unique_key, '#333333')
    
    display_name = format_model_name_only(m_name)
    
    print(f"      [{rank}] Importance: {display_name} ({m_type})")
    
    TITLE_FONT = {'fontsize': 18, 'fontweight': 'bold'}
    LABEL_FONT = {'fontsize': 16, 'fontweight': 'bold'}
    TICK_FONT = {'fontsize': 14, 'fontweight': 'bold'}
    
    # 1. Ridge Coefficients
    if m_type == 'Ridge':
        coef_paths = [
            os.path.join(INPUT_DIR, target_gas, f"Coef_{safe_name}_{m_type}.csv"),
            os.path.join(INPUT_DIR, target_gas, f"Coef_{safe_name}.csv")
        ]
        coef_path = None
        for p in coef_paths:
            if os.path.exists(p): coef_path = p; break
            
        if coef_path:
            df_imp = pd.read_csv(coef_path)
            
            if 'Abs_Coeff' not in df_imp.columns:
                if 'Abs' in df_imp.columns:
                    df_imp['Abs_Coeff'] = df_imp['Abs']
                else:
                    df_imp['Abs_Coeff'] = df_imp['Coefficient'].abs()
            
            df_imp = df_imp.sort_values(by='Abs_Coeff', ascending=False)
            if len(df_imp) > 20: df_imp = df_imp.head(20)
                
            plt.figure(figsize=(12, max(6, len(df_imp)*0.6)), dpi=700)
            
            colors = ['#d62728' if x >= 0 else '#1f77b4' for x in df_imp['Coefficient']]
            
            ax = sns.barplot(x='Coefficient', y='Feature', data=df_imp, palette=colors, edgecolor='black')
            
            max_val = df_imp['Abs_Coeff'].max()
            offset = max_val * 0.02
            
            for p in ax.patches:
                val = p.get_width()
                y_pos = p.get_y() + p.get_height() / 2
                x_pos = val + offset if val >= 0 else val - offset
                ha = 'left' if val >= 0 else 'right'
                
                ax.text(x_pos, y_pos, f"{val:.4f}", va='center', ha=ha, 
                        fontsize=12, fontweight='bold', color='black')
            
            plt.title(f"Ridge Coefficients (Top {rank}): {display_name}", **TITLE_FONT)
            plt.xlabel("Coefficient Value", **LABEL_FONT)
            plt.ylabel("Feature", **LABEL_FONT)
            plt.xticks(**TICK_FONT)
            plt.yticks(**TICK_FONT)
            
            plt.xlim(-max_val*1.4, max_val*1.4)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"FeatImp_{target_gas}_Rank{rank}_{safe_name}.png"), dpi=700)
            plt.close()
            
    # 2. Transformer SHAP
    elif m_type == 'Transformer' and SHAP_AVAILABLE:
        artifact_path = os.path.join(INPUT_DIR, target_gas, "artifacts")
        meta_path = os.path.join(artifact_path, f"meta_{safe_name}.pkl")
        model_path = os.path.join(artifact_path, f"model_{safe_name}.pth")
        bg_path = os.path.join(artifact_path, f"background_{safe_name}.csv")
        
        if os.path.exists(meta_path) and os.path.exists(model_path) and os.path.exists(bg_path):
            with open(meta_path, 'rb') as f: meta = pickle.load(f)
            config = meta['config']; features = meta['features']; scaler_x = meta['scaler_x']
            
            df_bg = pd.read_csv(bg_path)
            X_bg = df_bg.values
            
            model = TransformerRegressor(len(features), config['k'], config['heads'], config['depth']).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            def predict_wrapper(x_numpy):
                x_scaled = scaler_x.transform(x_numpy)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
                with torch.no_grad(): out = model(x_tensor)
                return out.cpu().numpy().flatten()
            
            explainer = shap.KernelExplainer(predict_wrapper, X_bg)
            shap_values = explainer.shap_values(X_bg, nsamples=50, silent=True)
            
            plt.figure(figsize=(12, max(6, len(features)*0.7)), dpi=700)
            
            shap.summary_plot(shap_values, X_bg, feature_names=features, plot_type="bar", show=False, color=model_color)
            
            ax = plt.gca()
            max_val = 0
            for p in ax.patches:
                val = p.get_width()
                if val > max_val: max_val = val
                ax.text(val + (val * 0.02), p.get_y() + p.get_height()/2, f"{val:.4f}", 
                        fontsize=12, fontweight='bold', va='center', color='black')
            
            plt.title(f"SHAP Importance (Top {rank}): {display_name}", **TITLE_FONT)
            plt.xlabel("mean(|SHAP value|)", **LABEL_FONT)
            plt.yticks(**TICK_FONT)
            plt.xticks(**TICK_FONT)
            
            plt.xlim(0, max_val * 1.3)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"FeatImp_{target_gas}_Rank{rank}_{safe_name}.png"), dpi=700)
            plt.close()

# ===================================================================
# 6. Main Execution
# ===================================================================
if __name__ == "__main__":
    print(f"📂 Analyzing results from: {INPUT_DIR}")
    
    print("🔄 Pre-scanning all gases to assign unified colors...")
    all_gas_results = {}
    unique_models = set()
    
    for gas in TARGET_GASES:
        top5, _ = get_top5_with_summary_stats(gas)
        if top5:
            all_gas_results[gas] = {'top5': top5}
            for m in top5:
                unique_key = f"{m['name']}_{m['type']}"
                unique_models.add(unique_key)
    
    sorted_models = sorted(list(unique_models))
    model_color_map = _assign_unified_model_colors(sorted_models)
        
    print(f"🎨 Assigned colors for {len(unique_models)} unique model configurations.")
    
    with open("gas_model_color_map.json", "w") as f:
        json.dump(model_color_map, f)
    print("💾 Saved color map to 'gas_model_color_map.json'")

    for gas in TARGET_GASES:
        if gas not in all_gas_results:
            print(f"\n⚠️ No data found for {gas}.")
            continue
            
        print(f"\n🚀 Processing {gas}...")
        data = all_gas_results[gas]
        top5 = data['top5']
        
        print(f"   ✅ Top 5 Models Identified.")
        
        plot_violin(top5, gas, model_color_map)
        #plot_heatmap(top5, gas)
        
        # print("   ⭐ Generating Feature Importance Plots...")
        # for i, model in enumerate(top5):
        #      plot_feature_importance(model, gas, rank=i+1, color_map=model_color_map)
            
    print(f"\n✨ All visualizations saved to: {VIS_OUTPUT_DIR}")
# %%
