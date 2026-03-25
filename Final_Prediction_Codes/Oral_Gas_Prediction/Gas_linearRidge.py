#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon
import os
import itertools
from tqdm import tqdm
import warnings
from matplotlib.colors import ListedColormap, BoundaryNorm

# 경고 무시
warnings.filterwarnings('ignore')

# ===================================================================
# 1. Configuration
# ===================================================================
# 메인 출력 폴더
MAIN_OUTPUT_DIR = "analysis_result_ridge_final_triangle"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

FILE_PATH = "PCR_NGS_Data.xlsx"

# 타겟 가스
TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
CORRELATION_THRESHOLD = 0.8
TARGET_VAR_FOR_CLUSTERING = 'Halitosis'

# Hyperparameters (Ridge Only)
RIDGE_GRID = {'alpha': [0.01, 0.1, 1.0]}

OUTER_FOLDS = 3
INNER_FOLDS = 3
RANDOM_STATE = 42

def set_seed(seed=42):
    np.random.seed(seed)

set_seed(RANDOM_STATE)

# ===================================================================
# 2. Data Loading & Naming Logic
# ===================================================================
def format_bacterial_name(raw_name):
    """
    세균명을 학명 약어로 변환 (예: PCR_Fusobacterium_nucleatum -> F. nucleatum)
    """
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
    """상관관계 기반 클러스터 생성"""
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
# 3. Training & Evaluation (Ridge)
# ===================================================================
def train_ridge(X_tr, y_tr, config):
    model = Ridge(alpha=config['alpha'], random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return model

def evaluate_model_nested_ridge(feature_names, X_df, y):
    X_val = X_df[feature_names].values
    if X_val.ndim == 1: X_val = X_val.reshape(-1, 1)
        
    outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    y_trues, y_preds = [], []
    r2_scores = []
    feature_coeffs = np.zeros(len(feature_names))
    
    keys, values = zip(*RIDGE_GRID.items())
    grid_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for tr_idx, te_idx in outer_kf.split(X_val):
        X_tr, X_te = X_val[tr_idx], X_val[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        sc_x = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc_x.transform(X_tr), sc_x.transform(X_te)
        
        # Inner CV
        best_loss = float('inf'); best_config = grid_combos[0]
        inner_kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for config in grid_combos:
            fold_losses = []
            for i_tr, i_val in inner_kf.split(X_tr_s):
                m = train_ridge(X_tr_s[i_tr], y_tr[i_tr], config)
                p = m.predict(X_tr_s[i_val])
                fold_losses.append(mean_squared_error(y_tr[i_val], p))
            
            if np.mean(fold_losses) < best_loss:
                best_loss = np.mean(fold_losses); best_config = config
                
        # Final Train
        final_model = train_ridge(X_tr_s, y_tr, best_config)
        pred = final_model.predict(X_te_s)
        
        feature_coeffs += final_model.coef_
        y_trues.extend(y_te)
        y_preds.extend(pred)
        r2_scores.append(r2_score(y_te, pred))
        
    feature_coeffs /= OUTER_FOLDS
    return np.mean(r2_scores), np.array(y_trues), np.array(y_preds), feature_coeffs

# ===================================================================
# 4. Statistical Test & Visualization (Lower Triangle, 4 decimals)
# ===================================================================
def calculate_pairwise_statistics_and_plot(results_list, target_name, output_path):
    n = len(results_list)
    # 이름 간소화
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
            
            if not np.allclose(results_list[i]['y_true'], results_list[j]['y_true']):
                p_values[i, j] = 1.0; continue

            se_a = (results_list[i]['y_true'] - results_list[i]['y_pred']) ** 2
            se_b = (results_list[j]['y_true'] - results_list[j]['y_pred']) ** 2
            
            try: _, p = wilcoxon(se_a, se_b); p_values[i, j] = p
            except: p_values[i, j] = 1.0

    # Save P-values CSV
    df_pvals = pd.DataFrame(p_values, index=[r['ModelName'] for r in results_list], columns=[r['ModelName'] for r in results_list])
    df_pvals.to_csv(os.path.join(output_path, f"Stats_Pvalues_{target_name}.csv"))

    # --- [New Plotting Logic Applied] ---
    plt.figure(figsize=(28, 28)) # 요청하신 큰 사이즈 유지

    # Mask: Upper Triangle
    mask = np.triu(np.ones_like(p_values, dtype=bool), k=0)

    # Sig Matrix for Color (P < 0.05 only)
    sig_matrix = (p_values < 0.05).astype(int)
    
    # Custom Colormap: 0 -> whitesmoke, 1 -> #FFF700 (Yellow)
    cmap = ListedColormap(['whitesmoke', '#FFF700'])

    # Annotation DataFrame (4 decimal places)
    df_annot = pd.DataFrame(p_values)
    annot_labels = df_annot.applymap(lambda x: f"{x:.4f}")

    # Heatmap Drawing
    ax = sns.heatmap(sig_matrix, 
                     mask=mask, 
                     cmap=cmap, 
                     annot=annot_labels, 
                     fmt="", 
                     cbar=False, 
                     linewidths=0.5, 
                     linecolor='lightgrey', 
                     square=True,
                     annot_kws={"size": 14, "color": "black"})

    # Axis Labels Styling
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontweight='bold', fontsize=18)
    ax.set_yticklabels(display_names, rotation=0, ha='right', fontweight='bold', fontsize=18)

    # Bold Text for Significant Values (P < 0.05)
    for t in ax.texts:
        try:
            val_str = t.get_text()
            if not val_str: continue
            
            if float(val_str) < 0.05:
                t.set_weight('bold')
        except ValueError:
            pass

    plt.title(f"Pair-wise {target_name} Prediction Model Comparison by Wilcoxon Test \n(Yellow: P-Value < 0.05)", fontsize=40, pad=20)
    plt.tight_layout()
    
    save_file = os.path.join(output_path, f"pairwise_model_comparison_heatmap_{target_name}_bold.png")
    plt.savefig(save_file, dpi=700, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Heatmap saved to {save_file}")

def plot_feature_importance(coeffs, features, model_name, target_name, output_path):

    # 1. 데이터 프레임 생성 및 정렬 (절대값 기준 내림차순)
    df_imp = pd.DataFrame({
        'Feature': features,
        'Coefficient': coeffs,
        'Abs_Coeff': np.abs(coeffs)
    }).sort_values(by='Abs_Coeff', ascending=False)
    
    # 2. CSV 저장
    safe_fname = model_name.replace(".", "").replace(" ", "_")
    csv_filename = f"Feature_Importance_{target_name}_{safe_fname}.csv"
    df_imp.to_csv(os.path.join(output_path, csv_filename), index=False)
    print(f"   💾 Feature importance data saved: {csv_filename}")

    # 3. 그래프 설정
    # Feature 개수에 따라 그래프 세로 길이 자동 조절
    height = max(6, len(features) * 0.5)
    plt.figure(figsize=(12, height))
    
    # 색상 설정 (양수: 빨강, 음수: 파랑)
    colors = ['#d62728' if x >= 0 else '#1f77b4' for x in df_imp['Coefficient']]
    
    # Barplot 그리기
    ax = sns.barplot(x='Coefficient', y='Feature', data=df_imp, palette=colors, edgecolor='black')
    
    # 기준선 (0)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')

    # 4. [핵심] 텍스트 위치 자동 계산 및 Bold 출력
    # 데이터의 최대 절대값을 기준으로 텍스트 간격(offset)과 X축 범위 설정
    max_val = df_imp['Abs_Coeff'].max()
    offset = max_val * 0.02  # 막대 길이의 2% 만큼 띄움
    
    # X축 범위 확장 (글자가 잘리지 않도록 양옆으로 15% 여유 공간)
    plt.xlim(-max_val * 1.25, max_val * 1.25)

    for i, (val, patch) in enumerate(zip(df_imp['Coefficient'], ax.patches)):
        # 막대 중심 Y좌표
        y_pos = patch.get_y() + patch.get_height() / 2
        
        # 양수는 오른쪽, 음수는 왼쪽에 텍스트 배치
        if val >= 0:
            x_pos = val + offset
            ha = 'left'
        else:
            x_pos = val - offset
            ha = 'right'
        
        ax.text(x_pos, y_pos, 
                f"{val:.4f}",           # 소수점 4자리
                va='center', ha=ha,     # 수직 중앙 정렬
                fontweight='bold',      # [Bold] 값 굵게
                fontsize=11, 
                color='black')

    # 5. [핵심] 제목 및 축 라벨 Bold 처리
    plt.title(f"Feature Importance (Ridge): {model_name}\nTarget: {target_name}", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel("Regression Coefficient", fontsize=14, fontweight='bold')
    plt.ylabel("Feature", fontsize=14, fontweight='bold')
    
    # 6. [핵심] 눈금(Tick) 텍스트 Bold 처리
    # X축 눈금 (숫자)
    plt.xticks(fontsize=12, fontweight='bold')
    # Y축 눈금 (변수명)
    plt.yticks(fontsize=12, fontweight='bold')
    
    # 그리드 및 레이아웃
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 7. 고화질 저장
    save_filename = f"Feature_Importance_{target_name}_{safe_fname}.png"
    plt.savefig(os.path.join(output_path, save_filename), dpi=700, bbox_inches='tight')
    plt.close()
    print(f"   📊 Plot saved: {save_filename}")
# ===================================================================
# 5. Main Execution
# ===================================================================
def run_comparison_analysis():
    if not os.path.exists(FILE_PATH): print("❌ File not found."); return

    df, pcr_cols = load_and_preprocess_data(FILE_PATH)
    if df is None: return

    # 1. Models Setup
    print("🔍 Generating Models...")
    clusters = get_split_correlation_clusters(df, pcr_cols, CORRELATION_THRESHOLD)
    models_to_evaluate = clusters.copy()
    
    for pcr_col in pcr_cols:
        model_name = f"Single_{pcr_col}"
        models_to_evaluate[model_name] = [pcr_col]
        
    print(f"   -> Total models: {len(models_to_evaluate)}")

    if len(models_to_evaluate) == 0: print("⚠️ No models to evaluate."); return

    all_metrics_summary = []
    
    # 2. Iterate Targets
    for target in TARGET_GASES:
        if target not in df.columns: continue
        
        print(f"\n{'='*50}\n🔥 Analyzing Target: {target}\n{'='*50}")
        
        target_output_dir = os.path.join(MAIN_OUTPUT_DIR, target)
        os.makedirs(target_output_dir, exist_ok=True)

        df_tgt = df.dropna(subset=[target])
        y = df_tgt[target].values
        
        target_results = []
        
        # 3. Train & Evaluate
        for model_name, feats in tqdm(models_to_evaluate.items(), desc=f"Scanning ({target})"):
            r2, yt, yp, coefs = evaluate_model_nested_ridge(feats, df_tgt, y)
            target_results.append({
                'ModelName': model_name, 'Features': feats,
                'R2': r2, 'y_true': yt, 'y_pred': yp, 'Coeffs': coefs
            })

        # 4. Save Predictions
        for res in target_results:
            pred_df = pd.DataFrame({'y_true': res['y_true'], 'y_pred': res['y_pred']})
            safe_name = res['ModelName'].replace(".", "").replace(" ", "_")
            pred_df.to_csv(os.path.join(target_output_dir, f"Pred_{safe_name}.csv"), index=False)

        # 5. Statistical Test (Lower Triangle Heatmap)
        if len(target_results) > 1:
            calculate_pairwise_statistics_and_plot(target_results, target, target_output_dir)

        # 6. Feature Importance (Best Model)
        if len(target_results) > 0:
            sorted_res = sorted(target_results, key=lambda x: x['R2'], reverse=True)
            best_res = sorted_res[0]
            
            print(f"   🏆 Best Model: {best_res['ModelName']} | R2: {best_res['R2']:.3f}")
            if len(best_res['Features']) > 1:
                plot_feature_importance(best_res['Coeffs'], best_res['Features'], 
                                        best_res['ModelName'], target, target_output_dir)

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
    summary_path = os.path.join(MAIN_OUTPUT_DIR, "All_Models_Ridge_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Analysis Complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    run_comparison_analysis()
# %%
