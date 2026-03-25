#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import re
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

# ===================================================================
# 1. 설정
# ===================================================================
DIR_RIDGE = "analysis_result_ridge_final_triangle"
DIR_TRANSFORMER = "analysis_result_transformer_final_triangle_learning_1e-4"

TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
RANDOM_STATE = 42

P_GINGIVALIS_COLOR = '#B55A5A'  # toned-down red
P_GINGIVALIS_SHADES = ['#B55A5A', '#9E4D4D', '#C77171']
DISTINCT_COLORS_NON_RED = ['#1f77b4', '#2ca58d', '#6a4c93', '#bcbd22', '#17becf', '#4c78a8', '#2e8b57']
COLOR_SIG = '#FFFF00'
COLOR_NOT_SIG = '#F0F0F0'

# ===================================================================
# 2. 유틸리티: 부트스트랩 Std 계산 & 데이터 로드
# ===================================================================
def calculate_bootstrap_std(y_true, y_pred, n_boot=1000):
    """
    예측 결과를 기반으로 부트스트랩을 수행하여 R2 Score의 표준편차(Std)를 추정합니다.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    r2_scores = []
    n_samples = len(y_true)
    
    for _ in range(n_boot):
        # 복원 추출 (Resampling with replacement)
        indices = rng.randint(0, n_samples, n_samples)
        
        # 샘플링 된 데이터의 타겟값이 하나뿐이면 R2 계산 불가하므로 스킵
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        r2 = r2_score(y_true[indices], y_pred[indices])
        r2_scores.append(r2)
    
    # 표준편차 반환
    return np.std(r2_scores)

def normalize_key(name):
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def normalize_text(text):
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def is_p_gingivalis_model(model_name):
    key = normalize_text(model_name)
    return ('pgingivalis' in key) or ('porphyromonasgingivalis' in key)

def build_violin_palette(top5_models):
    palette = []
    non_red_idx = 0
    p_gingivalis_idx = 0

    for m in top5_models:
        if is_p_gingivalis_model(m['name']):
            if p_gingivalis_idx == 0:
                palette.append(P_GINGIVALIS_COLOR)
            else:
                palette.append(P_GINGIVALIS_SHADES[p_gingivalis_idx % len(P_GINGIVALIS_SHADES)])
            p_gingivalis_idx += 1
        else:
            palette.append(DISTINCT_COLORS_NON_RED[non_red_idx % len(DISTINCT_COLORS_NON_RED)])
            non_red_idx += 1
    return palette

def calculate_overall_pvalue(top5_models):
    error_arrays = [np.asarray(m['abs_errors'], dtype=float) for m in top5_models]
    if len(error_arrays) < 2:
        return np.nan, "N/A"

    valid_lengths = [len(arr) for arr in error_arrays if len(arr) > 1]
    if len(valid_lengths) < 2:
        return np.nan, "N/A"

    same_length = len(set(valid_lengths)) == 1

    try:
        if same_length:
            _, p_val = stats.friedmanchisquare(*error_arrays)
            return p_val, "Friedman"
        _, p_val = stats.kruskal(*error_arrays)
        return p_val, "Kruskal-Wallis"
    except Exception:
        try:
            _, p_val = stats.kruskal(*error_arrays)
            return p_val, "Kruskal-Wallis"
        except Exception:
            return np.nan, "N/A"

def load_summary_r2_mapping(target_gas):
    mapping = {}
    
    # Ridge Summary
    for f in glob.glob(os.path.join(DIR_RIDGE, "*Summary.csv")):
        try:
            df = pd.read_csv(f)
            df_gas = df[df['Target'].astype(str).str.strip() == target_gas]
            for _, row in df_gas.iterrows():
                key = f"{normalize_key(row['ModelName'])}_ridge"
                mapping[key] = row['R2']
        except: pass

    # Transformer Summary
    for f in glob.glob(os.path.join(DIR_TRANSFORMER, "*Summary.csv")):
        try:
            df = pd.read_csv(f)
            df_gas = df[df['Target'].astype(str).str.strip() == target_gas]
            for _, row in df_gas.iterrows():
                key = f"{normalize_key(row['ModelName'])}_trans"
                mapping[key] = row['R2']
        except: pass
            
    return mapping

def load_regression_results(target_gas):
    # 1. Summary에서 Mean R2 로드
    r2_mapping = load_summary_r2_mapping(target_gas)
    model_data = []
    
    ridge_path = os.path.join(DIR_RIDGE, target_gas, "Pred_*.csv")
    trans_path = os.path.join(DIR_TRANSFORMER, target_gas, "Pred_*.csv")
    
    ridge_files = glob.glob(ridge_path)
    trans_files = glob.glob(trans_path)
    
    # 폴더 구조 Fallback
    if not ridge_files:
        ridge_files = glob.glob(os.path.join(DIR_RIDGE, "**", f"Pred_*{target_gas}*.csv"), recursive=True)
    if not trans_files:
        trans_files = glob.glob(os.path.join(DIR_TRANSFORMER, "**", "Pred_*.csv"), recursive=True)

    print(f"🔍 [{target_gas}] Found predictions: {len(ridge_files)} (Ridge), {len(trans_files)} (Trans)")
    
    def _process_files(file_list, algo_type):
        for f in file_list:
            if target_gas not in f: continue
            try:
                df = pd.read_csv(f)
                filename = os.path.basename(f)
                
                if algo_type == "Trans":
                    core_name = filename.replace("Pred_", "").replace("_Transformer.csv", "")
                    suffix = "_trans"
                else:
                    core_name = filename.replace("Pred_", "").replace(".csv", "")
                    suffix = "_ridge"
                
                norm_key = f"{normalize_key(core_name)}{suffix}"
                
                y_true = df['y_true'].values
                y_pred = df['y_pred'].values
                
                # 1. Mean R2: Summary 파일 우선 사용
                if norm_key in r2_mapping:
                    mean_r2 = r2_mapping[norm_key]
                else:
                    mean_r2 = r2_score(y_true, y_pred) # Fallback
                
                # 2. Std R2: Bootstrap으로 계산
                std_r2 = calculate_bootstrap_std(y_true, y_pred)
                
                abs_errors = np.abs(y_true - y_pred)
                sq_errors = (y_true - y_pred) ** 2
                
                viz_name = core_name.replace("_", " ") + f" ({'Ridge' if algo_type=='Ridge' else 'Transformer'})"
                
                model_data.append({
                    'name': viz_name,
                    'mean_r2': mean_r2, # Summary 값
                    'std_r2': std_r2,   # Bootstrap 값
                    'abs_errors': abs_errors,
                    'sq_errors': sq_errors
                })
            except Exception as e:
                pass

    _process_files(ridge_files, "Ridge")
    _process_files(trans_files, "Trans")
    
    return model_data

# ===================================================================
# 3. Top 5 선정
# ===================================================================
def get_top5_models(model_data):
    # Mean R2 기준 내림차순
    sorted_models = sorted(model_data, key=lambda x: x['mean_r2'], reverse=True)
    return sorted_models[:5]

# ===================================================================
# 4. Violin Plot (Mean ± Std 적용)
# ===================================================================
def plot_error_violin(top5_models, target_gas):
    plot_data = []
    model_labels = []
    for m in top5_models:
        # 이름 포맷팅: 줄바꿈 + Mean ± Std
        formatted_name = m['name'].replace(" (", "\n(")
        label = f"{formatted_name}\n($R^2$={m['mean_r2']:.3f}±{m['std_r2']:.3f})"
        model_labels.append(label)
        
        for err in m['abs_errors']:
            plot_data.append({
                'Model': label, 
                'Absolute Error': err,
            })
    
    df_plot = pd.DataFrame(plot_data)
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=model_labels, ordered=True)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=700)
    violin_palette = build_violin_palette(top5_models)
    palette_map = {label: color for label, color in zip(model_labels, violin_palette)}
    
    sns.violinplot(x='Model', y='Absolute Error', data=df_plot, 
                   hue='Model', dodge=False, palette=palette_map,
                   inner='quartile', linewidth=1.5, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.title(f"Top 5 Models Absolute Error Distribution - {target_gas}", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Model (Mean $R^2$ ± Bootstrap Std)", fontsize=14, fontweight='bold')
    plt.ylabel(f"Absolute Error ({target_gas})", fontsize=14, fontweight='bold')
    
    # X축 라벨 설정 (45도, 오른쪽 정렬)
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold', rotation_mode='anchor')
    
    # 라벨 오른쪽 미세 이동
    dx, dy = 10/72., 0/72. 
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

    # 전체 5개 모델 분포 차이 p-value (paired: Friedman, fallback: Kruskal-Wallis)
    p_val, method = calculate_overall_pvalue(top5_models)
    y_min = float(df_plot['Absolute Error'].min())
    y_max = float(df_plot['Absolute Error'].max())
    y_span = y_max - y_min if y_max > y_min else 1.0

    line_y = y_max + y_span * 0.08
    text_y = y_max + y_span * 0.12
    x_left = 0
    x_right = len(top5_models) - 1

    ax.plot([x_left, x_right], [line_y, line_y], color='black', linewidth=1.6)

    if np.isnan(p_val):
        p_text = "Overall p-value unavailable"
    elif p_val < 0.001:
        p_text = f"Overall {method} p < 0.001"
    else:
        p_text = f"Overall {method} p = {p_val:.4f}"

    ax.text((x_left + x_right) / 2, text_y, p_text, ha='center', va='bottom',
            fontsize=12, fontweight='bold')
    ax.set_ylim(y_min - y_span * 0.05, y_max + y_span * 0.20)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = f"Violin_Top5_Error_{target_gas}.png"
    plt.savefig(save_path, dpi=700)
    print(f"✅ Saved Violin Plot: {save_path}")
    plt.show()

# ===================================================================
# 5. Wilcoxon Heatmap (Mean ± Std 적용)
# ===================================================================
def plot_wilcoxon_heatmap(top5_models, target_gas):
    n = len(top5_models)
    
    # 히트맵 축 라벨에도 Mean ± Std 적용 (줄바꿈 포함)
    models = []
    for m in top5_models:
        clean_name = m['name'].replace(" (", "\n(")
        # 공간상 R2까지 넣으면 너무 길어질 수 있으나, "모든 곳"에 표시하길 원하셨으므로 포함
        # 글자 크기를 고려해 3줄로 포맷팅
        models.append(f"{clean_name}\n({m['mean_r2']:.2f}±{m['std_r2']:.2f})")
    
    p_values = np.zeros((n, n))
    print(f"🔄 Calculating pairwise Wilcoxon tests for {target_gas}...")
    
    for i in range(n):
        for j in range(n):
            if i == j: p_values[i, j] = 1.0; continue
            try: 
                _, p = stats.wilcoxon(top5_models[i]['sq_errors'], top5_models[j]['sq_errors'])
            except ValueError: p = 1.0
            p_values[i, j] = p

    fig, ax = plt.subplots(figsize=(11, 10), dpi=700) # 라벨 길어졌으므로 사이즈 약간 확대
    
    annot_labels = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j: annot_labels[i, j] = "-"
            elif p_values[i, j] < 0.001: annot_labels[i, j] = "<.001"
            else: annot_labels[i, j] = f"{p_values[i, j]:.3f}"

    cmap = ListedColormap([COLOR_SIG, COLOR_NOT_SIG])
    bounds = [0, 0.05, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)
    
    sns.heatmap(p_values, annot=annot_labels, fmt="", cmap=cmap, norm=norm,
                cbar=False, square=True, linewidths=1, linecolor='white', ax=ax)
    
    legend_elements = [
        Patch(facecolor=COLOR_SIG, edgecolor='lightgray', label='Significant (P < 0.05)'),
        Patch(facecolor=COLOR_NOT_SIG, edgecolor='lightgray', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1.02), 
              ncol=2, fontsize=10, frameon=False)

    # 축 라벨 설정
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9, fontweight='bold', rotation_mode='anchor')
    ax.set_yticklabels(models, rotation=0, fontsize=9, fontweight='bold')
    
    # 라벨 오른쪽 이동
    dx, dy = 10/72., 0/72.
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)
    
    # Bold Text Logic
    for text in ax.texts:
        try:
            val_str = text.get_text()
            if val_str == "-": continue
            if "<" in val_str or float(val_str) < 0.05:
                text.set_weight('bold')
                text.set_color('black')
            else:
                text.set_color('gray')
        except: pass

    plt.title(f"Pairwise Wilcoxon Test - {target_gas}", fontsize=14, fontweight='bold', pad=40)
    plt.tight_layout()
    
    save_path = f"Heatmap_Wilcoxon_{target_gas}.png"
    plt.savefig(save_path, dpi=700)
    print(f"✅ Saved Heatmap: {save_path}")
    plt.show()

# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    if not os.path.exists(DIR_RIDGE) or not os.path.exists(DIR_TRANSFORMER):
        print("❌ Data directories not found. Check paths.")
    else:
        for gas in TARGET_GASES:
            print(f"\n🚀 Processing {gas}...")
            
            # 1. 데이터 로드 (Mean + Boot_Std)
            model_data = load_regression_results(gas)
            
            if len(model_data) < 2:
                print(f"⚠️ Not enough models for {gas}. Skipping.")
                continue
                
            # 2. Top 5 선정
            top5 = get_top5_models(model_data)
            print(f"   Top 1 Model: {top5[0]['name']} (R2={top5[0]['mean_r2']:.3f}±{top5[0]['std_r2']:.3f})")
            
            # 3. Visualization
            plot_error_violin(top5, gas)
            plot_wilcoxon_heatmap(top5, gas)
            
        print("\n✨ All Gas Visualizations Completed.")
# %%
