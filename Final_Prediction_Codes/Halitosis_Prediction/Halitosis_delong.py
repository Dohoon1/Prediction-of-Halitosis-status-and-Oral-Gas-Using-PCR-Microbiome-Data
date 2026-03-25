#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

# ===================================================================
# 1. 설정
# ===================================================================
OUTPUT_DIR = "Analysis_Result_Halitosis_NewMethod_Combined_corr0.4"

# ===================================================================
# 2. DeLong Test 함수들
# ===================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    
    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r, :] = compute_midrank(predictions_sorted_transposed[r, m:])
        
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    
    return aucs, delongcov

def calc_pvalue(y_true, prob_A, prob_B):
    if len(y_true) != len(prob_A) or len(y_true) != len(prob_B): 
        return 0, 1.0
    
    mask_pos = y_true == 1
    mask_neg = y_true == 0
    if np.sum(mask_pos) == 0 or np.sum(mask_neg) == 0: 
        return 0, 1.0 
        
    X_pos = np.vstack((prob_A[mask_pos], prob_B[mask_pos]))
    X_neg = np.vstack((prob_A[mask_neg], prob_B[mask_neg]))
    preds_sorted = np.hstack((X_pos, X_neg))
    
    aucs, delongcov = fast_delong(preds_sorted, np.sum(y_true == 1))
    
    l = np.array([1, -1])
    diff = np.diff(aucs)
    sigma = np.sqrt(np.dot(np.dot(l, delongcov), l.T))
    
    if sigma == 0: 
        return aucs, 1.0
        
    z = np.abs(diff) / sigma
    p_value = 2 * (1 - stats.norm.cdf(z))[0]
    
    return aucs, p_value

# ===================================================================
# 3. 데이터 로드 및 시각화 실행
# ===================================================================
def run_delong_top6_heatmap():
    summary_path = os.path.join(OUTPUT_DIR, "Classification_Summary.csv")
    if not os.path.exists(summary_path):
        print(f"❌ Cannot find {summary_path}")
        return

    df = pd.read_csv(summary_path)
    
    # 1. Top 6 모델 선정 (Cluster 3 + Single 3)
    cluster_mask = df['ModelName'].str.contains("Cluster")
    single_mask = df['ModelName'].str.contains("Single")
    
    top3_cluster = df[cluster_mask].sort_values(by='Mean_AUC', ascending=False).head(3)
    top3_single = df[single_mask].sort_values(by='Mean_AUC', ascending=False).head(3)
    
    top6_df = pd.concat([top3_cluster, top3_single])
    
    # 2. 각 모델의 예측(Prediction) 데이터 로드
    loaded_models = []
    print("📂 Loading prediction files for Top 6 models...")
    for _, row in top6_df.iterrows():
        name = row['ModelName']
        m_type = row['Type']

        # --- [수정] 출력용 이름 깔끔하게 변환 ---
        display_name = name
        if "Cluster" in name:
            num = name.split('_')[-1] # 마지막 숫자 추출
            display_name = f"Cluster {num}"
        # (Single_ 접두사는 사용자 지침에 따라 원본 name 그대로 유지)
        # -----------------------------------------

        safe_name = name.replace(".", "").replace(" ", "_")
        pred_file = os.path.join(OUTPUT_DIR, f"Pred_{safe_name}_{m_type}.csv")
        
        if os.path.exists(pred_file):
            dat = pd.read_csv(pred_file)
            loaded_models.append({
                'name': f"{display_name}\n({m_type})",                
                'y_true': dat['y_true'].values,
                'y_prob': dat['y_prob'].values
            })
        else:
            print(f"   ⚠️ Missing file: {pred_file}")
            
    n = len(loaded_models)
    if n < 2:
        print("❌ Not enough models loaded to perform DeLong test.")
        return

    print("🔄 Calculating DeLong P-values...")
    p_vals = np.zeros((n, n))
    labels = [m['name'] for m in loaded_models]
    
    # 3. P-value 계산 행렬 생성
    for i in range(n):
        for j in range(n):
            if i == j:
                p_vals[i, j] = 1.0 # 자기 자신과의 비교는 p=1.0 처리
            else:
                _, p = calc_pvalue(loaded_models[i]['y_true'], 
                                   loaded_models[i]['y_prob'], 
                                   loaded_models[j]['y_prob'])
                p_vals[i, j] = p
                
    # 4. Heatmap 그리기
    print("🎨 Generating Heatmap...")
    fig, ax = plt.subplots(figsize=(11, 10), dpi=700)
    
    # P-value 텍스트 포맷팅
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j: annot[i,j] = "-"
            elif p_vals[i,j] < 0.001: annot[i,j] = "<.001"
            else: annot[i,j] = f"{p_vals[i,j]:.3f}"
            
    # 색상 맵 설정 (진한 노랑색 적용)
    COLOR_SIG = '#FFD700'
    COLOR_NOT_SIG = '#F0F0F0'
    cmap = ListedColormap([COLOR_SIG, COLOR_NOT_SIG])
    norm = BoundaryNorm([0, 0.05, 1.0], cmap.N)
    
    sns.heatmap(p_vals, annot=annot, fmt="", cmap=cmap, norm=norm, 
                cbar=False, square=True, linewidths=1, linecolor='white',
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)
    
    # 범례(Legend)
    legend_elements = [Patch(facecolor=COLOR_SIG, edgecolor='lightgray', label='Significant (P < 0.05)'),
                       Patch(facecolor=COLOR_NOT_SIG, edgecolor='lightgray', label='Not Significant')]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.2, 1.0), ncol=2, frameon=False, fontsize=12)
    
    # 축 라벨 설정
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11, fontweight='bold', rotation_mode='anchor')
    ax.set_yticklabels(labels, rotation=0, fontsize=11, fontweight='bold')
    
    # 라벨 위치 미세조정
    dx, dy = 10/72., 0/72. 
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels(): 
        label.set_transform(label.get_transform() + offset)
    
    # 타이틀
    plt.title("Pairwise DeLong Test P-values \n (Top 3 Cluster and Top 3 Single Models)", fontsize=16, fontweight='bold', pad=40)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(OUTPUT_DIR, "DeLong_Heatmap_Top6_Standalone.png")
    plt.savefig(save_path, dpi=700)
    print(f"✅ Saved Heatmap to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_delong_top6_heatmap()
# %%
