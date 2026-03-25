#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import json
import matplotlib.colors as mcolors
import colorsys

# ===================================================================
# 1. 설정
# ===================================================================
OUTPUT_DIR = "Analysis_Result_Halitosis_NewMethod_Combined_corr0.4"
COLOR_MAP_FILE = "gas_model_color_map.json" # Gas 코드에서 생성한 파일

def to_violin_display_color(hex_color):
    """seaborn.violinplot 기본 saturation(0.75)이 적용된 실제 표시색으로 변환."""
    return mcolors.to_hex(sns.desaturate(hex_color, 0.75))

def build_bright_palette():
    return [
        "#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de", "#00c7be",
        "#ff2d55", "#5856d6", "#ffd60a", "#30b0c7", "#6acb2d", "#ff6b3d",
        "#1e90ff", "#ff4d6d", "#00b894", "#f39c12", "#8e44ad", "#2ecc71"
    ]

def plot_roc_with_unified_colors(output_dir, color_map_file, show_plot=True):
    # 1. Summary 파일 로드
    summary_path = os.path.join(output_dir, "Classification_Summary.csv")
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    
    # 2. Top 3 Cluster & Top 3 Single 선정 (Mean AUC 기준)
    cluster_df = df[df['ModelName'].str.contains("Cluster")].sort_values(by='Mean_AUC', ascending=False).head(3)
    single_df = df[df['ModelName'].str.contains("Single")].sort_values(by='Mean_AUC', ascending=False).head(3)
    top_models = pd.concat([cluster_df, single_df])
    
    print(f"📊 Selected Top 6 Models:")
    for _, row in top_models.iterrows():
        print(f"   - {row['ModelName']} ({row['Type']}): AUC {row['Mean_AUC']:.3f}")

    # 3. 색상 맵(Color Map) 연동
    global_color_map = {}
    if os.path.exists(color_map_file):
        with open(color_map_file, "r") as f:
            global_color_map = json.load(f)
        print(f"🎨 Successfully loaded Gas color map. ({len(global_color_map)} colors)")
    else:
        print(f"⚠️ Color map file not found ({color_map_file}). Will generate new colors.")

    # Violin 기준 색상 맵 (표시색 기준): 이 key는 반드시 고정
    violin_display_map = {k: to_violin_display_color(v) for k, v in global_color_map.items()}
    bright_palette = build_bright_palette()
    used_display_colors = set()
    used_display_colors_lower = set()

    # 4. Plotting 준비
    plt.figure(figsize=(10, 8), dpi=700)
    mean_fpr = np.linspace(0, 1, 100)
    color_rows = []
    palette_idx = 0
    
    for _, row in top_models.iterrows():
        name = row['ModelName']
        m_type = row['Type']
        
        # --- [수정] 출력용 이름 깔끔하게 변환 ---
        display_name = name
        if "Cluster" in name:
            num = name.split('_')[-1] # 마지막 숫자 추출
            display_name = f"Cluster {num}"
        # (Single_ 접두사는 사용자 지침에 따라 원본 name 그대로 유지)
        # -----------------------------------------

        # 모델 고유 키 생성 (Gas 방식과 동일: ModelName_Type)
        unique_key = f"{name}_{m_type}"
        
        # 색상 결정 (우선순위: Violin exact key 고정 > 밝은 팔레트 신규 배정)
        color_source = "bright_palette_new"
        line_color = None
        base_color = None
        if unique_key in violin_display_map:
            line_color = violin_display_map[unique_key]
            base_color = global_color_map.get(unique_key, line_color)
            color_source = "violin_exact_fixed"
        else:
            while True:
                if palette_idx < len(bright_palette):
                    cand = bright_palette[palette_idx]
                    palette_idx += 1
                else:
                    # 팔레트 소진 시 고채도 밝은 색상 추가 생성
                    hue = (palette_idx * 0.61803398875) % 1.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
                    cand = mcolors.to_hex(rgb)
                    palette_idx += 1
                if cand.lower() not in used_display_colors_lower:
                    line_color = cand
                    base_color = cand
                    break

        used_display_colors.add(line_color)
        used_display_colors_lower.add(line_color.lower())

        color_rows.append({
            "ModelName": name,
            "Type": m_type,
            "BaseColorHex": base_color,
            "ColorHex": line_color,
            "ColorSource": color_source
        })
            
        # 파일명 규칙에 맞춰 경로 생성
        safe_name = name.replace(".", "").replace(" ", "_")
        pred_file = os.path.join(output_dir, f"Pred_{safe_name}_{m_type}.csv")
        
        if not os.path.exists(pred_file):
            print(f"⚠️ Prediction file not found: {pred_file}")
            continue
            
        # 예측 결과 로드
        data = pd.read_csv(pred_file)
        y_true = data['y_true'].values
        y_prob = data['y_prob'].values
        
        # --- Bootstrap for Std (음영) Calculation ---
        tprs = []
        aucs = []
        n_bootstraps = 10000
        rng = np.random.RandomState(42)
        
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2: continue
            
            fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_prob[indices])
            interp_tpr = np.interp(mean_fpr, fpr_b, tpr_b)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr_b, tpr_b))
            
        # 평균 및 표준편차
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        
        # Use summary metrics for legend to keep figure text consistent
        # with Classification_Summary.csv.
        summary_auc = row['Mean_AUC']
        summary_std_auc = row['Std_AUC']
        
        # --- Plot 선 & 음영(Std) ---
        label_text = f"{display_name} ({m_type})\nAUC = {summary_auc:.3f} ± {summary_std_auc:.3f}"
        # 메인 곡선
        plt.plot(mean_fpr, mean_tpr, color=line_color, lw=2.5, label=label_text)
        
        # Std 음영
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1) # 상한선 1 제한
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0) # 하한선 0 제한
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=line_color, alpha=0.15)

    # 스타일링
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', alpha=0.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Halitosis Classification Models\n(Top 3 Cluster and Top 3 Single models)', fontsize=16, fontweight='bold', pad=15)
    
    # 폰트 및 라벨 설정
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='black')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(output_dir, "ROC_Top6_with_Std_Unified_Color.png")
    plt.savefig(save_path, dpi=700)
    pd.DataFrame(color_rows).to_csv(
        os.path.join(output_dir, "ROC_Top6_Color_Mapping.csv"), index=False
    )
    print(f"\n✅ ROC Plot saved to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    plot_roc_with_unified_colors(OUTPUT_DIR, COLOR_MAP_FILE, show_plot=False)
# %%
