#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===================================================================
# 설정
# ===================================================================
INPUT_DIR = "analysis_result_transformer_final_triangle_learning_1e-4" 
SUMMARY_FILE = "All_Models_Transformer_Summary.csv"
OUTPUT_FILE = "Integrated_Top3_Models_Rotated_On_Bar.png"

def plot_integrated_top3(input_dir, summary_file, output_filename):
    file_path = os.path.join(input_dir, summary_file)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 데이터 로드
    df = pd.read_csv(file_path)
    
    targets = df['Target'].unique()
    n_targets = len(targets)
    
    # 그래프 설정 (회전된 글씨 공간 확보를 위해 세로 길이 충분히 지정)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 9), constrained_layout=True)
    if n_targets == 1: axes = [axes]
        
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        # 상위 3개 모델 추출
        top3 = df[df['Target'] == target].sort_values(by='R2', ascending=False).head(3).reset_index(drop=True)
        
        # [중요] Y축 범위 설정: 회전된 글씨가 잘리지 않도록 상단 여유를 많이 둠 (1.35배)
        max_r2 = top3['R2'].max()
        if max_r2 > 0:
            ax.set_ylim(0, max_r2 * 1.4)
        else:
             ax.set_ylim(min(top3['R2']) * 1.4, 0)

        # 막대 그래프
        bars = ax.bar(top3.index, top3['R2'], color=colors[:len(top3)], edgecolor='black', width=0.88)
        
        # 타이틀 및 축 설정
        ax.set_title(f"Target: {target}", fontsize=18, fontweight='bold', pad=20)
        
        # X축은 다시 순위(Top N)로 표시
        ax.set_xticks(top3.index)
        ax.set_xticklabels([f"Top {j+1}" for j in top3.index], fontsize=14, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("R2 Score", fontsize=16, fontweight='bold')
            
        plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 라벨 추가
        for j, bar in enumerate(bars):
            height = bar.get_height()
            model_name = top3.loc[j, 'ModelName']
            r2_score = top3.loc[j, 'R2']
            
            # 1. R2 Score (막대 내부 중앙 or 바로 위)
            if abs(height) > (ax.get_ylim()[1] * 0.15):
                ax.text(bar.get_x() + bar.get_width()/2, height / 2, 
                        f"{r2_score:.3f}", 
                        ha='center', va='center', fontsize=14, fontweight='bold', color='black')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, height, 
                        f"{r2_score:.3f}", 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

            # 2. [수정] 모델명 (막대 위 + 45도 회전)
            # 막대 끝에서 조금 더 위쪽으로 띄움
            text_y = max(0, height) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            
            ax.text(bar.get_x(), text_y, 
                    f"{model_name}", 
                    rotation=45,        # 45도 회전
                    ha='left',          # 회전 시 왼쪽 정렬이 보기에 자연스러움 (사선 방향)
                    va='bottom', 
                    fontsize=13, 
                    fontweight='bold', 
                    color='black')

    fig.suptitle("Top 3 Transformer Models for Each Gas", fontsize=24, fontweight='bold', y=1.1)
    
    save_path = os.path.join(input_dir, output_filename)
    plt.savefig(save_path, dpi=700, bbox_inches='tight')
    print(f"✅ 그래프가 저장되었습니다: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_integrated_top3(INPUT_DIR, SUMMARY_FILE, OUTPUT_FILE)
# %%
