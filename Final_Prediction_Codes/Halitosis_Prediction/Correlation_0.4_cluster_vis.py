#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import warnings

warnings.filterwarnings('ignore')

# ===================================================================
# 1. 설정
# ===================================================================
FILE_PATH = "PCR_NGS_Data.xlsx"
OUTPUT_DIR = "Analysis_Result_Halitosis_NewMethod_Combined_corr0.4"
CORRELATION_THRESHOLD = 0.4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# 2. 데이터 로드
# ===================================================================
def format_bacterial_name(raw_name):
    clean = raw_name.replace("PCR_", "").replace("PCR", "").replace("__", " ").replace("_", " ").strip()
    parts = clean.split()
    if len(parts) >= 2: return f"{parts[0][0].upper()}. {parts[1]}"
    else: return clean

def load_data_for_clustering(file_path):
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
        pcr_columns_clean = []
        
        for col in df.columns:
            c_str = str(col).strip()
            if "PCR" in c_str:
                formatted_name = format_bacterial_name(c_str)
                selected[formatted_name] = pd.to_numeric(df[col], errors='coerce')
                pcr_columns_clean.append(formatted_name)
            elif 'Halitosis' in c_str:
                selected['Halitosis'] = pd.to_numeric(df[col], errors='coerce')
                
        df_clean = pd.DataFrame(selected).dropna(subset=['Halitosis'])
        return df_clean, pcr_columns_clean
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# ===================================================================
# 3. 개별 클러스터 독립 시각화 함수
# ===================================================================
def visualize_separate_clusters(df, pcr_features, threshold, output_dir):
    print(f"🔍 Analyzing Combined Clusters (Threshold = {threshold})...")
    
    # 1. 상관계수 행렬 계산 (절대값)
    corr_matrix = df[pcr_features].corr().abs().fillna(0)
    
    # 2. 전체 NetworkX 그래프 생성
    G = nx.from_pandas_adjacency(corr_matrix)
    
    # 3. Threshold 미만 엣지 및 Self-loop 제거
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w < threshold]
    G.remove_edges_from(edges_to_remove)
    G.remove_edges_from(nx.selfloop_edges(G)) # 본인-본인 연결 제거
    
    # 4. 연결된 컴포넌트(서로 묶인 클러스터 덩어리들) 추출
    # 노드가 2개 이상 연결된 덩어리만 추출
    components = [c for c in nx.connected_components(G) if len(c) >= 2]
    
    if not components:
        print("⚠️ No clusters found above the given threshold.")
        return

    print(f"✅ Found {len(components)} separate clusters. Generating plots...")

    # 5. 각 클러스터 덩어리별로 독립된 그래프 그리기
    for i, comp in enumerate(components):
        cluster_id = i + 1
        sub_G = G.subgraph(comp)
        
        # 클러스터 내의 노드 개수에 비례하여 이미지 크기 자동 조절 (최소 6x6)
        fig_size = max(6, len(comp) * 0.7) 
        plt.figure(figsize=(fig_size, fig_size), dpi=700)
        
        # 노드가 겹치지 않게 레이아웃 간격 조절 (k값)
        pos = nx.spring_layout(sub_G, k=2.5, seed=42)
        
        # 각 클러스터마다 노드 색상을 번갈아 다르게 지정해 구분감 부여
        node_color = '#add8e6' if cluster_id % 2 != 0 else '#ffcc99'
        
        # 노드 그리기
        nx.draw(sub_G, pos, with_labels=True, 
                node_color=node_color, edge_color='gray', 
                node_size=5000, font_size=10, font_weight='bold', alpha=0.95)
        
        # 엣지 라벨(상관계수) 추출 및 진하게(bold) 그리기
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in sub_G.edges(data=True)}
        
        nx.draw_networkx_edge_labels(sub_G, pos, 
                                     edge_labels=edge_labels, 
                                     font_color='darkred', 
                                     font_weight='bold', 
                                     font_size=11,
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # 제목 추가
        plt.title(f"Cluster {cluster_id} \n(Species: {len(comp)})", 
                  fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 개별 파일로 저장
        save_path = os.path.join(output_dir, f"Cluster_Network_Separate_C{cluster_id}.png")
        plt.savefig(save_path, dpi=700, bbox_inches='tight')
        plt.close()
        
        print(f"   -> Saved: {save_path} (Nodes: {len(comp)})")

# ===================================================================
# 4. 실행부
# ===================================================================
if __name__ == "__main__":
    df, pcr_cols = load_data_for_clustering(FILE_PATH)
    
    if df is not None:
        visualize_separate_clusters(df, pcr_cols, CORRELATION_THRESHOLD, OUTPUT_DIR)
# %%
