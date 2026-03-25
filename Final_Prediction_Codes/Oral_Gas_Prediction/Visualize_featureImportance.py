#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import pickle
import math
import shap  # SHAP 라이브러리 필수
from sklearn.metrics import mean_squared_error

# ===================================================================
# 1. 설정
# ===================================================================
DIR_RIDGE = "analysis_result_ridge_final_triangle"
DIR_TRANSFORMER = "analysis_result_transformer_final_triangle_learning_1e-4"

TARGET_GASES = ['H2S_ppb', 'CH3SH_ppb', 'VSCs_ppb']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# 2. Transformer 모델 클래스 (불러오기를 위해 정의 필수)
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
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V).transpose(1,2).contiguous().view(b,t,h*d_k)
        return self.unifyheads(out), attn_weights

class TransformerEncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4*k), nn.ReLU(), nn.Linear(4*k, k))

    def forward(self, x):
        attended, weights = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x), weights

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, k, heads, depth, dropout=0.1): 
        super().__init__()
        self.embedding_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(1, k), nn.SiLU(), nn.Linear(k, k)) 
            for _ in range(seq_len)
        ])
        self.transformer_encoder = nn.ModuleList([TransformerEncoderBlock(k, heads) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(k, 1)

    def forward(self, x):
        embedded_list = [self.embedding_blocks[i](x[:, i].unsqueeze(-1)) for i in range(x.size(1))]
        x_emb = torch.stack(embedded_list, dim=1)
        attn_maps = []
        for layer in self.transformer_encoder:
            x_emb, weights = layer(x_emb)
            attn_maps.append(weights)
        x_pooled = x_emb.mean(dim=1)
        return self.fc(self.dropout(x_pooled)).squeeze(-1), attn_maps

# ===================================================================
# 3. Top 5 모델 선정 로직
# ===================================================================
def get_top5_models_from_summary(target_gas):
    model_list = []
    
    # Ridge
    for f in glob.glob(os.path.join(DIR_RIDGE, "*Summary.csv")):
        try:
            df = pd.read_csv(f)
            df = df[df['Target'] == target_gas]
            for _, row in df.iterrows():
                model_list.append({
                    'name': row['ModelName'],
                    'r2': row['R2'],
                    'type': 'Ridge',
                    'dir': os.path.join(DIR_RIDGE, target_gas)
                })
        except: pass

    # Transformer
    for f in glob.glob(os.path.join(DIR_TRANSFORMER, "*Summary.csv")):
        try:
            df = pd.read_csv(f)
            df = df[df['Target'] == target_gas]
            for _, row in df.iterrows():
                model_list.append({
                    'name': row['ModelName'],
                    'r2': row['R2'],
                    'type': 'Transformer',
                    'dir': os.path.join(DIR_TRANSFORMER, target_gas)
                })
        except: pass

    model_list.sort(key=lambda x: x['r2'], reverse=True)
    return model_list[:5]

# ===================================================================
# 4. Ridge 시각화 (Coefficient)
# ===================================================================
def visualize_ridge_coefficients(model_info, target_gas):
    model_name = model_info['name']
    base_dir = model_info['dir']
    safe_name = model_name.replace(".", "").replace(" ", "_")
    search_pattern = os.path.join(base_dir, f"Feature_Importance_*{safe_name}*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️ File not found for Ridge: {model_name}")
        return

    df_imp = pd.read_csv(files[0])
    
    plt.figure(figsize=(10, max(6, len(df_imp) * 0.4)), dpi=700)
    colors = ['#d62728' if x >= 0 else '#1f77b4' for x in df_imp['Coefficient']]
    
    ax = sns.barplot(x='Coefficient', y='Feature', data=df_imp, palette=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1)
    
    max_val = df_imp['Abs_Coeff'].max()
    offset = max_val * 0.05
    plt.xlim(-max_val * 1.3, max_val * 1.3)
    
    for i, (val, patch) in enumerate(zip(df_imp['Coefficient'], ax.patches)):
        y = patch.get_y() + patch.get_height() / 2
        align = 'left' if val >= 0 else 'right'
        pos = val + offset if val >= 0 else val - offset
        ax.text(pos, y, f"{val:.4f}", va='center', ha=align, fontsize=10, fontweight='bold', color='black')

    plt.title(f"Ridge Coefficients: {model_name}\nTarget: {target_gas} ($R^2$={model_info['r2']:.3f})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ===================================================================
# 5. Transformer 시각화 (SHAP Analysis)
# ===================================================================
def visualize_transformer_shap(model_info, target_gas):
    model_name = model_info['name']
    base_dir = model_info['dir']
    safe_name = model_name.replace(".", "").replace(" ", "_")
    artifact_dir = os.path.join(base_dir, "saved_artifacts")
    
    meta_path = os.path.join(artifact_dir, f"meta_{safe_name}.pkl")
    data_path = os.path.join(artifact_dir, f"background_data_{safe_name}.csv")
    model_path = os.path.join(artifact_dir, f"model_{safe_name}.pth")
    
    if not (os.path.exists(meta_path) and os.path.exists(data_path) and os.path.exists(model_path)):
        print(f"⚠️ Artifacts missing for Transformer: {model_name}")
        return

    # 1. Load Data & Config
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    config = meta['config']
    features = meta['features']
    scaler_x = meta['scaler_x']
    
    df_bg = pd.read_csv(data_path)
    X_bg = df_bg.values
    
    # 2. Re-build Model
    model = TransformerRegressor(len(features), config['k'], config['heads'], config['depth']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Define Predict Wrapper for SHAP
    # SHAP KernelExplainer expects a function that inputs numpy array and outputs numpy array
    def predict_wrapper(x_numpy):
        # x_numpy is typically unscaled if passed from DataFrame, but here we used Scaled training.
        # However, data_path saved original scale values? No, usually safer to save Original and scale inside.
        # Let's check: previous code saved `df_tgt[features].sample(...)`. This is ORIGINAL scale.
        
        # Scale Input
        x_scaled = scaler_x.transform(x_numpy)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            out, _ = model(x_tensor)
        
        # SHAP expects output shape (N, ) or (N, 1)
        return out.cpu().numpy().flatten()

    print(f"   🤖 Calculating SHAP values for {model_name}...")
    
    # 4. SHAP Kernel Explainer
    # Use the background data (approx 100 samples) as the reference distribution
    # link='identity' because we want to explain the raw output (StandardScaled Gas Concentration)
    explainer = shap.KernelExplainer(predict_wrapper, X_bg)
    
    # Calculate SHAP values for the background set (or a subset to be faster)
    shap_values = explainer.shap_values(X_bg, nsamples=100, silent=True)
    
    # 5. Visualization (Summary Plot - Bar)
    plt.figure(figsize=(10, max(6, len(features) * 0.5)), dpi=700)
    
    # plot_type="bar": 각 Feature의 평균적인 중요도(절대값 평균)를 보여줌
    shap.summary_plot(shap_values, X_bg, feature_names=features, plot_type="bar", show=False, color='#9467bd')
    
    # Title & Style
    plt.title(f"SHAP Feature Importance: {model_name}\nTarget: {target_gas} ($R^2$={model_info['r2']:.3f})", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("mean(|SHAP value|) (Average Impact on Model Output)", fontsize=12, fontweight='bold')
    
    # Save & Show
    plt.tight_layout()
    # Save separately if needed, here just show
    plt.show()

# ===================================================================
# 6. Main Execution
# ===================================================================
def run_top5_feature_analysis():
    print("🚀 Starting Top 5 Feature Importance Analysis (SHAP for Transformer)...")
    
    for gas in TARGET_GASES:
        print(f"\n{'='*60}")
        print(f"🔥 Analyzing Target: {gas}")
        print(f"{'='*60}")
        
        top5 = get_top5_models_from_summary(gas)
        
        if not top5:
            print(f"No models found for {gas}")
            continue
            
        print(f"✅ Top 5 Models identified (Best R2: {top5[0]['r2']:.3f})")
        
        for i, model in enumerate(top5):
            print(f"\n[{i+1}/5] Analyzing {model['name']} ({model['type']})...")
            
            if model['type'] == 'Ridge':
                visualize_ridge_coefficients(model, gas)
            
            elif model['type'] == 'Transformer':
                # [수정] Permutation 대신 SHAP 호출
                visualize_transformer_shap(model, gas)
                
    print("\n✨ All Analysis Completed.")

if __name__ == "__main__":
    if not os.path.exists(DIR_RIDGE) or not os.path.exists(DIR_TRANSFORMER):
        print("❌ Directories not found.")
    else:
        run_top5_feature_analysis()
# %%
