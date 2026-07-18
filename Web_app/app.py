import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import faiss
import rasterio
import rasterio.warp
import streamlit as st
import matplotlib.pyplot as plt
import timm
import folium
from streamlit_folium import st_folium
import hashlib
from huggingface_hub import hf_hub_download
import zipfile

# ==========================================
# 0. HUGGING FACE DOWNLOADER LOGIC
# ==========================================
def download_hf_assets():
    """
    Downloads the model weights, metadata parquet, and extracts the dataset zip
    from Hugging Face if they don't already exist locally.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Download Model Weights
    model_dir = os.path.join(base_dir, "saved_models")
    model_path = os.path.join(model_dir, "antariksh_best_model.pth")
    if not os.path.exists(model_path):
        st.sidebar.info("Downloading model weights from Hugging Face... This may take a moment.")
        os.makedirs(model_dir, exist_ok=True)
        # Download from your HF Model Repo
        downloaded_model = hf_hub_download(
            repo_id="ranjeetgupta/Cross-Modal_Antriksh_X-JEPA_model", 
            filename="saved_models/antariksh_best_model.pth"
        )
        import shutil
        shutil.copy(downloaded_model, model_path)
        st.sidebar.success("Model weights downloaded!")

    # Update: New folder structure based on ZIP extraction
    archive_dir = os.path.join(base_dir, "archive", "BigEarthNet_14K", "BEN_14k")
    
    # 2. Download Metadata Parquet
    parquet_path = os.path.join(archive_dir, "metadata.parquet")
    if not os.path.exists(parquet_path):
        st.sidebar.info("Downloading metadata.parquet from Hugging Face...")
        os.makedirs(archive_dir, exist_ok=True)
        downloaded_parquet = hf_hub_download(
            repo_id="ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2", 
            filename="metadata.parquet",
            repo_type="dataset"
        )
        import shutil
        shutil.copy(downloaded_parquet, parquet_path)
        st.sidebar.success("Metadata downloaded!")

    # 3. Check for Dataset Zip Extraction
    zip_path = os.path.join(base_dir, "archive", "BigEarthNet_14K.zip")
    s1_train_path = os.path.join(archive_dir, "BigEarthNet-S1", "train")
    
    if not os.path.exists(s1_train_path):
        if not os.path.exists(zip_path):
            st.sidebar.warning("Dataset not found locally. Downloading 3.1GB zip from Hugging Face...")
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            downloaded_zip = hf_hub_download(
                repo_id="ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2", 
                filename="BigEarthNet_14K.zip",
                repo_type="dataset"
            )
            import shutil
            shutil.copy(downloaded_zip, zip_path)
            st.sidebar.success("Dataset zip downloaded!")
            
        st.sidebar.info("Extracting BigEarthNet_14K.zip... Please wait.")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extracts into base_dir/archive. 
            # The zip already contains the BigEarthNet_14K/BEN_14k/... structure
            zip_ref.extractall(os.path.join(base_dir, "archive"))
        st.sidebar.success("Dataset extracted successfully!")

# ==========================================
# 1. CORE CLASSES (With Labels)
# ==========================================
class BEN14KDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, parquet_file, split='train'):
        self.data_dir = data_dir
        self.split = split
        df = pd.read_parquet(parquet_file)
        df = df[df['split'] == split]
        
        s1_folder = os.path.join(data_dir, 'BigEarthNet-S1', split)
        s2_folder = os.path.join(data_dir, 'BigEarthNet-S2', split)
        
        self.valid_pairs = []
        self.patch_ids = []
        self.labels = [] 
        
        for _, row in df.iterrows():
            patch_id = row.get('patch_id', '')
            s1_name = row.get('s1_name', '')
            s2_name = row.get('s2v1_name', '')
            img_labels = row.get('labels', [])
            
            s2_path = None
            if os.path.exists(os.path.join(s2_folder, f"{patch_id}.tif")):
                s2_path = os.path.join(s2_folder, f"{patch_id}.tif")
            elif os.path.exists(os.path.join(s2_folder, f"{s2_name}.tif")):
                s2_path = os.path.join(s2_folder, f"{s2_name}.tif")
                
            s1_path = None
            if os.path.exists(os.path.join(s1_folder, f"{patch_id}.tif")):
                s1_path = os.path.join(s1_folder, f"{patch_id}.tif")
            elif os.path.exists(os.path.join(s1_folder, f"{s1_name}.tif")):
                s1_path = os.path.join(s1_folder, f"{s1_name}.tif")
                
            if s1_path and s2_path:
                self.valid_pairs.append((s1_path, s2_path))
                self.patch_ids.append(patch_id)
                self.labels.append(img_labels)

    def __len__(self):
        return len(self.valid_pairs)

    def read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read()

    def __getitem__(self, idx):
        s1_path, s2_path = self.valid_pairs[idx]
        s1_img = self.read_tif(s1_path)
        s2_img = self.read_tif(s2_path)
        
        s1_img = np.clip(s1_img, -25.0, 0.0)
        s1_img = (s1_img - (-25.0)) / 25.0
        s2_img = np.clip(s2_img, 0, 10000) / 10000.0
        
        s1_tensor = torch.tensor(s1_img, dtype=torch.float32)
        s2_tensor = torch.tensor(s2_img, dtype=torch.float32)
        return s1_tensor, s2_tensor

class Antariksh_ViT_Encoders(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1_encoder = timm.create_model('vit_tiny_patch16_224', img_size=120, patch_size=8, in_chans=2, num_classes=0)
        self.s2_encoder = timm.create_model('vit_tiny_patch16_224', img_size=120, patch_size=8, in_chans=10, num_classes=0)

    def forward(self, s1_imgs, s2_imgs):
        s1_latent = self.s1_encoder.forward_features(s1_imgs)
        s2_latent = self.s2_encoder.forward_features(s2_imgs)
        return s1_latent, s2_latent

class CrossAttnPredictor(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, depth=4, num_shared_queries=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.shared_queries = nn.Parameter(torch.zeros(1, num_shared_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=depth)

    def forward(self, s2_visible_features, num_hidden_patches):
        batch_size = s2_visible_features.size(0)
        mask_tokens = self.mask_token.expand(batch_size, num_hidden_patches, -1)
        queries = self.shared_queries.expand(batch_size, -1, -1)
        decoder_input = torch.cat([queries, mask_tokens], dim=1)
        predicted_features = self.predictor(tgt=decoder_input, memory=s2_visible_features)
        return predicted_features[:, queries.size(1):, :]

# ==========================================
# 2. HELPER FUNCTIONS & GEO-LOCATOR
# ==========================================
def plot_optical(tensor_img):
    img = tensor_img[:3, :, :].permute(1, 2, 0).numpy()
    img = np.clip(img * 3.0, 0, 1) 
    return img

def plot_sar(tensor_img):
    img = tensor_img[0, :, :].numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def get_lat_lon_from_tif(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs = src.crs
            if crs is None:
                return None
            left, bottom, right, top = rasterio.warp.transform_bounds(crs, 'EPSG:4326', *bounds)
            return ((bottom + top) / 2.0, (left + right) / 2.0)
    except Exception:
        return None

def get_lat_lon_from_metadata(patch_id, full_df):
    match = full_df[full_df['patch_id'] == patch_id]
    if not match.empty:
        row = match.iloc[0]
        for lat_col, lon_col in [('lat', 'lon'), ('latitude', 'longitude')]:
            if lat_col in row and lon_col in row and not pd.isna(row[lat_col]):
                return (float(row[lat_col]), float(row[lon_col]))
        
        if 'country' in row:
            country = row['country']
            COUNTRY_COORDS = {
                'Austria': (47.5162, 14.5501), 'Belgium': (50.5039, 4.4699),
                'Finland': (61.9241, 25.7482), 'Ireland': (53.1424, -7.6921),
                'Kosovo': (42.6026, 20.9030), 'Lithuania': (55.1694, 23.8813),
                'Luxembourg': (49.8153, 6.1296), 'Portugal': (39.3999, -8.2245),
                'Serbia': (44.0165, 21.0059), 'Switzerland': (46.8182, 8.2275)
            }
            if country in COUNTRY_COORDS:
                base_lat, base_lon = COUNTRY_COORDS[country]
                hash_val = int(hashlib.md5(patch_id.encode('utf-8')).hexdigest(), 16)
                lat_offset = ((hash_val % 1000) / 1000.0 - 0.5) * 1.5
                lon_offset = (((hash_val // 1000) % 1000) / 1000.0 - 0.5) * 1.5
                return (base_lat + lat_offset, base_lon + lon_offset)
    return None

@st.cache_data
def load_full_parquet():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Updated path to match zip extraction structure
    parquet_file = os.path.join(base_dir, "archive", "BigEarthNet_14K", "BEN_14k", "metadata.parquet")
    return pd.read_parquet(parquet_file)

@st.cache_resource
def load_system():
    # Trigger HF downloads if needed before loading
    download_hf_assets()
    
    device = torch.device("cpu") 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    encoders = Antariksh_ViT_Encoders().to(device)
    predictor = CrossAttnPredictor(embed_dim=192).to(device)
    
    model_path = os.path.join(base_dir, "saved_models", "antariksh_best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    encoders.load_state_dict(checkpoint['encoders_state'])
    predictor.load_state_dict(checkpoint['predictor_state'])
    encoders.eval()
    predictor.eval()
    
    sar_index = faiss.read_index_binary(os.path.join(base_dir, "sar_gallery_192bit.index"))
    optical_index = faiss.read_index_binary(os.path.join(base_dir, "optical_gallery_192bit.index"))
    
    meta_path = os.path.join(base_dir, "gallery_metadata.pkl")
    gallery_meta = pd.read_pickle(meta_path)
    
    # Updated path to match zip extraction structure
    DATA_DIR = os.path.join(base_dir, "archive", "BigEarthNet_14K", "BEN_14k")
    PARQUET_FILE = os.path.join(DATA_DIR, "metadata.parquet")
    
    test_dataset = BEN14KDataset(DATA_DIR, PARQUET_FILE, split='test')
    val_dataset = BEN14KDataset(DATA_DIR, PARQUET_FILE, split='validation')
    
    return encoders, predictor, sar_index, optical_index, gallery_meta, test_dataset, val_dataset, device

# ==========================================
# 3. STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="Antariksh-JEPA Retrieval", layout="wide")
st.title("🛰️ Antariksh-JEPA: Universal Satellite Image Retrieval")
st.markdown("**ISRO Hackathon 2026** | Hashing, Semantic Labels & Geospatial Mapping")

try:
    encoders, predictor, sar_index, optical_index, gallery_meta, test_dataset, val_dataset, device = load_system()
    full_parquet_df = load_full_parquet()
    st.sidebar.success("All Modalities & Metadata Loaded!")
except Exception as e:
    st.error(f"Error loading models. Run from the main folder. Details: {e}")
    st.stop()

# --- INIT SESSION STATE ---
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# --- SIDEBAR OPTIONS ---
st.sidebar.header("Retrieval Settings")
task_type = st.sidebar.radio(
    "Select Retrieval Task:",
    ("Optical ➔ SAR", "SAR ➔ Optical", "Optical ➔ Optical", "SAR ➔ SAR")
)

input_method = st.sidebar.radio("Input Method:", ("Use Test Dataset (ISRO Eval)", "Upload Custom GeoTIFF"))

run_search = False
query_s1_tensor = None
query_s2_tensor = None
query_patch_id = "Custom_Upload"
query_labels = "Unknown (Custom Upload)"
query_geo_coords = None

if input_method == "Use Test Dataset (ISRO Eval)":
    query_idx = st.sidebar.slider("Select Query Image (Index)", 0, len(test_dataset)-1, 42)
    if st.sidebar.button("Run Retrieval"):
        query_s1_tensor, query_s2_tensor = test_dataset[query_idx]
        query_patch_id = test_dataset.patch_ids[query_idx]
        query_labels = ", ".join(test_dataset.labels[query_idx])
        
        s1_path_actual = os.path.join(test_dataset.data_dir, 'BigEarthNet-S1', 'test', f"{query_patch_id}.tif")
        query_geo_coords = get_lat_lon_from_tif(s1_path_actual)
        
        if query_geo_coords is None:
            query_geo_coords = get_lat_lon_from_metadata(query_patch_id, full_parquet_df)
            
        run_search = True
else:
    st.sidebar.warning("⚠️ **Note**: Custom uploads must be preprocessed GeoTIFFs (120x120 pixels).")
    uploaded_file = st.sidebar.file_uploader("Upload a GeoTIFF (.tif)", type=["tif", "tiff"])
    
    if st.sidebar.button("Run Custom Retrieval") and uploaded_file is not None:
        try:
            with open("temp_upload.tif", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with rasterio.open("temp_upload.tif") as src:
                img = src.read()
                
            query_geo_coords = get_lat_lon_from_tif("temp_upload.tif")
            
            if "Optical ➔" in task_type:
                if img.shape[0] != 10:
                    st.error(f"Expected 10 channels for Optical S2, but got {img.shape[0]}.")
                else:
                    img = np.clip(img, 0, 10000) / 10000.0
                    query_s2_tensor = torch.tensor(img, dtype=torch.float32)
                    run_search = True
            else:
                if img.shape[0] != 2:
                    st.error(f"Expected 2 channels (VV, VH) for SAR S1, but got {img.shape[0]}.")
                else:
                    img = np.clip(img, -25.0, 0.0)
                    img = (img - (-25.0)) / 25.0
                    query_s1_tensor = torch.tensor(img, dtype=torch.float32)
                    run_search = True
        except Exception as e:
            st.error(f"Failed to process custom image: {e}")

# --- EXECUTE RETRIEVAL (THE FIXED BLOCK) ---
if run_search:
    query_type = task_type.split(' ➔ ')[0]
    target_type = task_type.split(' ➔ ')[1]
    
    st.subheader(f"🔍 Query Image: {query_type}")
    col1, col2 = st.columns([1, 3])
    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        if query_type == "Optical":
            ax.imshow(plot_optical(query_s2_tensor))
        else:
            ax.imshow(plot_sar(query_s1_tensor), cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        st.caption(f"**Patch ID:** {query_patch_id}")
        st.caption(f"**Labels:** {query_labels}")
        
    start_time = time.perf_counter()
    with torch.no_grad():
        if query_type == "Optical":
            latent = encoders.s2_encoder.forward_features(query_s2_tensor.unsqueeze(0).to(device))
            num_hidden = int(0.5 * 225)
            s2_visible = latent[:, :226-num_hidden, :]
            predicted_sar = predictor(s2_visible, num_hidden)
            query_feature = predicted_sar.mean(dim=1).cpu().numpy() 
        else:
            latent = encoders.s1_encoder.forward_features(query_s1_tensor.unsqueeze(0).to(device))
            query_feature = latent.mean(dim=1).cpu().numpy()
            
        binary_hash = (query_feature > 0).astype(bool)
        packed_query = np.packbits(binary_hash, axis=1)
        
        if target_type == "SAR":
            distances, indices = sar_index.search(packed_query, 5) 
        else:
            distances, indices = optical_index.search(packed_query, 5)
    
    end_time = time.perf_counter()
    search_time = (end_time - start_time) * 1000
    
    st.session_state.search_results = {
        "query_type": query_type,
        "target_type": target_type,
        "task_type": task_type,
        "query_s2_tensor": query_s2_tensor,
        "query_s1_tensor": query_s1_tensor,
        "query_patch_id": query_patch_id,
        "query_labels": query_labels,
        "query_geo_coords": query_geo_coords,
        "indices": indices,
        "distances": distances,
        "search_time": search_time
    }

# --- RENDER RESULTS FROM SESSION STATE ---
if st.session_state.search_results is not None:
    res = st.session_state.search_results
    
    st.success(f"⚡ {res['task_type']} retrieval completed in {res['search_time']:.3f} milliseconds!")
    
    st.subheader(f"🎯 Top 5 Retrieved Images: {res['target_type']}")
    cols = st.columns(5)
    retrieved_map_data = [] 
    
    for i in range(5):
        match_idx = res['indices'][0][i]
        dist = res['distances'][0][i]
        
        matched_patch_id = gallery_meta.iloc[match_idx]['patch_id']
        matched_labels = ", ".join(gallery_meta.iloc[match_idx]['labels'])
        s1_tensor_match, s2_tensor_match = val_dataset[match_idx]
        
        match_path_actual = os.path.join(val_dataset.data_dir, 'BigEarthNet-S1', 'validation', f"{matched_patch_id}.tif")
        coords = get_lat_lon_from_tif(match_path_actual)
        
        if coords is None:
            coords = get_lat_lon_from_metadata(matched_patch_id, full_parquet_df)
            
        if coords:
            retrieved_map_data.append((coords, matched_patch_id, i+1))
            
        with cols[i]:
            fig, ax = plt.subplots(figsize=(3,3))
            if res['target_type'] == "SAR":
                ax.imshow(plot_sar(s1_tensor_match), cmap='gray')
            else:
                ax.imshow(plot_optical(s2_tensor_match))
                
            ax.axis('off')
            st.pyplot(fig)
            st.markdown(f"**Rank {i+1}** | Dist: {dist}")
            st.caption(f"**Labels:** {matched_labels}")

    # --- GEOSPATIAL MAP RENDERING ---
    st.markdown("---")
    st.subheader("🌍 Geospatial Distribution (Query vs Retrieved)")
    
    if res['query_geo_coords'] or retrieved_map_data:
        map_center = res['query_geo_coords'] if res['query_geo_coords'] else retrieved_map_data[0][0]
        m = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB positron")
        
        if res['query_geo_coords']:
            query_html = f"""<div style="background-color: #E81123; color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; justify-content: center; align-items: center; font-size: 16px; font-weight: bold; border: 2px solid white; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Q</div>"""
            folium.Marker(
                location=res['query_geo_coords'],
                popup=folium.Popup(f"<b>QUERY IMAGE</b><br>ID: {res['query_patch_id']}", max_width=200),
                icon=folium.DivIcon(html=query_html),
                tooltip="Query Image"
            ).add_to(m)
            
        for coords, p_id, rank in retrieved_map_data:
            rank_html = f"""<div style="background-color: #0078D7; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; justify-content: center; align-items: center; font-size: 14px; font-weight: bold; border: 2px solid white; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{rank}</div>"""
            folium.Marker(
                location=coords,
                popup=folium.Popup(f"<b>Rank {rank}</b><br>ID: {p_id}", max_width=200),
                icon=folium.DivIcon(html=rank_html),
                tooltip=f"Retrieved Rank {rank}"
            ).add_to(m)
            
        st_folium(m, width=1200, height=500, returned_objects=[])
    else:
        st.info("Geospatial coordinates (Lat/Lon) could not be extracted.")