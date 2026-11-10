"""Local Streamlit demo for the contrastive Antariksh-JEPA retrieval model.

This app intentionally uses the locally evaluated contrastive checkpoint and the
matching held-out test indexes. It does not download or use the old prototype
checkpoint. Set ANTARIKSH_RUN_DIR only if the experiment artifacts live elsewhere.
"""

import copy
import io
import html
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
import streamlit as st
import streamlit.components.v1 as components
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from asset_cache import extract_test_tiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "notebooks" / "publication_runs_contrastive"
HF_MODEL_REPO = "ranjeetgupta/Cross-Modal_Antriksh_X-JEPA_model"
HF_DATASET_REPO = "ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2"


def download_contrastive_assets():
    """Download a single, matching checkpoint/index set from the Hub cache."""
    revision = os.environ.get("ANTARIKSH_MODEL_REVISION", "b0a88022965887cc615ac264b7109072e77689cb")
    names = ["best_checkpoint.pt", "final_test_evaluation/sar_test_192bit.index",
             "final_test_evaluation/optical_test_192bit.index",
             "final_test_evaluation/test_gallery_metadata.pkl"]
    paths = [Path(hf_hub_download(repo_id=HF_MODEL_REPO,
             filename=f"publication_runs_contrastive/{name}", revision=revision)) for name in names]
    run_dir = paths[0].parent
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("The contrastive checkpoint/index bundle is incomplete.")
    return run_dir


def prepare_cloud_test_dataset():
    """Cache the archive on the machine running Streamlit and extract test TIFFs."""
    revision = os.environ.get("ANTARIKSH_DATASET_REVISION", "main")
    parquet = hf_hub_download(repo_id=HF_DATASET_REPO, filename="metadata.parquet",
                              repo_type="dataset", revision=revision)
    archive = hf_hub_download(repo_id=HF_DATASET_REPO, filename="BigEarthNet_14K.zip",
                              repo_type="dataset", revision=revision)
    cache_root = Path(os.environ.get("ANTARIKSH_TEST_CACHE_DIR", PROJECT_ROOT / "archive" / "contrastive_test_cache"))
    return extract_test_tiles(archive, cache_root), Path(parquet)


class BEN14KDataset(Dataset):
    """Read the same test images, ordering, and preprocessing used in evaluation."""

    def __init__(self, data_dir, parquet_file, split="test"):
        metadata = pd.read_parquet(parquet_file)
        if "split" in metadata.columns:
            metadata = metadata[metadata["split"].str.lower() == split.lower()]
        s1_dir = Path(data_dir) / "BigEarthNet-S1" / split
        s2_dir = Path(data_dir) / "BigEarthNet-S2" / split
        self.valid_pairs, self.patch_ids, self.labels = [], [], []

        for _, row in metadata.iterrows():
            patch_id = row.get("patch_id", "")
            s1_name = row.get("s1_name", "")
            s2_name = row.get("s2v1_name", "")
            s1_candidates = [s1_dir / f"{patch_id}.tif", s1_dir / f"{s1_name}.tif"]
            s2_candidates = [s2_dir / f"{patch_id}.tif", s2_dir / f"{s2_name}.tif"]
            s1_path = next((path for path in s1_candidates if path.exists()), None)
            s2_path = next((path for path in s2_candidates if path.exists()), None)
            if s1_path and s2_path:
                self.valid_pairs.append((s1_path, s2_path))
                self.patch_ids.append(patch_id)
                self.labels.append(row.get("labels", []))

    def __len__(self):
        return len(self.valid_pairs)

    @contextmanager
    def open_raster(self, index, modality):
        path = self.valid_pairs[index][0 if modality == "sar" else 1]
        with rasterio.open(path) as source:
            yield source

    def __getitem__(self, index):
        with self.open_raster(index, "sar") as source:
            sar = source.read().astype(np.float32)
        with self.open_raster(index, "optical") as source:
            optical = source.read().astype(np.float32)
        sar = (np.clip(sar, -25.0, 0.0) + 25.0) / 25.0
        optical = np.clip(optical, 0.0, 10000.0) / 10000.0
        return torch.from_numpy(sar), torch.from_numpy(optical)


class RemoteBEN14KDataset(BEN14KDataset):
    """Read requested test TIFFs from a seekable remote ZIP into memory only."""

    def __init__(self, split="test"):
        revision = os.environ.get("ANTARIKSH_DATASET_REVISION", "main")
        self.fs = HfFileSystem()
        self.base = f"datasets/{HF_DATASET_REPO}@{revision}"
        self.archive_path = f"{self.base}/BigEarthNet_14K.zip"
        with self.fs.open(f"{self.base}/metadata.parquet", "rb", cache_type="none") as f:
            metadata = pd.read_parquet(io.BytesIO(f.read()))
        if "split" in metadata.columns:
            metadata = metadata[metadata["split"].str.lower() == split.lower()]
        with self.fs.open(self.archive_path, "rb", cache_type="none") as remote:
            with zipfile.ZipFile(remote) as archive:
                members = {}
                for name in archive.namelist():
                    parts = name.replace("\\", "/").split("/")
                    for i, part in enumerate(parts):
                        if part in ("BigEarthNet-S1", "BigEarthNet-S2") and parts[i + 1:i + 2] == [split] and len(parts) == i + 3:
                            key = (part, parts[-1])
                            if key in members:
                                raise ValueError(f"Duplicate ZIP member for {key}")
                            members[key] = name
        self.valid_pairs, self.patch_ids, self.labels = [], [], []
        for _, row in metadata.iterrows():
            patch_id = row.get("patch_id", "")
            s1_name, s2_name = row.get("s1_name", ""), row.get("s2v1_name", "")
            s1 = next((members.get(("BigEarthNet-S1", f"{value}.tif")) for value in (patch_id, s1_name) if members.get(("BigEarthNet-S1", f"{value}.tif"))), None)
            s2 = next((members.get(("BigEarthNet-S2", f"{value}.tif")) for value in (patch_id, s2_name) if members.get(("BigEarthNet-S2", f"{value}.tif"))), None)
            if s1 and s2:
                self.valid_pairs.append((s1, s2))
                self.patch_ids.append(patch_id)
                self.labels.append(row.get("labels", []))

    @contextmanager
    def open_raster(self, index, modality):
        member = self.valid_pairs[index][0 if modality == "sar" else 1]
        with self.fs.open(self.archive_path, "rb", cache_type="none") as remote:
            with zipfile.ZipFile(remote) as archive:
                content = archive.read(member)
        with MemoryFile(content) as memory:
            with memory.open() as source:
                yield source


class AntarikshViTEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1_encoder = timm.create_model(
            "vit_tiny_patch16_224", img_size=120, patch_size=8, in_chans=2, num_classes=0
        )
        self.s2_encoder = timm.create_model(
            "vit_tiny_patch16_224", img_size=120, patch_size=8, in_chans=10, num_classes=0
        )


class PositionalCrossPredictor(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, depth=4, num_shared_queries=8, num_patches=225):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.shared_queries = nn.Parameter(torch.zeros(1, num_shared_queries, embed_dim))
        self.target_position = nn.Embedding(num_patches, embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)

    def forward(self, source_visible, target_positions):
        batch_size, target_count = target_positions.shape
        masks = self.mask_token.expand(batch_size, target_count, -1)
        masks = masks + self.target_position(target_positions)
        queries = self.shared_queries.expand(batch_size, -1, -1)
        decoded = self.decoder(torch.cat([queries, masks], dim=1), source_visible)
        return decoded[:, queries.size(1) :]


class BinaryHashHead(nn.Module):
    def __init__(self, embed_dim, hash_bits):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hash_bits), nn.Tanh())

    def forward(self, features):
        return self.projection(features)


class ContrastiveAntarikshJEPA(nn.Module):
    """Checkpoint-compatible inference model for the publication contrastive run."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.online_encoders = AntarikshViTEncoders()
        self.target_encoders = copy.deepcopy(self.online_encoders)
        for parameter in self.target_encoders.parameters():
            parameter.requires_grad = False
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.optical_to_sar = PositionalCrossPredictor(config["embed_dim"], num_patches=self.num_patches)
        self.sar_to_optical = PositionalCrossPredictor(config["embed_dim"], num_patches=self.num_patches)
        self.hash_head = BinaryHashHead(config["embed_dim"], config["hash_bits"])


def select_tokens(tokens, mask):
    selected_per_item = int(mask.sum(dim=1)[0].item())
    return tokens[mask].reshape(tokens.size(0), selected_per_item, tokens.size(-1))


@torch.no_grad()
def deterministic_hash_code(model, image, modality, device):
    """Exactly matches deterministic_hash_codes() used in final_test_evaluation."""
    model.eval()
    target_count = int(model.num_patches * model.config["target_ratio"])
    positions = torch.arange(target_count, device=device).unsqueeze(0)
    visible = torch.ones(model.num_patches, dtype=torch.bool, device=device)
    visible[target_count : 2 * target_count] = False
    image = image.unsqueeze(0).to(device)

    if modality == "optical":
        tokens = model.online_encoders.s2_encoder.forward_features(image)[:, 1:]
        predicted = model.optical_to_sar(select_tokens(tokens, visible.unsqueeze(0)), positions)
    elif modality == "sar":
        tokens = model.online_encoders.s1_encoder.forward_features(image)[:, 1:]
        predicted = model.sar_to_optical(select_tokens(tokens, visible.unsqueeze(0)), positions)
    else:
        raise ValueError("modality must be optical or sar")

    continuous = model.hash_head(predicted.mean(dim=1)).cpu().numpy()
    return np.ascontiguousarray(np.packbits(continuous > 0, axis=1))


def labels_to_text(labels):
    if isinstance(labels, str):
        return labels
    if isinstance(labels, (list, tuple, np.ndarray)):
        return ", ".join(map(str, labels))
    return str(labels)


def _percentile_rescale(image, low=2.0, high=98.0):
    """Display-only contrast stretch; model inputs and retrieval codes are unchanged."""
    lower, upper = np.percentile(image, [low, high])
    return np.clip((image - lower) / max(upper - lower, 1e-8), 0.0, 1.0)


def optical_preview(tensor):
    """Consistent Sentinel-2 B04/B03/B02 true-colour-style preview (R/G/B)."""
    bands = tensor.detach().cpu().numpy()
    # Dataset band order begins B02, B03, B04. Reorder it to conventional RGB.
    rgb = np.moveaxis(bands[[2, 1, 0]], 0, -1)
    # Fixed reflectance scaling keeps hue comparable across tiles.  Per-tile RGB
    # percentile stretching can change the channel balance and create a green cast.
    rgb = np.clip(rgb / 0.30, 0.0, 1.0)
    return np.power(rgb, 1.0 / 2.2)


def sar_preview(tensor):
    image = tensor[0].numpy()
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def footprint_wgs84(dataset, index):
    """Return the true georeferenced footprint of a held-out tile in longitude/latitude."""
    # S1 and S2 of a pair are co-registered.  Sentinel-2 is used consistently here.
    with dataset.open_raster(index, "optical") as source:
        if source.crs is None:
            raise ValueError(f"No CRS is available for tile {dataset.patch_ids[index]}.")
        west, south, east, north = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
    polygon = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {
        "polygon": polygon,
        "longitude": (west + east) / 2.0,
        "latitude": (south + north) / 2.0,
        "bounds": (west, south, east, north),
    }


def _combined_map_zoom(footprints):
    """Choose a readable overview zoom from all footprint centres."""
    longitudes = [item["longitude"] for item in footprints]
    latitudes = [item["latitude"] for item in footprints]
    span = max(max(longitudes) - min(longitudes), max(latitudes) - min(latitudes))
    if span < 0.02:
        return 13
    if span < 0.10:
        return 11
    if span < 0.50:
        return 9
    if span < 2.0:
        return 7
    if span < 8.0:
        return 5
    return 3


def combined_footprint_map(query_footprint, gallery_footprints):
    """Return a Leaflet map with visible Q/rank badges inside every footprint."""
    rank_colours = [
        "#1e88e5", "#2563eb", "#0e7490", "#16a34a", "#7c3aed",
        "#d97706", "#be185d", "#0891b2", "#65a30d", "#4b5563",
    ]
    layer_data = [{
        "polygon": query_footprint["polygon"], "label": f"Q — Query: {query_footprint['patch_id']}",
        "marker": "Q", "colour": "#dc2626", "longitude": query_footprint["longitude"],
        "latitude": query_footprint["latitude"],
    }]
    for rank, footprint in enumerate(gallery_footprints, start=1):
        colour = rank_colours[(rank - 1) % len(rank_colours)]
        layer_data.append({
            "polygon": footprint["polygon"], "label": f"R{rank} — {footprint['patch_id']} | Hamming: {footprint['hamming_distance']}",
            "marker": str(rank), "colour": colour,
            "longitude": footprint["longitude"], "latitude": footprint["latitude"],
        })
    payload = json.dumps(layer_data)
    return f'''<!doctype html>
<html><head><meta charset="utf-8" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
html, body, #map {{ height: 100%; width: 100%; margin: 0; }}
.rank-badge {{ background: transparent; border: 0; }}
.rank-badge span {{ display:flex; width:34px; height:34px; align-items:center; justify-content:center;
  border-radius:50%; border:3px solid #0f172a; box-sizing:border-box; color:#fff;
  font:700 18px/1 Arial,sans-serif; text-shadow:0 1px 1px #0f172a; }}
.overlap-badges {{ display:flex; align-items:center; gap:2px; background:transparent; border:0; }}
.overlap-badges span {{ display:flex; width:34px; height:34px; align-items:center; justify-content:center;
  border-radius:50%; border:3px solid #0f172a; box-sizing:border-box; color:#fff;
  font:700 18px/1 Arial,sans-serif; text-shadow:0 1px 1px #0f172a; }}
</style></head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const items = {payload};
const map = L.map('map', {{scrollWheelZoom:true}});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Tiles &copy; Esri — Source: Esri, TomTom, Garmin, FAO, NOAA, USGS, OpenStreetMap contributors', maxZoom: 19
}}).addTo(map);
const allBounds = [];
const query = items[0];
const galleries = items.slice(1);
const sameCentre = (a, b) => Math.abs(a.latitude - b.latitude) < 1e-8 && Math.abs(a.longitude - b.longitude) < 1e-8;
const overlappingRanks = galleries.filter((item) => sameCentre(query, item));

// First draw translucent gallery fills, then the red query, then dashed gallery borders.
// This makes a shared footprint visibly purple/red-blue rather than allowing one polygon to hide another.
galleries.forEach((item) => {{
  const latLngs = item.polygon.map((point) => [point[1], point[0]]);
  const polygon = L.polygon(latLngs, {{color:item.colour, weight:0, fillColor:item.colour, fillOpacity:0.18}}).addTo(map);
  polygon.bindTooltip(item.label, {{sticky:true}});
  allBounds.push(...latLngs);
}});
const queryLatLngs = query.polygon.map((point) => [point[1], point[0]]);
const queryPolygon = L.polygon(queryLatLngs, {{color:query.colour, weight:5, fillColor:query.colour, fillOpacity:0.23}}).addTo(map);
queryPolygon.bindTooltip(query.label, {{sticky:true}});
allBounds.push(...queryLatLngs);
galleries.forEach((item) => {{
  const latLngs = item.polygon.map((point) => [point[1], point[0]]);
  const outline = L.polygon(latLngs, {{color:item.colour, weight:3, dashArray:'8 6', fill:false}}).addTo(map);
  outline.bindTooltip(item.label, {{sticky:true}});
}});

const marked = new Set();
if (overlappingRanks.length) {{
  const overlapHtml = `<div class="overlap-badges"><span style="background:${{query.colour}}">Q</span>${{overlappingRanks.map((item) => `<span style="background:${{item.colour}}">${{item.marker}}</span>`).join('')}}</div>`;
  const overlapBadge = L.divIcon({{className:'overlap-badges', html:overlapHtml, iconSize:[36 + 36 * overlappingRanks.length,34], iconAnchor:[18 + 18 * overlappingRanks.length,17]}});
  L.marker([query.latitude, query.longitude], {{icon:overlapBadge, keyboard:false, zIndexOffset:1000}}).addTo(map)
    .bindTooltip(`Exact footprint overlap: Q and rank ${{overlappingRanks.map((item) => item.marker).join(', ')}}`, {{direction:'top'}});
  marked.add('Q');
  overlappingRanks.forEach((item) => marked.add(item.marker));
}}
items.forEach((item) => {{
  if (marked.has(item.marker)) return;
  const badge = L.divIcon({{className:'rank-badge', html:`<span style="background:${{item.colour}}">${{item.marker}}</span>`, iconSize:[34,34], iconAnchor:[17,17]}});
  L.marker([item.latitude, item.longitude], {{icon:badge, keyboard:false}}).addTo(map).bindTooltip(item.label, {{direction:'top'}});
}});
map.fitBounds(L.latLngBounds(allBounds).pad(0.22));
</script></body></html>'''


def resolve_run_dir():
    return Path(os.environ.get("ANTARIKSH_RUN_DIR", str(DEFAULT_RUN_DIR))).expanduser().resolve()


@st.cache_resource(show_spinner="Loading contrastive checkpoint and gallery (first download may take time)...")
def load_system(run_dir_text):
    # Hub is the default source. An explicit run directory is an opt-in local override.
    run_dir = Path(run_dir_text) if "ANTARIKSH_RUN_DIR" in os.environ else download_contrastive_assets()
    checkpoint_path = run_dir / "best_checkpoint.pt"
    evaluation_dir = run_dir / "final_test_evaluation"
    required = [
        checkpoint_path,
        evaluation_dir / "sar_test_192bit.index",
        evaluation_dir / "optical_test_192bit.index",
        evaluation_dir / "test_gallery_metadata.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing contrastive assets:\n" + "\n".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    use_local = "ANTARIKSH_DATA_DIR" in os.environ or "ANTARIKSH_PARQUET_FILE" in os.environ
    if use_local:
        if not (os.environ.get("ANTARIKSH_DATA_DIR") and os.environ.get("ANTARIKSH_PARQUET_FILE")):
            raise ValueError("Set both ANTARIKSH_DATA_DIR and ANTARIKSH_PARQUET_FILE for a local dataset.")
        data_dir = Path(os.environ["ANTARIKSH_DATA_DIR"])
        parquet_file = Path(os.environ["ANTARIKSH_PARQUET_FILE"])
        if not data_dir.exists() or not parquet_file.exists():
            raise FileNotFoundError("Configured local dataset paths are unavailable.")

    model = ContrastiveAntarikshJEPA(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sar_index = faiss.read_index_binary(str(evaluation_dir / "sar_test_192bit.index"))
    optical_index = faiss.read_index_binary(str(evaluation_dir / "optical_test_192bit.index"))
    gallery_metadata = pd.read_pickle(evaluation_dir / "test_gallery_metadata.pkl")
    if not use_local:
        data_dir, parquet_file = prepare_cloud_test_dataset()
    test_dataset = BEN14KDataset(data_dir, parquet_file, split="test")
    if len(test_dataset) != len(gallery_metadata) or sar_index.ntotal != len(gallery_metadata) or optical_index.ntotal != len(gallery_metadata):
        raise RuntimeError(
            f"Index metadata has {len(gallery_metadata)} items but local test dataset has {len(test_dataset)}. "
            "Use the exact dataset used during final evaluation."
        )
    if "patch_id" not in gallery_metadata or list(map(str, gallery_metadata["patch_id"])) != list(map(str, test_dataset.patch_ids)):
        raise RuntimeError("Gallery metadata does not match the test dataset patch IDs in order. Refusing to display incorrect retrieval results.")
    return model, device, sar_index, optical_index, gallery_metadata, test_dataset, checkpoint["epoch"] + 1


st.set_page_config(page_title="Antariksh-JEPA Contrastive Retrieval", layout="wide")
st.title("Antariksh-JEPA: Contrastive Cross-Modal Retrieval")
st.caption("Contrastive checkpoint • 192-bit (24-byte) image hashes • held-out test gallery")

run_dir = resolve_run_dir()
try:
    model, device, sar_index, optical_index, gallery_metadata, test_dataset, best_epoch = load_system(str(run_dir))
except Exception as error:
    st.error(str(error))
    st.stop()

st.sidebar.success(f"Contrastive checkpoint loaded (best validation epoch {best_epoch})")
st.sidebar.caption("Model: Hugging Face cache" if "ANTARIKSH_RUN_DIR" not in os.environ else f"Run directory: {run_dir}")
st.sidebar.caption("Dataset: local override" if "ANTARIKSH_DATA_DIR" in os.environ else "Dataset: streamed from Hugging Face ZIP")
task = st.sidebar.radio(
    "Retrieval direction",
    ("Optical → SAR", "SAR → Optical", "Optical → Optical", "SAR → SAR"),
)
input_mode = st.sidebar.radio("Query source", ("Held-out test image", "Custom GeoTIFF"))
top_k = st.sidebar.slider("Retrieved images", min_value=1, max_value=10, value=5)

query = None
query_index = None
query_id = "Custom upload"
query_labels = "Unknown"
custom_query_footprint = None
custom_map_note = None

query_modality, target_modality = task.lower().split(" → ")

if input_mode == "Held-out test image":
    index = st.sidebar.slider("Test query index", 0, len(test_dataset) - 1, min(42, len(test_dataset) - 1))
    if st.sidebar.button("Run retrieval", type="primary"):
        sar, optical = test_dataset[index]
        query = optical if query_modality == "optical" else sar
        query_index = index
        query_id = test_dataset.patch_ids[index]
        query_labels = labels_to_text(test_dataset.labels[index])
else:
    upload = st.sidebar.file_uploader("Upload 120×120 GeoTIFF", type=("tif", "tiff"))
    if st.sidebar.button("Run custom retrieval", type="primary"):
        if upload is None:
            st.sidebar.error("Select a GeoTIFF first.")
        else:
            try:
                with MemoryFile(upload.getvalue()) as memory_file:
                    with memory_file.open() as source:
                        raw = source.read().astype(np.float32)
                        if source.crs is not None:
                            west, south, east, north = transform_bounds(
                                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
                            )
                            custom_query_footprint = {
                                "polygon": [[west, south], [east, south], [east, north], [west, north], [west, south]],
                                "longitude": (west + east) / 2.0,
                                "latitude": (south + north) / 2.0,
                                "bounds": (west, south, east, north),
                                "patch_id": upload.name,
                            }
                        else:
                            custom_map_note = "The uploaded GeoTIFF has no CRS/georeferencing metadata, so its spatial footprint cannot be placed on the map."
                expected_channels = 10 if query_modality == "optical" else 2
                if raw.shape != (expected_channels, 120, 120):
                    raise ValueError(f"Expected shape ({expected_channels}, 120, 120), received {raw.shape}.")
                if query_modality == "optical":
                    raw = np.clip(raw, 0.0, 10000.0) / 10000.0
                else:
                    raw = (np.clip(raw, -25.0, 0.0) + 25.0) / 25.0
                query = torch.from_numpy(raw)
                query_id = upload.name
                query_labels = "Unknown (custom GeoTIFF)"
            except Exception as error:
                st.sidebar.error(f"Cannot use this GeoTIFF: {error}")

if query is not None:
    with st.spinner("Creating 192-bit hash and searching the matching test gallery..."):
        started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if started:
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
        else:
            import time
            started_at = time.perf_counter()

        packed_query = deterministic_hash_code(model, query, query_modality, device)
        target_index = sar_index if target_modality == "sar" else optical_index
        search_count = min(target_index.ntotal, top_k + int(query_modality == target_modality and query_index is not None))
        distances, retrieved = target_index.search(packed_query, search_count)
        retrieved = list(map(int, retrieved[0]))
        distances = list(map(int, distances[0]))
        if query_modality == target_modality and query_index is not None:
            pairs = [(item, dist) for item, dist in zip(retrieved, distances) if item != query_index]
            retrieved, distances = zip(*pairs[:top_k]) if pairs else ([], [])
        else:
            retrieved, distances = retrieved[:top_k], distances[:top_k]

        if device.type == "cuda":
            ended.record()
            torch.cuda.synchronize()
            elapsed_ms = started.elapsed_time(ended)
        else:
            elapsed_ms = (time.perf_counter() - started_at) * 1000

    st.success(f"{task} completed in {elapsed_ms:.2f} ms (end-to-end hash inference + binary search).")
    query_column, information_column = st.columns([1, 2])
    with query_column:
        st.subheader("Query")
        if query_modality == "optical":
            st.image(optical_preview(query), clamp=True)
        else:
            st.image(sar_preview(query), clamp=True)
    with information_column:
        st.subheader("Query metadata")
        st.write(f"**Patch ID:** {query_id}")
        st.write(f"**Labels:** {query_labels}")
        st.write("**Representation:** one pooled 192-bit / 24-byte image hash")

    st.subheader(f"Top-{len(retrieved)} retrieved {target_modality.upper()} images")
    columns = st.columns(max(len(retrieved), 1))
    for rank, (match_index, distance) in enumerate(zip(retrieved, distances), start=1):
        sar_match, optical_match = test_dataset[match_index]
        metadata = gallery_metadata.iloc[match_index]
        with columns[rank - 1]:
            if target_modality == "sar":
                st.image(sar_preview(sar_match), clamp=True)
            else:
                st.image(optical_preview(optical_match), clamp=True)
            exact_pair = query_index is not None and match_index == query_index and query_modality != target_modality
            st.markdown(f"**Rank {rank}**{' · exact paired tile' if exact_pair else ''}")
            st.caption(f"Hamming distance: {distance}")
            st.caption(f"Patch ID: {metadata['patch_id']}")
            st.caption(f"Labels: {labels_to_text(metadata['labels'])}")

    # Render true GeoTIFF footprints for held-out queries and for georeferenced custom uploads.
    if retrieved and (query_index is not None or custom_query_footprint is not None):
        st.divider()
        st.subheader("Spatial footprints: query and all retrieved gallery tiles")
        st.caption("Red Q is the query footprint. Markers 1–k are retrieved gallery footprints in rank order. When footprints overlap exactly, adjacent Q + rank badges and red/blue outlined regions make the overlap explicit. This map supports spatial inspection; aggregate paired-retrieval metrics remain the formal evidence.")
        try:
            if query_index is not None:
                query_footprint = footprint_wgs84(test_dataset, query_index)
                query_footprint["patch_id"] = query_id
                query_role = "Query"
            else:
                query_footprint = custom_query_footprint
                query_role = "Custom GeoTIFF query"
            gallery_footprints = []
            location_rows = [{
                "marker": "Q", "role": query_role, "patch_id": query_id,
                "latitude": query_footprint["latitude"], "longitude": query_footprint["longitude"],
                "hamming_distance": None,
            }]
            for rank, (match_index, distance) in enumerate(zip(retrieved, distances), start=1):
                footprint = footprint_wgs84(test_dataset, match_index)
                footprint["patch_id"] = test_dataset.patch_ids[match_index]
                footprint["hamming_distance"] = distance
                gallery_footprints.append(footprint)
                location_rows.append({
                    "marker": str(rank), "role": "Exact paired tile" if query_index is not None and match_index == query_index else "Retrieved gallery tile",
                    "patch_id": footprint["patch_id"], "latitude": footprint["latitude"],
                    "longitude": footprint["longitude"], "hamming_distance": distance,
                })
            components.html(combined_footprint_map(query_footprint, gallery_footprints), height=520, scrolling=False)
            st.dataframe(pd.DataFrame(location_rows), use_container_width=True, hide_index=True)
        except Exception as error:
            st.warning(f"Combined footprint map could not be rendered: {error}")
    elif retrieved and custom_map_note:
        st.info(custom_map_note)
else:
    st.info("Choose a query image in the sidebar, then run retrieval.")
