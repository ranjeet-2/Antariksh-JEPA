🛰️**Antariksh-JEPA: A Noise-Robust Predictive Hashing Engine for Cross-Modal RS Image Retrieval**

**Problem Statement:** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data

📌 **Description**
Satellite remote sensing systems acquire Earth observation data using different sensors such as optical, multispectral, and Synthetic Aperture Radar (SAR). Each sensor captures different physical characteristics of the Earth's surface. Optical images provide rich visual and spectral information, while SAR images provide structural information and can operate under cloud cover and nighttime conditions.

With the rapid growth of satellite image archives, content-based retrieval of relevant images has become an important requirement. This project focuses on developing an efficient, multi-modal satellite image retrieval system capable of both same-modal and cross-modal retrieval.

The core of our solution is Antariksh-JEPA, which synthesizes the X-JEPA paradigm and Hirschfeld-Gebelein-Rényi Maximal Correlation (HGRMC) to tackle the inherent Modality Gap and Semantic Abstraction Gap. To ensure ultra-fast retrieval times required by ISRO, we employ Explicit Semantic Preserving Deep Hashing.

**🎯 Objectives Achieved**
Same-modal retrieval: Optical-to-Optical, SAR-to-SAR.
Cross-modal retrieval: Optical-to-SAR, SAR-to-Optical.
Ranking: Outputting the top-5 and top-10 most relevant images for each query.
Speed: O(1) average retrieval time via binary hashing and bitwise XOR operations (FAISS).
Noise Immunity: Resilient against SAR speckle and Optical cloud cover via latent space predictive alignment.

**🚀 Quick Start Guide**
**1\. Prerequisites**
Ensure you have Python 3.8+ installed. Install the required dependencies:
pip install -r requirements.txt

**2\. Download Dataset \& Model Weights**
Due to size constraints, the dataset and model weights are hosted on Hugging Face. The application is designed to automatically download these if they are missing, but you can also download them manually.

Automated Setup (Recommended):
Simply run the Streamlit app. The code will automatically download the 3.1GB dataset and the .pth model file to their correct directories.

Manual Setup:
Dataset: Download BigEarthNet\_14K.zip from [this Hugging Face link](https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/BigEarthNet_14K.zip) and place it in the archive/ directory..

Metadata: Download metadata.parquet from [here](https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/metadata.parquet) and place it in the archive/BEN\_14k/ folder.

Model: Download antariksh\_best\_model.pth from [this Hugging Face link](https://huggingface.co/ranjeetgupta/Cross-Modal_Antriksh_X-JEPA_model/resolve/main/saved_models/antariksh_best_model.pth) and place it in the saved\_models/ folder.

**3\. Run the Web Application**
Launch the interactive geospatial retrieval dashboard using Streamlit:
cd Web\_app
streamlit run app.py

**🧠 Methodology \& Architecture**
Our architecture bridges the modality gap without relying on inefficient pixel-level reconstruction.

Disjoint Masking \& Feature Extraction: ViT backbones encode Optical and SAR patches without spatial overlap.

Cross-Attentional Predictive Alignment (X-JEPA): The model predicts the latent representation of the target modality using context from the source modality.

HGR Maximal Correlation: Augments the Prediction Space Alignment (PSA) loss to filter out sensor-specific noise (e.g., speckle) and enforce orthogonal decorrelation.

Semantic Preserving Deep Hashing: Converts the 192-D real-valued vectors into compact binary hash codes using a tanh activation layer, enabling lightning-fast FAISS index searches.

Georeferenced Domain Adaptation: Leverages coordinates (Label-Deficit ZSL) to align modalities where human annotations are scarce.

(For detailed scientific foundations, refer to the papers in the research\_papers/ directory).
*\[1.Zero-Shot Minimal Supervision (Label-Deficit): We leverage georeferenced domain adaptation to align SAR and Optical modalities where human annotations are scarce. (Ref: Zero-Shot Cross-Modal Retrieval for RS Images)*

*2.Cross-Attentional Predictive Alignment (X-JEPA): The model abandons inefficient pixel-level reconstruction. Instead, it predicts the latent representation of the target modality using context from the source modality. (Ref: Choudhury et al., X-JEPA WACV 2026)*

*3.HGR Maximal Correlation: Augments the prediction loss to filter out sensor-specific noise (e.g., SAR speckle) and enforce orthogonal decorrelation, ensuring high F1-Scores even in noisy conditions. (Ref: Robust Cross-Modal Retrieval via Maximal Correlation)*

*4.Semantic Preserving Deep Hashing: Converts the 192-D continuous real-valued vectors into compact binary hash codes using a tanh layer, solving the computational bottleneck for massive EO archives. (Ref: Multisensor Fusion and Explicit Semantic Preserving Deep Hashing)]*

**📊 Evaluation \& Results**

The evaluation criteria mandate high F1-scores and low retrieval times. Our framework utilizes the Asymmetric Shared Space logic:
SAR Queries: Processed via the raw SAR Encoder.
Optical Queries: Processed via the Cross-Attentional Predictor to translate into the common latent space.
This ensures accurate distance calculations across all modes.
(The training convergence graph demonstrates the stability of the HGRMC augmented loss over 100 epochs).


**📁 Repository Structure**

ISRO/

│

├── Web\_app/

│   └── app.py                        # Streamlit UI \& Retrieval Logic (Downloads HF assets automatically)

│

├── notebooks/

│   ├── Master-Pipeline.ipynb         # Prototyping Notebook; Final Evaluation \& Visualization Notebook

│   ├── build\_faiss.ipynb             # Script to build binary FAISS indices

│   ├── evaluate.ipynb                # Evaluation metrics calculation

│   ├── train.ipynb                   # Main training loop

│   ├── losses.ipynb                  # Custom HGRMC \& PSA Loss definitions

│   ├── predictor.ipynb               # Cross-Attentional Predictor architecture

│   ├── x-jepa\_model.ipynb            # ViT Encoder definitions

│   └── Pytorch\_dataset\_class.ipynb   # BEN-14K DataLoader

│

├── research\_papers/                  # Foundational literature (X-JEPA, Deep Hashing, etc.)

│

├── assets/                           # Images for README and presentations

│

├── gallery\_metadata.pkl              # Pre-computed metadata for the gallery

├── optical\_gallery\_192bit.index      # FAISS Index for Optical Gallery

├── sar\_gallery\_192bit.index          # FAISS Index for SAR Gallery

├── requirements.txt                  # Python dependencies

└── .gitignore                        # Excludes large files and weights





🤝 Team



GeoOperations Explorer



Built for the Bharatiya Antariksh Hackathon 2026.

