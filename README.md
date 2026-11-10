##### **🛰️ Antariksh-JEPA: A Noise-Robust Predictive Hashing Engine for Cross-Modal RS Image Retrieval**

> **Publication experiment and deployment update:** The local `Web_app/app.py` uses the contrastive publication checkpoint and its matching held-out test indexes. Instructions further below describing `saved_models/antariksh_best_model.pth` refer to the older prototype, not the contrastive model. The following update supersedes those deployment instructions.

### Contrastive retrieval app

The app loads `publication_runs_contrastive/best_checkpoint.pt` and the matching binary FAISS indexes and gallery metadata from the [Hugging Face model repository](https://huggingface.co/ranjeetgupta/Cross-Modal_Antriksh_X-JEPA_model/tree/main/publication_runs_contrastive). The [dataset repository](https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/tree/main) contains `metadata.parquet` and a 3.14 GB ZIP. On first startup the app downloads the ZIP into the Hugging Face cache and extracts only held-out test TIFFs to the app server's disk; later queries read those files from disk. Startup requires free disk space for both ZIP and extracted tiles and may repeat after a cloud restart or cache reset. When the app runs in Streamlit Cloud this storage is on the server, not the browser user's computer. Running the app locally instead uses the local computer's disk. Only use trusted checkpoint, pickle and dataset artifacts.

Install dependencies with `pip install -r requirements.txt` and run `streamlit run Web_app/app.py`. For a local dataset, set both `ANTARIKSH_DATA_DIR` (the directory containing `BigEarthNet-S1` and `BigEarthNet-S2`) and `ANTARIKSH_PARQUET_FILE` (the metadata file). Optionally set `ANTARIKSH_RUN_DIR` to the matching contrastive run folder. `ANTARIKSH_MODEL_REVISION` and `ANTARIKSH_DATASET_REVISION` can pin Hugging Face revisions; the model defaults to the observed bundle commit and the dataset defaults to `main`.

`ANTARIKSH_TEST_CACHE_DIR` optionally changes the extracted-tile cache root. The Hugging Face cache location can be configured separately using `HF_HOME`. Do not commit the archive or extracted images to GitHub.

The two indexes, checkpoint, gallery metadata and underlying test images must come from the same evaluation run. The app verifies gallery counts and patch ID ordering; it does not establish that files or model weights are identical to those used in evaluation. A 192-bit hash occupies 24 bytes before index and metadata overhead. Semantic label overlap and exact paired retrieval measure different outcomes.

### Publication experiment figures

The public `notebooks/Master-Pipeline.ipynb` has saved outputs cleared and local Windows paths replaced with `CHANGE_ME_DATA_DIR` and `CHANGE_ME_RUNS_PARENT`. Set these placeholders to your dataset and experiment-output locations before running the affected cells. The original local notebook is unchanged.

The following figures are from the locally reported contrastive experiment; they are not proof of performance on other datasets.

- [Dataset and preprocessing](assets/publication/figure_2_dataset_preprocessing.png)
- [Training and retrieval architecture](assets/publication/figure_3_proposed_methodology_clean.png)
- [Training and validation loss](assets/publication/figure_4_training_validation_loss.png)
- [Baseline versus contrastive retrieval](assets/publication/figure_5_original_vs_contrastive_cross_modal.png)
- [Qualitative retrieval examples](assets/publication/figure_6_qualitative_retrieval_examples.png)





> 🔗 **Live Interactive App:** [Click here to launch Antariksh-JEPA](https://antariksh-jepa-cross-modal-retrieval-system.streamlit.app/)





**Problem Statement:** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data





\-----





###### 📌 **Description**



Satellite remote sensing systems acquire Earth observation data using different sensors such as optical, multispectral, and Synthetic Aperture Radar (SAR). Each sensor captures different physical characteristics of the Earth's surface. Optical images provide rich visual and spectral information, while SAR images provide structural information and can operate under cloud cover and nighttime conditions.



With the rapid growth of satellite image archives, content-based retrieval of relevant images has become an important requirement. This project focuses on developing an efficient, multi-modal satellite image retrieval system capable of both same-modal and cross-modal retrieval.



The core of our solution is Antariksh-JEPA, which synthesizes the X-JEPA paradigm and Hirschfeld-Gebelein-Rényi Maximal Correlation (HGRMC) to tackle the inherent Modality Gap and Semantic Abstraction Gap. To ensure ultra-fast retrieval times required by ISRO, we employ Explicit Semantic Preserving Deep Hashing.





\-----





###### **🎯 Objectives Achieved**



* **Same-modal retrieval:** Optical-to-Optical, SAR-to-SAR.
* **Cross-modal retrieval:** Optical-to-SAR, SAR-to-Optical.
* **Ranking:** Outputting the top-5 and top-10 most relevant images for each query.
* **Speed:** O(1) average retrieval time via binary hashing and bitwise XOR operations (FAISS).
* **Noise Immunity:** Resilient against SAR speckle and Optical cloud cover via latent space predictive alignment.





\-----





###### **🚀 Quick Start Guide**



1. **Prerequisites**



Ensure you have Python 3.8+ installed. Install the required dependencies:



```bash

pip install -r requirements.txt

```





**2. Download Dataset \& Model Weights**



Due to size constraints, the dataset and model weights are hosted on Hugging Face. The application is designed to automatically download these if they are missing, but you can also download them manually.



* **Automated Setup (Recommended):**

&#x20; Run the `Streamlit` app. The contrastive checkpoint, indexes and dataset ZIP are downloaded to the app server, and the test tiles are cached on its disk.



* **Manual Setup:**

  * **Dataset:** Download `BigEarthNet_14K.zip` from [this Hugging Face link](https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/BigEarthNet_14K.zip) and place it in the `archive/` directory.
  * **Metadata:** Download `metadata.parquet` from [here](https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/metadata.parquet) and place it in the `archive/BEN_14k/` folder.
  * **Model and indexes:** Use the matching [contrastive checkpoint and index bundle](https://huggingface.co/ranjeetgupta/Cross-Modal_Antriksh_X-JEPA_model/tree/main/publication_runs_contrastive); configure `ANTARIKSH_RUN_DIR` for a local copy.





**3. Run the Web Application**



Launch the interactive geospatial retrieval dashboard using Streamlit:



```bash
streamlit run Web_app\app.py

```





\-----





###### **🧠 Methodology \& Architecture**



Our architecture bridges the modality gap without relying on inefficient pixel-level reconstruction.



* **Disjoint Masking \& Feature Extraction:** `ViT` backbones encode Optical and SAR patches without spatial overlap.
* **Cross-Attentional Predictive Alignment (X-JEPA):** The model predicts the latent representation of the target modality using context from the source modality.
* **HGR Maximal Correlation:** Augments the Prediction Space Alignment (`PSA`) loss to filter out sensor-specific noise (e.g., speckle) and enforce orthogonal decorrelation.
* **Semantic Preserving Deep Hashing:** Converts the `192-D` real-valued vectors into compact binary hash codes using a `tanh` activation layer, enabling lightning-fast `FAISS` index searches.
* **Georeferenced Domain Adaptation:** Leverages coordinates (Label-Deficit `ZSL`) to align modalities where human annotations are scarce.





*(For detailed scientific foundations, refer to the papers in the `research_papers/` directory).*



1. **Zero-Shot Minimal Supervision (Label-Deficit):** We leverage georeferenced domain adaptation to align `SAR` and `Optical` modalities where human annotations are scarce. *(Ref: Zero-Shot Cross-Modal Retrieval for RS Images)*
2. **Cross-Attentional Predictive Alignment (X-JEPA):** The model abandons inefficient pixel-level reconstruction. Instead, it predicts the latent representation of the target modality using context from the source modality. *(Ref: Choudhury et al., X-JEPA WACV 2026)*
3. **HGR Maximal Correlation:** Augments the prediction loss to filter out sensor-specific noise (e.g., SAR speckle) and enforce orthogonal decorrelation, ensuring high F1-Scores even in noisy conditions. *(Ref: Robust Cross-Modal Retrieval via Maximal Correlation)*
4. **Semantic Preserving Deep Hashing:** Converts the `192-D` continuous real-valued vectors into compact binary hash codes using a `tanh` layer, solving the computational bottleneck for massive `EO` archives. *(Ref: Multisensor Fusion and Explicit Semantic Preserving Deep Hashing)*





\-----





###### **📊 Evaluation \& Results**



The evaluation criteria mandate high F1-scores and low retrieval times. Our framework utilizes the `Asymmetric Shared Space` logic:



* **SAR Queries:** Processed via the raw `SAR Encoder`.
* **Optical Queries:** Processed via the `Cross-Attentional Predictor` to translate into the common latent space.



This ensures accurate distance calculations across all modes.



(The training convergence graph demonstrates the stability of the HGRMC augmented loss over 100 epochs). Refer `assets/`





\-----





###### **📁 Repository Structure**



```text
ISRO/

├── Web_app/
│   └── app.py                        # Streamlit UI \& Retrieval Logic (Downloads HF assets automatically)
├── notebooks/
│   ├── Master-Pipeline.ipynb         # Prototyping Notebook; Final Evaluation \& Visualization Notebook
│   ├── build_faiss.ipynb             # Script to build binary FAISS indices
│   ├── evaluate.ipynb                # Evaluation metrics calculation
│   ├── train.ipynb                   # Main training loop
│   ├── losses.ipynb                  # Custom HGRMC \& PSA Loss definitions
│   ├── predictor.ipynb               # Cross-Attentional Predictor architecture
│   ├── x-jepa_model.ipynb            # ViT Encoder definitions
│   └── Pytorch_dataset_class.ipynb   # BEN-14K DataLoader
├── research_papers/                  # Foundational literature (X-JEPA, Deep Hashing, etc.)
├── assets/                           # Images for README and presentations
├── gallery_metadata.pkl              # Pre-computed metadata for the gallery
├── optical_gallery_192bit.index      # FAISS Index for Optical Gallery
├── sar_gallery_192bit.index          # FAISS Index for SAR Gallery
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Excludes large files and weights


```





\---





###### > **- GeoOperations Explorer**

> - Built for the Opto-SAR retrieval

