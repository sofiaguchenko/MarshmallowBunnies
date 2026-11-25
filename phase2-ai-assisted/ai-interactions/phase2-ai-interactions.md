# Phase 2 AI Interactions Log — Team Marshmallow Bunnies (AI-Assisted Pair)

Log each AI-assisted session immediately after completing a module or notebook step. Tag every entry with the target file path so all code artifacts remain traceable.

---

## Entry 01 — Repository Architecture & Centralized Path Management

- **Target File(s):** `src/paths.py`, project directory tree
- **Goal & Prompt Summary:** Establish a modular repository layout separating reusable Python modules (`src/`), sequential experiment notebooks (`notebooks/`), cached outputs (`outputs/`), the Streamlit app (`app/`), and live AI logs (`ai-interactions/`) to prevent the relative path errors encountered in Phase 1.
- **AI Output / Artifact:** Generated directory layout and `src/paths.py` anchored via `Path(__file__).resolve().parents[1]` with `ensure_dirs()`.
- **Verification & Empirical Check:** Confirmed `PROJECT_ROOT` resolves consistently to `phase2-ai-assisted/` when imported from `notebooks/` and `app/`.
- **Decision:** **Accepted** (verify `IMAGES_DIR` and `METADATA_CSV` match the Human-only pair's dataset locations).

---

## Entry 02 — Visual Attribute Extraction & Text-Masking Module

- **Target File(s):** `src/visual_features.py`
- **Goal & Prompt Summary:** Build reusable functions to extract cover brightness, HSV color warmth/saturation, frontal face counts/area ratios, and OCR-based text masking (using Gaussian blur rather than black boxes to preserve luminance) to test visual stereotypes and title-text leakage in CLIP.
- **AI Output / Artifact:** Implemented `extract_visual_attributes()` (OpenCV HSV + Haar cascade face detection) and `mask_cover_text()` (EasyOCR detection + `(51, 51)` Gaussian-blur text replacement).
- **Verification & Empirical Check:** Pinned `"numpy<2"` and `"opencv-python-headless==4.11.0.86"` to resolve `CascadeClassifier` and NumPy 2.x incompatibilities. Visual inspection of blurred outputs confirmed dominant title/author text blocks are masked while preserving cover color palettes (e.g., 26.3% masked area on _Away in a Manger_). Noted a key limitation: heavily stylized or wide-tracked fonts at `224×224` resolution (e.g., _A Head Full of Ghosts_, 10.9% masked) partially evade EasyOCR at `min_conf=0.4`, making post-masking accuracy drops a conservative lower bound on title-text reliance.
- **Decision:** **Modified** — retained core module logic, pinned environment dependencies, and documented the stylized-font OCR sensitivity bound for the final report.

---

## Entry 03 — Dataset Schema Alignment & Feature Extraction Notebook

- **Target File(s):** `src/paths.py`, `notebooks/01_visual_and_ocr_features.ipynb`
- **Goal & Prompt Summary:** Align Phase 2 data loading with the repository's `archive/` layout and Phase 1's sampling design, then extract visual attributes and OCR-masked covers across harmonized benchmarks.
- **AI Output / Artifact:** Pointed `IMAGES_DIR` to `archive/title30cat/224x224/`, parsed semicolon-delimited `book30-listing-train.csv` and `book30-listing-test.csv` files, and extracted features for `6genre_p1_exact` (3,000 covers matching Phase 1's `random_state=42` sample) and `30genre_test` (3,000 stratified test covers, 100 per category).
- **Verification & Empirical Check:** Completed extraction for all 6,000 covers (`6000/6000`). Encountered a `pyarrow`/NumPy 2.x binary mismatch on `.to_parquet()`; exported intermediate results to `.csv`, restarted the kernel under `numpy<2`, and verified `outputs/visual_features.parquet` and all 6,000 masked images in `outputs/masked_covers/`.
- **Decision:** **Accepted (with CSV-to-Parquet recovery step)** — full 6,000-row feature table verified and persisted.

---

## Entry 04 — Controlled Prompt-Ladder & Margin Scoring Module

- **Target File(s):** `src/clip_eval.py`
- **Goal & Prompt Summary:** Create a modular CLIP inference pipeline using `openai/clip-vit-base-patch32` to evaluate prompt sensitivity systematically across a 4-level prompt ladder and compute top-1/top-2 similarity margins and calibrated probabilities.
- **AI Output / Artifact:** Implemented `PROMPT_LADDER` (`L1_minimal`, `L2_neutral`, `L3_stylistic_mood`, `L4_prompt_ensemble`), `load_clip()`, `build_genre_text_embeddings()` (with L2-normalized template averaging), `encode_images()`, and `score_embeddings()`.
- **Verification & Empirical Check:** Saved module and verified clean imports (`DEFAULT_MODEL_NAME`, `PROMPT_LADDER`, scoring utilities) inside Notebook 02.
- **Decision:** **Accepted.**

---

## Entry 05 — Prompt-Ladder & OCR-Masking Evaluation Notebook

- **Target File(s):** `notebooks/02_prompt_ladder_eval.ipynb`, `outputs/phase2_predictions.parquet`
- **Goal & Prompt Summary:** Encode original and text-masked cover embeddings once each with `openai/clip-vit-base-patch32` and evaluate all four prompt-ladder levels across both `6genre_p1_exact` and `30genre_test` benchmarks to quantify OCR leakage, prompt-induced flip rates, and similarity margins.
- **AI Output / Artifact:** Built 4-cell evaluation pipeline outputting `outputs/phase2_predictions.parquet` and a cross-benchmark summary table.
- **Verification & Empirical Check:** Completed inference across all 6,000 covers. On `6genre_p1_exact`, `L2_neutral` achieved 61.10% accuracy (dropping 7.37 pp to 53.73% when text-masked, with a 25.70% OCR flip rate), while `L4_prompt_ensemble` reached 61.17% and cut the OCR accuracy drop to 3.50 pp (57.67% masked). On `30genre_test`, `L2_neutral` scored 32.27% (25.00% masked, -7.27 pp), `L3_stylistic_mood` scored 33.00% (23.93% masked, -9.07 pp), and `L4_prompt_ensemble` led at 33.93% (25.90% masked) while reducing prompt-flip instability vs. `L2_neutral` from 47.97% (`L3`) to 35.30% (`L4`).
- **Decision:** **Accepted** — verified output table saved to `outputs/phase2_predictions.parquet`.

---

## Entry 06 — Per-Genre OCR Reliance, Visual Stereotype Pull & Calibration Analysis

- **Target File(s):** `notebooks/03_stereotype_analysis.ipynb`, `outputs/figures/fig1_ocr_and_prompt_sensitivity.png`, `fig2_visual_stereotypes.png`, `fig3_calibration_and_confusion.png`
- **Goal & Prompt Summary:** Decompose the aggregate prompt-ladder results by genre, test whether False Positives exhibit statistically significant visual stereotype shifts (brightness, HSV saturation, warm hue ratio, face area ratio) using Mann–Whitney U tests on text-masked predictions, and evaluate top-1/top-2 cosine-margin calibration.
- **AI Output / Artifact:** Generated three publication figures and statistical tables comparing per-genre OCR accuracy drops, False-Positive Z-score shifts against non-predicted covers, calibration curves across margin quintiles, and row-normalized confusion matrices before and after text masking.
- **Verification & Empirical Check:** Executed Notebook 03 end-to-end and verified all three figures saved to `outputs/figures/`.
- **Decision:** **Accepted.**

---

## Entry 07 — Black-Box Model Internals, Latent Concept Probing & Group Saliency

- **Target File(s):** `notebooks/04_model_internals_and_saliency.ipynb`, `outputs/figures/fig4_text_prototype_similarity.png`, `fig5_latent_concept_alignment.png`, `fig6_vit_patch_saliency.png`, `fig7_group_saliency_correct_vs_error.png`, `fig8_representation_space_knn.png`
- **Goal & Prompt Summary:** Move beyond input–output black-box testing to inspect `openai/clip-vit-base-patch32` internally across multiple interpretability techniques: (1) 512-D text-prototype cosine similarity, (2) semantic concept-axis projection (_Dark/Ominous_, _Portrait_, _Illustrated_, _Typography_), (3) `7×7` ViT-B/32 `[CLS]` attention-gradient saliency on individual OCR-flip cases and across a `n=300` correct-vs.-incorrect group comparison, and (4) 10-Nearest-Neighbor (`10-NN`) purity and 2D SVD geometry in image embedding space.
- **AI Output / Artifact:** Implemented eager-attention hooks on the final transformer encoder layer (`layers[-1].self_attn`) and concept-vector projections, producing Figures 4–8.
- **Verification & Empirical Check:** Verified all internal representation and group-level saliency plots rendered and exported cleanly.
- **Decision:** **Accepted.**

---

## Entry 08 — Interactive Streamlit Dashboard for Exploration & Live Testing

- **Target File(s):** `app/streamlit_app.py`
- **Goal & Prompt Summary:** Build an interactive exploration tool allowing users to inspect prompt-ladder metrics, filter OCR-flip error groups, compute live `7×7` ViT-B/32 patch saliency overlays, browse all Phase II figures, and test custom prompts against original vs. text-masked covers.
- **AI Output / Artifact:** Created 4-tab Streamlit application (`streamlit_app.py`) reading `outputs/phase2_predictions.parquet` and caching CLIP via `@st.cache_resource` for on-demand saliency and prompt experiments.
- **Verification & Empirical Check:** Verified app startup and interactive tabs against saved Phase II outputs.
- **Decision:** **Accepted.**

---

## Synthesis: Failure Patterns & Emerging Best Practices in AI-Assisted Analysis

### Observed AI Failure Patterns

1. **Unverified File-Schema & Directory Assumptions:** Initial AI-generated data loaders assumed comma-delimited CSVs without headers and nested genre folders (`archive/title30cat/<genre>/<file>`), whereas the actual BookCover30 archive used semicolon-delimited CSVs with headers and a flat `224x224/` directory.
2. **Transitive Dependency & Binary Drift:** Installing `easyocr` via unpinned `pip` upgraded OpenCV to 5.0 and NumPy to 2.2.6, breaking `cv2.CascadeClassifier` and `pyarrow` Parquet serialization in the active Miniconda environment.
3. **Black-Box Bias in Initial Experimental Design:** Without explicit prompting toward internal model representations, initial AI-proposed notebooks focused on external feature correlations and isolated example saliency maps rather than group-level saliency statistics and embedding geometry.
4. - **Verification & Empirical Check:** Initial run raised `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'` on live saliency/prompt tabs because Streamlit ran in a Python 3.13 / `transformers` 5.x environment where `get_text_features()` and `get_image_features()` return `BaseModelOutputWithPooling` instead of a raw `torch.Tensor`. Added a `_to_tensor()` compatibility helper unwrapping `.pooler_output` and verified all 4 dashboard tabs end-to-end.
5. 4. **Framework Major-Version API Drift (`transformers` 4.x vs. 5.x):** AI-generated CLIP helper functions assumed `get_text_features()` and `get_image_features()` returned raw `torch.Tensor` objects (`transformers` 4.x behavior), which failed under `transformers` 5.x until wrapped with `.pooler_output` extraction.

### Emerging Best Practices

1. **Ground Prompts With Exact Schemas & Snippets First:** Providing exact CSV header rows, directory diagnostics, and Phase I sampling code eliminated schema guesswork and ensured exact `random_state=42` alignment across phases.
2. **Modularize Logic into `src/` & Separate Expensive Extraction from Analysis:** Separating EasyOCR masking (`01`), CLIP embedding/scoring (`02`), statistical plots (`03`), and eager-attention hooks (`04`) into cached Parquet checkpoints prevented re-running 18-minute OCR jobs when iterating on visualizations.
3. **Require Hypothesis-Driven Group Comparisons:** Explicitly asking for aggregate comparisons across error groups (`n=300` title-band saliency ratios, `10-NN` neighborhood purity, and latent concept projections) turned qualitative visualizations into falsifiable evidence of how `ViT-B/32` relies on typography and visual tropes.
