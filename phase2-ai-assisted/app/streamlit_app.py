import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Ensure phase2-ai-assisted/ is importable when running `streamlit run app/streamlit_app.py`
PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from src.clip_eval import DEFAULT_MODEL_NAME, PROMPT_LADDER, get_device
from src.paths import FIGURES_DIR, PREDICTIONS_PARQUET, SELECTED_6_GENRES

st.set_page_config(
    page_title="CLIP BookCover30 Bias & Internals Explorer",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PARQUET.exists():
        st.error(
            f"Missing predictions file: {PREDICTIONS_PARQUET}. Run Notebook 02 first."
        )
        st.stop()
    return pd.read_parquet(PREDICTIONS_PARQUET)


@st.cache_resource
def load_eager_clip() -> Tuple[CLIPModel, CLIPProcessor, str]:
    device = get_device()
    processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPModel.from_pretrained(
        DEFAULT_MODEL_NAME,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, processor, device


def _to_tensor(output) -> torch.Tensor:
    """Handle both transformers 4.x (Tensor) and 5.x (BaseModelOutputWithPooling)."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def compute_saliency_overlay(
    image_path: str,
    target_prompt: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 7x7 ViT-B/32 [CLS] attention-gradient saliency overlay for a target prompt."""
    model, processor, device = load_eager_clip()

    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB").resize((224, 224)))

    text_inputs = processor(text=[target_prompt], return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        text_tensor = _to_tensor(model.get_text_features(**text_inputs))
        text_embed = F.normalize(text_tensor, p=2, dim=-1)[0]

    img_inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    saved_attn: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        attn_weights = out[1]
        attn_weights.retain_grad()
        saved_attn["weights"] = attn_weights

    handle = model.vision_model.encoder.layers[-1].self_attn.register_forward_hook(
        hook_fn
    )
    try:
        model.zero_grad(set_to_none=True)
        vision_out = model.vision_model(
            pixel_values=img_inputs["pixel_values"], output_attentions=True
        )
        img_embed = F.normalize(
            model.visual_projection(vision_out.pooler_output), p=2, dim=-1
        )
        sim = (img_embed @ text_embed.unsqueeze(-1)).squeeze()
        sim.backward()

        attn = saved_attn["weights"]
        grad = attn.grad
        cls_cam = (attn * grad).clamp(min=0).mean(dim=1)[0, 0, 1:]
        grid = cls_cam.detach().cpu().numpy().reshape(7, 7)
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    finally:
        handle.remove()

    heatmap = cv2.resize(grid, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    overlay = np.uint8(0.55 * rgb + 0.45 * colored)
    return rgb, overlay


def score_custom_prompts(
    orig_path: str,
    mask_path: str,
    candidate_prompts: List[str],
) -> pd.DataFrame:
    """Score original vs. text-masked cover against custom text prompts."""
    model, processor, device = load_eager_clip()
    logit_scale = float(model.logit_scale.exp().item())

    with Image.open(orig_path) as im1, Image.open(mask_path) as im2:
        imgs = [im1.convert("RGB"), im2.convert("RGB")]

    with torch.no_grad():
        img_in = processor(images=imgs, return_tensors="pt").to(device)
        img_feats = F.normalize(
            _to_tensor(model.get_image_features(**img_in)), p=2, dim=-1
        )

        txt_in = processor(
            text=candidate_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        txt_feats = F.normalize(
            _to_tensor(model.get_text_features(**txt_in)), p=2, dim=-1
        )

        sims = (img_feats @ txt_feats.T).cpu()
        probs = torch.softmax(sims * logit_scale, dim=-1).numpy()
        sims_np = sims.numpy()

    return pd.DataFrame(
        {
            "Prompt": candidate_prompts,
            "Orig Cosine Sim": np.round(sims_np[0], 4),
            "Masked Cosine Sim": np.round(sims_np[1], 4),
            "Delta Sim (Masked - Orig)": np.round(sims_np[1] - sims_np[0], 4),
            "Orig Prob (%)": np.round(probs[0] * 100, 2),
            "Masked Prob (%)": np.round(probs[1] * 100, 2),
        }
    ).sort_values("Orig Cosine Sim", ascending=False)


# --- Header & Sidebar ---
df_all = load_predictions()

st.title("CLIP Zero-Shot Genre Bias & Black-Box Internals Explorer")
st.caption(
    "Phase II Investigation | Model: openai/clip-vit-base-patch32 | Dataset: BookCover30"
)

with st.sidebar:
    st.header("Evaluation Scope")
    benchmark = st.selectbox(
        "Benchmark Split",
        options=["6genre_p1_exact", "30genre_test"],
        format_func=lambda b: (
            "6-Genre Exact Phase I Sample (n=3,000)"
            if b == "6genre_p1_exact"
            else "30-Genre Stratified Test Set (n=3,000)"
        ),
    )
    prompt_level = st.selectbox(
        "Prompt Ladder Level",
        options=list(PROMPT_LADDER.keys()),
        index=1,
    )
    st.markdown("**Active Prompt Template(s):**")
    for tmpl in PROMPT_LADDER[prompt_level]:
        st.code(tmpl, language="text")

df_bench = df_all[df_all["benchmark"] == benchmark].reset_index(drop=True)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Prompt Ladder & OCR Leakage",
        "2. Cover & ViT Saliency Inspector",
        "3. Model Internals & Figures",
        "4. Live Prompt & Masking Lab",
    ]
)

# ==============================================================================
# TAB 1: PROMPT LADDER & OCR LEAKAGE
# ==============================================================================
with tab1:
    st.subheader("Causal Impact of Prompt Framing & Title-Text Masking")

    orig_col = f"orig_{prompt_level}_pred"
    mask_col = f"mask_{prompt_level}_pred"
    margin_col = f"orig_{prompt_level}_margin"

    orig_acc = (df_bench[orig_col] == df_bench["genre"]).mean() * 100
    mask_acc = (df_bench[mask_col] == df_bench["genre"]).mean() * 100
    ocr_drop = orig_acc - mask_acc
    ocr_flip = (df_bench[orig_col] != df_bench[mask_col]).mean() * 100
    mean_margin = df_bench[margin_col].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Cover Accuracy", f"{orig_acc:.2f}%")
    c2.metric("Text-Masked Accuracy", f"{mask_acc:.2f}%", delta=f"{-ocr_drop:.2f} pp")
    c3.metric("Prediction Flip Rate (OCR)", f"{ocr_flip:.2f}%")
    c4.metric("Mean Top-1 vs Top-2 Margin", f"{mean_margin:.4f}")

    st.markdown("#### Full 4-Level Prompt Ladder Comparison")
    ladder_rows = []
    neutral_pred = df_bench["orig_L2_neutral_pred"]
    for lvl in PROMPT_LADDER.keys():
        o_p = df_bench[f"orig_{lvl}_pred"]
        m_p = df_bench[f"mask_{lvl}_pred"]
        o_a = (o_p == df_bench["genre"]).mean() * 100
        m_a = (m_p == df_bench["genre"]).mean() * 100
        ladder_rows.append(
            {
                "Prompt Level": lvl,
                "Orig Acc (%)": round(o_a, 2),
                "Masked Acc (%)": round(m_a, 2),
                "OCR Drop (pp)": round(o_a - m_a, 2),
                "OCR Flip Rate (%)": round((o_p != m_p).mean() * 100, 2),
                "Flips vs L2_neutral (%)": round((o_p != neutral_pred).mean() * 100, 2),
                "Mean Margin": round(df_bench[f"orig_{lvl}_margin"].mean(), 4),
            }
        )
    st.dataframe(pd.DataFrame(ladder_rows), hide_index=True)

    st.markdown("#### Per-Genre Accuracy: Original vs. Text-Masked")
    genre_perf = []
    for g in sorted(df_bench["genre"].unique()):
        sub = df_bench[df_bench["genre"] == g]
        genre_perf.append(
            {
                "Genre": g,
                "Original (%)": round((sub[orig_col] == g).mean() * 100, 2),
                "Text-Masked (%)": round((sub[mask_col] == g).mean() * 100, 2),
                "OCR Drop (pp)": round(
                    ((sub[orig_col] == g).mean() - (sub[mask_col] == g).mean()) * 100, 2
                ),
                "Mean Text Area (%)": round(sub["text_area_ratio"].mean() * 100, 1),
            }
        )
    df_gp = pd.DataFrame(genre_perf).sort_values("OCR Drop (pp)", ascending=False)
    st.bar_chart(df_gp.set_index("Genre")[["Original (%)", "Text-Masked (%)"]])
    st.dataframe(df_gp, hide_index=True)

# ==============================================================================
# TAB 2: COVER & VIT SALIENCY INSPECTOR
# ==============================================================================
with tab2:
    st.subheader("Individual & Group Error Inspector (Original vs. Text-Masked)")

    f1, f2, f3 = st.columns(3)
    genres_avail = ["All Genres"] + sorted(df_bench["genre"].unique().tolist())
    sel_genre = f1.selectbox("Filter by True Genre", genres_avail)
    error_mode = f2.selectbox(
        "Prediction Behavior",
        [
            "OCR-Flip Error (Correct Orig -> Wrong Masked)",
            "Always Misclassified (Both Wrong)",
            "Robustly Correct (Both Correct)",
            "All Covers",
        ],
    )
    min_text_pct = f3.slider("Min OCR Blurred Area (%)", 0, 50, 5)

    filtered = df_bench[df_bench["text_area_ratio"] * 100 >= min_text_pct].copy()
    if sel_genre != "All Genres":
        filtered = filtered[filtered["genre"] == sel_genre]

    orig_ok = filtered[f"orig_{prompt_level}_pred"] == filtered["genre"]
    mask_ok = filtered[f"mask_{prompt_level}_pred"] == filtered["genre"]

    if error_mode.startswith("OCR-Flip"):
        filtered = filtered[orig_ok & (~mask_ok)]
    elif error_mode.startswith("Always Misclassified"):
        filtered = filtered[(~orig_ok) & (~mask_ok)]
    elif error_mode.startswith("Robustly Correct"):
        filtered = filtered[orig_ok & mask_ok]

    st.write(f"Matching covers: **{len(filtered):,}**")

    if len(filtered) == 0:
        st.warning(
            "No covers match the current filters. Try lowering the minimum OCR area or changing the filter."
        )
    else:
        max_idx = min(len(filtered) - 1, 99)
        sample_idx = st.number_input(
            "Select Cover Index (0 to 99)", min_value=0, max_value=max_idx, value=0
        )
        row = filtered.iloc[int(sample_idx)]

        st.markdown(
            f"**Title:** {row['title']}  |  **Author:** {row['author']}  |  **True Genre:** `{row['genre']}`"
        )
        st.caption(
            f"Extracted OCR Text: `{row['ocr_text']}` | Masked Area: `{row['text_area_ratio']:.1%}`"
        )

        col_orig, col_mask, col_meta = st.columns([1, 1, 1.3])
        with col_orig:
            st.image(row["image_path"], caption="Original Cover", width=240)
            st.write(
                f"**Orig Pred ({prompt_level}):** `{row[f'orig_{prompt_level}_pred']}`"
            )
            st.write(
                f"**2nd Choice:** `{row[f'orig_{prompt_level}_top2']}` (margin: `{row[f'orig_{prompt_level}_margin']:.4f}`)"
            )

        with col_mask:
            st.image(
                row["masked_image_path"], caption="OCR Text-Masked Cover", width=240
            )
            st.write(
                f"**Masked Pred ({prompt_level}):** `{row[f'mask_{prompt_level}_pred']}`"
            )
            st.write(
                f"**2nd Choice:** `{row[f'mask_{prompt_level}_top2']}` (margin: `{row[f'mask_{prompt_level}_margin']:.4f}`)"
            )

        with col_meta:
            st.markdown("**Extracted Visual Attributes**")
            attr_df = pd.DataFrame(
                {
                    "Attribute": [
                        "Brightness",
                        "Saturation",
                        "Warm Pixel Ratio",
                        "Face Count",
                        "Face Area Ratio",
                    ],
                    "Value": [
                        round(row["mean_brightness"], 3),
                        round(row["mean_saturation"], 3),
                        round(row["warm_pixel_ratio"], 3),
                        int(row["face_count"]),
                        round(row["face_area_ratio"], 4),
                    ],
                }
            )
            st.dataframe(attr_df, hide_index=True)

        show_sal = st.checkbox(
            "Compute Live ViT-B/32 Patch Saliency Heatmaps (7x7 Grid)"
        )
        if show_sal:
            with st.spinner("Computing ViT-B/32 attention-gradient saliency..."):
                true_prompt = f"A book cover of a {row['genre']} book."
                wrong_genre = row[f"mask_{prompt_level}_pred"]
                pred_prompt = f"A book cover of a {wrong_genre} book."

                _, sal_true = compute_saliency_overlay(row["image_path"], true_prompt)
                _, sal_orig_pred = compute_saliency_overlay(
                    row["image_path"], pred_prompt
                )
                _, sal_mask_pred = compute_saliency_overlay(
                    row["masked_image_path"], pred_prompt
                )

            s1, s2, s3 = st.columns(3)
            s1.image(sal_true, caption=f"Orig -> True ({row['genre']})", width=240)
            s2.image(
                sal_orig_pred, caption=f"Orig -> Masked Pred ({wrong_genre})", width=240
            )
            s3.image(
                sal_mask_pred,
                caption=f"Masked -> Masked Pred ({wrong_genre})",
                width=240,
            )

# ==============================================================================
# TAB 3: MODEL INTERNALS & SAVED FIGURES GALLERY
# ==============================================================================
with tab3:
    st.subheader("Black-Box Representations, Latent Concepts & Group Saliency")
    st.markdown(
        "Synthesizes external visual attributes (Notebooks 01–03) with internal CLIP representations "
        "and ViT-B/32 self-attention gradients (Notebook 04)."
    )

    figure_catalog = [
        (
            "fig4_text_prototype_similarity.png",
            "1. Text-Encoder Prototype Overlap (512-D Cosine Space)",
            "Shows baseline cosine similarity between genre text embeddings before any image is seen.",
        ),
        (
            "fig5_latent_concept_alignment.png",
            "2. Latent Concept Probing (Dark/Ominous, Portrait, Illustrated, Typography)",
            "Projects genre text vectors onto semantic axes in CLIP's shared embedding space to explain visual stereotype pull.",
        ),
        (
            "fig7_group_saliency_correct_vs_error.png",
            "3. Group-Level ViT-B/32 Title-Band Saliency (Correct vs. Incorrect, n=300)",
            "Tests whether misclassified covers rely differently on top/bottom typography bands vs. central artwork.",
        ),
        (
            "fig8_representation_space_knn.png",
            "4. 512-D Image Embedding Organization (2D SVD & 10-NN Purity)",
            "Visualizes how input genres cluster in CLIP's image space and how 10-NN purity shifts after text masking.",
        ),
        (
            "fig2_visual_stereotypes.png",
            "5. False-Positive Visual Stereotype Amplification (OpenCV Attributes)",
            "Quantifies Z-score shifts in brightness, saturation, warmth, and face area among misclassified covers.",
        ),
        (
            "fig3_calibration_and_confusion.png",
            "6. Margin Calibration & Before/After Text-Masking Confusion Matrices",
            "Compares empirical accuracy across cosine-margin quintiles and genre confusion shifts.",
        ),
    ]

    for fname, title, desc in figure_catalog:
        fpath = FIGURES_DIR / fname
        st.markdown(f"#### {title}")
        st.caption(desc)
        if fpath.exists():
            st.image(str(fpath))
        else:
            st.info(
                f"Figure not found yet: `{fpath.name}` (generate in Notebook 03 or 04)."
            )

# ==============================================================================
# TAB 4: LIVE PROMPT & MASKING LAB
# ==============================================================================
with tab4:
    st.subheader("Interactive Prompt-Engineering & OCR Perturbation Test")
    st.write(
        "Select any cover from the benchmark and edit the candidate text prompts below to observe "
        "how CLIP's cosine similarities shift between the Original and Text-Masked cover."
    )

    lab_idx = st.number_input(
        "Benchmark Cover Row Index", min_value=0, max_value=len(df_bench) - 1, value=0
    )
    lab_row = df_bench.iloc[int(lab_idx)]

    default_prompts = "\n".join(
        [f"A book cover of a {g} book." for g in SELECTED_6_GENRES]
    )
    custom_text = st.text_area(
        "Candidate Prompts (one per line)", value=default_prompts, height=160
    )
    prompts_list = [line.strip() for line in custom_text.splitlines() if line.strip()]

    if st.button("Run Live CLIP Comparison") and prompts_list:
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            st.image(
                lab_row["image_path"],
                caption=f"Original ({lab_row['genre']})",
                width=224,
            )
        with col_b:
            st.image(lab_row["masked_image_path"], caption="Text-Masked", width=224)
        with col_c:
            res_df = score_custom_prompts(
                lab_row["image_path"], lab_row["masked_image_path"], prompts_list
            )
            st.dataframe(res_df, hide_index=True)
