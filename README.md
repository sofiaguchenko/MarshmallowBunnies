# 📚 Judging a Book by Its Cover

### Genre Bias & Visual Stereotypes in AI

**Team:** Marshmallow Bunnies  
**Course:** Visualization for AI  
**University:** JKU Linz

## Project Overview

Can AI recognize a book's genre from its cover, or does it rely on visual stereotypes?

We investigate how **CLIP** classifies book genres based on cover images. Our goal is to understand whether the model recognizes meaningful visual cues or relies on shortcuts such as colors, typography, and common design patterns.

Rather than focusing only on classification accuracy, we use visualizations to explore systematic errors, biases, and prediction behavior.

## Team

| Student              | Student ID |
| -------------------- | ---------- |
| Daryna Vorona        | k12148163  |
| Vladyslav Kravchenko | k12148132  |
| Sofiia Guchenko      | k12148479  |
| Milana Koval         | k12148486  |

## Model & Dataset

|             | Details                                                                                               |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| **Model**   | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) (`openai/clip-vit-base-patch32`) |
| **Task**    | Zero-shot image classification                                                                        |
| **Dataset** | BookCover30                                                                                           |
| **Subset**  | 3,000 book covers across 6 genres                                                                     |

CLIP compares image representations with text descriptions of genres to predict the most likely category without additional model training.

## Research Questions

- Which genres does CLIP confuse most frequently?
- Do visual characteristics such as brightness, colors, or faces influence predictions?
- How does changing the text prompt affect classification?
- How confident is CLIP when making incorrect predictions?

## Methodology

We treat CLIP as a black-box model and investigate its predictions through:

- **Genre Analysis:** Comparing classification accuracy and identifying common genre confusions.
- **Visual Features:** Exploring relationships between cover characteristics and predictions.
- **Prompt Sensitivity:** Comparing neutral and descriptive text prompts.
- **Confidence Analysis:** Investigating prediction confidence and high-confidence errors.
- **Interactive Exploration:** Using a Streamlit dashboard to examine individual predictions and patterns.

Our goal is to understand model behavior beyond aggregate performance metrics.

## Key Findings from AI-aided pair

Using neutral prompts, CLIP achieved **62.3% accuracy**. Changing to descriptive prompts reduced accuracy to **52.3%** and changed approximately **40% of predictions**.

We also observed associations between visual features and classification errors, as well as incorrect predictions with high confidence.

For the complete analysis, figures, and conclusions, see the research report.

## Key findings from human-only pair

The human-only notebook evaluates CLIP locally on the BookCover30 subset.

The main results from the local run were:

- neutral prompt accuracy: 0.3247 (**32.47%**)
- descriptive prompt accuracy: 0.3342 (**33.42%**)
- prediction changes between prompt styles: 2613 examples

This means that prompt wording has a strong impact on model behavior: many examples changed prediction when the text prompt was reformulated. At the same time, the difference in overall accuracy between the two prompt styles is modest, which suggests that prompt engineering alone does not guarantee a large gain in performance.

The confusion patterns remain consistent with the broader project goal: visually similar genres are still difficult to distinguish and prompt wording can shift the model toward different genre interpretations.

## Analysis notebooks and reports

- [notebooks/phase1-ai-aided-pair.ipynb](notebooks/phase1-ai-aided-pair.ipynb) — earlier AI-assisted analysis
- [notebooks/phase1-human-only.ipynb](notebooks/phase1-human-only.ipynb) — local CLIP evaluation and prompt comparison
- [reports/Report_phase1-ai-aided-pair.md](reports/Report_phase1-ai-aided-pair.md) — original report for the AI-assisted run
- [reports/phase1-ai-vs-human-comparison.md](reports/phase1-ai-vs-human-comparison.md) — comparison between the two analyses

## Repository Structure

```text
MarshmallowBunnies/
├── ai-interactions/
│   └── phase1-ai-interactions.md
├── dashboard/
│   └── dashboard.py
├── data/
│   └── clip_predictions.csv
├── notebooks/
│   ├── phase1-ai-aided-pair.ipynb
│   └── phase1-human-only.ipynb
├── outputs/
├── reports/
│   ├── Report_phase1-ai-aided-pair.md
│   ├── phase1-ai-vs-human-comparison.md
│   └── output_*.png
├── .gitignore
├── README.md
├── ai_reflection_midpoint.md
├── requirements.txt
├── team-info.json
└── workload.json
```

## Quick setup

```bash
git clone https://github.com/sofiaguchenko/MarshmallowBunnies.git
cd MarshmallowBunnies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run the interactive dashboard:**

```bash
python -m streamlit run dashboard/dashboard.py
```

**Open the Jupyter Notebook:**

```bash
jupyter notebook notebooks/phase1-ai-aided-pair.ipynb
```

The dashboard uses saved predictions from `data/clip_predictions.csv`, so running CLIP again is not required.

**Dataset:** The original BookCover30 images are not included in the repository. To display individual book covers or reproduce the analysis, place the dataset in `archive/title30cat/224x224/` at the repository root, preserving the original filenames and directory structure.
