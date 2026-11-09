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

| Student | Student ID |
|---|---|
| Darina Vorona | k12148163 |
| Vladyslav Kravchenko | k12148132 |
| Sofiia Guchenko | k12148479 |
| Milana Koval | k12148486 |

## Model & Dataset

| | Details |
|---|---|
| **Model** | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) (`openai/clip-vit-base-patch32`) || **Task** | Zero-shot image classification |
| **Dataset** | BookCover30 |
| **Subset** | 3,000 book covers across 6 genres |

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

## Repository Structure

```text
MarshmallowBunnies/
├── README.md
├── team-info.json
├── workload.json
├── requirements.txt
├── ai_reflection_midpoint.md
├── ai-interactions/
│   └── phase1-ai-interactions.md
├── data/
│   └── clip_predictions.csv
├── notebooks/
│   └── phase1-ai-aided-pair.ipynb
└── dashboard/
    └── dashboard.py
```

## Installation & Usage

**Requirements:** Python 3.10+, pip

Clone the repository and install dependencies:

```bash
git clone https://github.com/sofiaguchenko/MarshmallowBunnies.git
cd MarshmallowBunnies
python -m pip install -r requirements.txt
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
