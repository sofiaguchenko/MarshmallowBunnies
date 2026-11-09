# AI Interaction Log — CLIP Book Cover Classification

**Project:** Judging a Book by Its Cover — Genre Bias & Visual Stereotypes in AI  
**Team:** Marshmallow Bunnies  
**Course:** Visualization for AI  
**AI Tool:** ChatGPT (OpenAI)  
**Scope:** AI-aided pair, Phase I

## 1. Overview

We used ChatGPT as an assistant during our investigation of CLIP's book genre classification behavior.

AI support was mainly used for planning experiments, implementing Python code, debugging, designing visualizations, and interpreting results.

We did not treat ChatGPT's suggestions as automatically correct. We executed the code ourselves, inspected the outputs, and decided which analyses were useful for our research questions.

This document summarizes important interactions and how they influenced our work. Prompt descriptions are summaries of the original requests rather than verbatim transcripts.

## 2. Research Planning and Dataset Preparation

**Prompt summary:** Help us design an interesting visual investigation using CLIP and a dataset of book covers.

**AI contribution:**

ChatGPT helped structure the project around the question of whether CLIP recognizes book genres through meaningful visual information or relies on common design stereotypes.

It suggested investigating classification errors, visual characteristics, prompt sensitivity, and confidence instead of reporting only overall accuracy.

**Our decisions:**

We selected the pretrained [CLIP ViT-B/32 model](https://huggingface.co/openai/clip-vit-base-patch32) (`openai/clip-vit-base-patch32`) from Hugging Face and a balanced subset of BookCover30 containing 3,000 covers from six genres.

We decided to investigate the model as a black box, using its inputs and outputs.

**Outcome:** A set of research questions and a reproducible experimental workflow in Jupyter Notebook.

## 3. CLIP Classification and Genre Analysis

**Prompt summary:** Help us implement zero-shot genre classification with CLIP and visualize the results.

**AI contribution:**

ChatGPT assisted with Python code for model inference, prediction processing, accuracy calculations, and visualizations of classification performance.

The analysis included genre-level accuracy and confusion patterns.

**Our decisions:**

We used the generated code as a starting point, ran the experiments locally, and examined the actual predictions before drawing conclusions.

**Outcome:**

With neutral genre prompts, CLIP achieved approximately **62.33% accuracy** on our selected dataset.

The results showed that some genres were easier to recognize than others, motivating further investigation of systematic errors.

## 4. Investigating Visual Characteristics

**Prompt summary:** Can we investigate whether visual characteristics of book covers influence CLIP's genre predictions?

**AI contribution:**

ChatGPT helped develop analyses of cover properties, including brightness and the presence of faces.

It also helped create visualizations comparing prediction patterns across groups of images.

**Our decisions:**

We focused on relationships that could be measured using our dataset and inspected the results rather than assuming that a visual feature caused a prediction.

**Outcome:**

We observed several interesting associations:

- Among History covers, those with detected faces were more frequently predicted as Biography (approximately 54.5%) than those without detected faces (approximately 23.3%).
- Darker Romance covers were more frequently predicted as Thriller (approximately 23.2% in the darkest group compared with 4.0% in the brightest group).

These observations suggest possible visual shortcuts, but they do not establish causation.

## 5. Prompt Sensitivity Experiment

**Prompt summary:** How can we test whether changing genre descriptions affects CLIP's classification results?

**AI contribution:**

ChatGPT helped implement a comparison between neutral genre prompts and more descriptive prompts.

It supported the calculation of accuracy differences, changed predictions, and visual comparisons between the two prompt strategies.

**Our decisions:**

We compared predictions for the same 3,000 images using both prompt sets, keeping the image dataset unchanged.

**Outcome:**

| Metric | Result |
|---|---|
| Neutral prompt accuracy | 62.33% |
| Descriptive prompt accuracy | 52.30% |
| Predictions changed | 1,199 / 3,000 |
| Percentage changed | 39.97% |

Descriptive prompts did not improve overall accuracy. They also increased the number of predictions assigned to History, from 486 to 1,168, including many incorrect predictions.

This became an important finding: CLIP's output depends strongly on how the candidate genres are described.

## 6. Confidence Analysis

**Prompt summary:** Help us investigate how confident CLIP is in its predictions, especially when it makes mistakes.

**AI contribution:**

ChatGPT helped structure an analysis of prediction confidence and suggested visualizing high-confidence classification errors.

**Our decisions:**

We investigated confidence alongside correctness rather than treating a high score as evidence that a prediction was correct.

We also kept confidence analysis connected to the earlier genre and prompt experiments.

**Outcome:**

We added confidence-focused visualizations and explored incorrect predictions with high model scores.

An important interpretation is that CLIP's relative similarity-based scores should not automatically be treated as calibrated probabilities of correctness.

## 7. Interactive Streamlit Dashboard

**Prompt summary:** Help us create an interactive dashboard to explore our saved CLIP predictions and investigate individual errors.

**AI contribution:**

ChatGPT helped develop a Streamlit dashboard with five main sections:

1. Model Overview
2. Genre Analysis
3. Error Explorer
4. Prompt Comparison Explorer
5. Confidence Analysis

The dashboard uses the saved results in `clip_predictions.csv`, allowing us to explore predictions without rerunning CLIP.

**Our decisions and modifications:**

We kept the dashboard as a separate Python file rather than embedding its implementation inside the notebook.

We also reorganized the repository into separate `dashboard/`, `data/`, and `notebooks/` directories.

After this change, the original relative CSV path needed to be updated. We modified the data-loading code to resolve the path relative to the project directory:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "clip_predictions.csv")
```

**Outcome:**

The dashboard provides interactive exploration of genre predictions, errors, prompt changes, and confidence.

During development, we also examined individual covers whose predicted genres changed between prompt strategies. These examples helped connect aggregate statistics with actual model behavior.

## 8. Debugging and Limitations of AI Assistance

ChatGPT was useful for quickly proposing code and explaining possible approaches, but its outputs still required verification.

Important limitations we encountered included:

- **Code integration:** Suggested code had to be adapted to the existing notebook and data structure.
- **File paths:** Moving the dashboard and CSV into separate directories required correcting the data-loading path.
- **Interpretation:** Statistical associations between visual features and predictions could not automatically be interpreted as causal relationships.
- **Prompt sensitivity:** More detailed text prompts did not necessarily produce better classification results.
- **Reproducibility:** The dashboard depends on saved predictions, while displaying original book covers requires access to the image dataset.

These issues showed why generated code and explanations should be tested against real outputs.

## 9. Human Decisions and AI Influence

| Activity | AI contribution | Human decision |
|---|---|---|
| Research design | Suggested investigation directions | Selected research questions and scope |
| Classification | Assisted with implementation | Executed and validated experiments |
| Visual feature analysis | Suggested metrics and plots | Selected and interpreted relevant patterns |
| Prompt comparison | Helped implement comparison | Evaluated accuracy and prediction changes |
| Confidence analysis | Suggested analysis approach | Investigated errors and score interpretation |
| Dashboard | Assisted with Streamlit implementation | Organized files and adapted code |
| Debugging | Suggested fixes | Tested and applied changes |

## 10. Main Takeaways

Our experience showed that ChatGPT can support visual AI research by speeding up coding, debugging, and exploration.

However, its main value was not simply generating code. It helped us move from initial observations to more focused questions, such as why certain visual characteristics correlate with genre predictions and why descriptive prompts change the results.

We remained responsible for executing experiments, checking the outputs, selecting meaningful findings, and interpreting limitations.

We used ChatGPT as a research assistant, not as a replacement for our own analysis and decisions.
