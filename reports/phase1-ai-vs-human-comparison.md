# Phase 1 comparison: AI-aided pair vs. human-only pair

## Overview

This report compares the earlier AI-assisted analysis with the later human-only notebook that evaluates CLIP locally on the BookCover30 subset. The goal is to highlight where the two approaches agree, where they diverge, and what we can conclude from both together.

## 1. Research setup

Both analyses use the same general setting:

- dataset: BookCover30
- model: CLIP ViT-B/32
- task: zero-shot genre classification
- label set: six genres
- focus: genre confusion, prompt sensitivity, and model behavior

The main difference is the workflow:

- the AI-aided pair notebook focuses on a broader exploratory analysis and report structure
- the human-only notebook focuses on a cleaner local reproduction with direct prompt comparison and saved outputs

## 2. Summary of the AI-assisted findings

The earlier report concluded that CLIP shows clear genre-related visual patterns, but also systematic confusion between categories.

Main findings from that analysis:

- overall neutral accuracy was around 62.33%
- descriptive prompts reduced accuracy to 52.30%
- many predictions changed when the prompt wording changed
- genre confusion was strongest between visually similar categories
- confidence was only partly informative, since some high-confidence predictions were still wrong

This analysis showed that CLIP can recognize visible cover patterns, but it is also sensitive to prompt wording and design stereotypes.

## 3. Summary of the human-only findings

The later human-only notebook re-ran the evaluation locally and recorded prompt-style differences directly.

Observed results from the local run:

- neutral prompt accuracy: 32.47%
- descriptive prompt accuracy: 33.42%
- prompt changes between styles: 2,613 examples

These numbers are lower than the earlier report, but they reflect a different evaluation setup and a direct saved output from the local notebook. The human-only analysis is especially informative because it shows how strongly the predictions can vary from one prompt formulation to another.

## 4. Comparison of the two approaches

### Agreement

Both analyses agree on the following points:

- CLIP does not behave like a perfectly reliable genre detector
- prompt phrasing changes the model's decisions materially
- genre confusion is a central issue
- image appearance strongly influences the prediction outcome
- the model is useful as a baseline, but not as a final decision system

### Differences

The two analyses differ in emphasis and reported performance:

- the AI-assisted report shows higher accuracy values and a more polished narrative
- the human-only notebook shows a more direct prompt-comparison result and a larger number of changed predictions
- the human-only run is more conservative and closer to the raw evaluation pipeline

This is not necessarily a contradiction. The difference likely reflects the specific experimental setup, the exact prompt wording, and the decision to treat the notebook as a direct local reproduction rather than a curated exploratory narrative.

## 5. Interpretation

Taken together, the findings suggest that CLIP is strong enough to pick up strong visual markers in book covers, but weak enough that it still confuses many related genres.

The most important takeaway is that zero-shot classification is highly sensitive to labeling and prompting strategy. Even small wording changes can trigger large prediction shifts. That means performance should be interpreted as a behavior of the prompt-model pair rather than as an intrinsic property of the data alone.

## 6. Final conclusion

The AI-assisted and human-only analyses are complementary rather than competing.

The AI-aided report provides the broader conceptual interpretation and a polished discussion of genre bias and visual stereotypes. The human-only notebook provides a direct, reproducible local evaluation and a clearer sense of how prompt variation affects classification decisions.

Overall, both analyses support the same conclusion: CLIP is a useful baseline for book-cover genre classification, but it is still unreliable for consistent, high-stakes genre prediction without additional domain-specific adaptation or metadata.
