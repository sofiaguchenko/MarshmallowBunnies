
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CLIP Book Genre Explorer",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("clip_predictions.csv")

df = load_data()

st.title("📚 CLIP Book Genre Explorer")
st.write(
    "Phase I: Investigating genre bias and visual "
    "stereotypes in book cover classification."
)

# Overall statistics
st.header("1. Model Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Book Covers", len(df))
col2.metric("Neutral Accuracy", f"{df['correct'].mean():.1%}")
col3.metric(
    "Descriptive Accuracy",
    f"{df['descriptive_correct'].mean():.1%}"
)

# Interactive genre selection
st.header("2. Genre Analysis")

genres = sorted(df["Category"].unique())

selected_genre = st.selectbox(
    "Select a book genre:",
    genres
)

genre_df = df[df["Category"] == selected_genre]

neutral_acc = genre_df["correct"].mean()
descriptive_acc = genre_df["descriptive_correct"].mean()

st.write(f"### {selected_genre}")
st.write(f"Number of books: {len(genre_df)}")

st.bar_chart(
    pd.DataFrame({
        "Accuracy": [
            neutral_acc * 100,
            descriptive_acc * 100
        ]
    }, index=["Neutral", "Descriptive"])
)

st.caption(
    "Comparison of CLIP classification accuracy "
    "using neutral and descriptive prompts."
)


# Interactive Error Explorer
st.header("3. Error Explorer")

prompt_type = st.radio(
    "Select prompt type:",
    ["Neutral", "Descriptive"],
    horizontal=True
)

prediction_col = (
    "predicted_genre"
    if prompt_type == "Neutral"
    else "descriptive_prediction"
)

# Show only incorrect predictions for selected genre
errors_df = genre_df[
    genre_df[prediction_col] != genre_df["Category"]
]

st.write(f"Misclassified books: {len(errors_df)}")

# Filter by predicted genre
predicted_options = sorted(errors_df[prediction_col].dropna().unique())

selected_prediction = st.selectbox(
    "Predicted as:",
    ["All"] + predicted_options
)

if selected_prediction != "All":
    errors_df = errors_df[
        errors_df[prediction_col] == selected_prediction
    ]

# Display book covers
st.subheader("Misclassified Book Covers")

num_examples = st.slider(
    "Number of examples:",
    min_value=3,
    max_value=18,
    value=6,
    step=3
)

examples = errors_df.head(num_examples)

for start in range(0, len(examples), 3):
    cols = st.columns(3)

    for col, (_, book) in zip(
        cols, examples.iloc[start:start + 3].iterrows()
    ):
        with col:
            st.image(
                book["image_path"],
                use_container_width=True
            )
            st.caption(f"Title: {book['Title']}")
            st.caption(f"True: {book['Category']}")
            st.caption(f"Predicted: {book[prediction_col]}")


# Prompt Comparison Explorer
st.header("4. Prompt Comparison Explorer")

comparison_df = genre_df[
    genre_df["predicted_genre"] !=
    genre_df["descriptive_prediction"]
].copy()

st.write(
    f"Books with changed predictions: {len(comparison_df)}"
)

comparison_df["outcome"] = "Changed but still wrong"

comparison_df.loc[
    (comparison_df["correct"] == True) &
    (comparison_df["descriptive_correct"] == False),
    "outcome"
] = "New error from descriptive"

comparison_df.loc[
    (comparison_df["correct"] == False) &
    (comparison_df["descriptive_correct"] == True),
    "outcome"
] = "Corrected by descriptive"

outcome = st.selectbox(
    "Select prediction outcome:",
    [
        "All changed predictions",
        "Corrected by descriptive",
        "New error from descriptive",
        "Changed but still wrong"
    ]
)

if outcome != "All changed predictions":
    comparison_df = comparison_df[
        comparison_df["outcome"] == outcome
    ]

st.write(f"Matching books: {len(comparison_df)}")

num_comparisons = st.slider(
    "Number of comparisons:",
    min_value=1,
    max_value=12,
    value=6
)

for _, book in comparison_df.head(num_comparisons).iterrows():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(book["image_path"], width=180)

    with col2:
        st.write(f"**{book['Title']}**")
        st.write(f"True genre: {book['Category']}")
        st.write(f"Neutral: {book['predicted_genre']}")
        st.write(
            f"Descriptive: {book['descriptive_prediction']}"
        )
        st.caption(book["outcome"])

    st.divider()

# 5. Confidence Analysis

st.header("5. Confidence Analysis")

st.write(
    "Explore how confident CLIP is in its predictions "
    "and investigate high-confidence mistakes."
)

confidence_df = df.copy()
confidence_df["confidence"] = pd.to_numeric(
    confidence_df["confidence"], errors="coerce"
)
confidence_df = confidence_df.dropna(subset=["confidence"])

# Compare average confidence
st.subheader("Correct vs Incorrect Predictions")

confidence_summary = (
    confidence_df.groupby("correct")["confidence"]
    .mean()
)

st.bar_chart(
    pd.DataFrame({
        "Prediction": ["Incorrect", "Correct"],
        "Average confidence": [
            confidence_summary.get(False, 0),
            confidence_summary.get(True, 0)
        ]
    }).set_index("Prediction")
)

# High-confidence mistakes
st.subheader("High-Confidence Mistakes")

threshold = st.slider(
    "Minimum confidence:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

high_conf_errors = confidence_df[
    (confidence_df["correct"] == False) &
    (confidence_df["confidence"] >= threshold)
].sort_values("confidence", ascending=False)

st.write(
    f"High-confidence mistakes: {len(high_conf_errors)}"
)

for _, book in high_conf_errors.head(10).iterrows():
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(book["image_path"], width=160)

    with col2:
        st.write(f"**{book['Title']}**")
        st.write(f"True genre: {book['Category']}")
        st.write(f"Predicted: {book['predicted_genre']}")
        st.write(f"Confidence: {book['confidence']:.1%}")

    st.divider()
