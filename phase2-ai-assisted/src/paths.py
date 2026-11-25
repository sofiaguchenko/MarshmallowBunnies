from pathlib import Path
from typing import List

# phase2-ai-assisted/src/paths.py -> parents[1] is phase2-ai-assisted/, parents[2] is MarshmallowBunnies/
PHASE2_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Shared BookCover30 dataset in root archive/
ARCHIVE_DIR: Path = REPO_ROOT / "archive"
IMAGES_DIR: Path = ARCHIVE_DIR / "title30cat" / "224x224"
TRAIN_LISTING_CSV: Path = ARCHIVE_DIR / "book30-listing-train.csv"
TEST_LISTING_CSV: Path = ARCHIVE_DIR / "book30-listing-test.csv"
BOOK32_LISTING_CSV: Path = ARCHIVE_DIR / "book32-listing.csv"
TRAIN_LABELS_TXT: Path = ARCHIVE_DIR / "bookcover30-labels-train.txt"
TEST_LABELS_TXT: Path = ARCHIVE_DIR / "bookcover30-labels-test.txt"

# Canonical 6-genre subset from Phase 1
SELECTED_6_GENRES: List[str] = [
    "Romance",
    "Mystery, Thriller & Suspense",
    "Science Fiction & Fantasy",
    "History",
    "Biographies & Memoirs",
    "Children's Books",
]

# Phase 1 artifacts (for baseline comparisons)
PHASE1_DATA_DIR: Path = REPO_ROOT / "data"
PHASE1_OUTPUTS_DIR: Path = REPO_ROOT / "outputs"

# Phase 2 outputs & logs inside phase2-ai-assisted/
OUTPUTS_DIR: Path = PHASE2_ROOT / "outputs"
MASKED_IMAGES_DIR: Path = OUTPUTS_DIR / "masked_covers"
FEATURES_PARQUET: Path = OUTPUTS_DIR / "visual_features.parquet"
PREDICTIONS_PARQUET: Path = OUTPUTS_DIR / "phase2_predictions.parquet"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

AI_INTERACTIONS_LOG: Path = (
    PHASE2_ROOT / "ai-interactions" / "phase2-ai-interactions.md"
)


def ensure_dirs() -> None:
    """Create Phase 2 output directories if they do not already exist."""
    for directory in (
        OUTPUTS_DIR,
        MASKED_IMAGES_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
