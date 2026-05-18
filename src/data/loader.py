"""Data loader for BRIGHTER cleaned CSVs."""

import pandas as pd
from pathlib import Path
from typing import Optional


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

LANGUAGES = ["pcm", "chn", "mar", "ary", "tat", "vmw", "ptmz", "zul", "ind"]


def load_split(
    base_path: str,
    language: str,
    split: str,
    emotions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load a single split (train/dev/test) for a language.

    Expects CSV files at: {base_path}/{language}/{split}.csv
    Each CSV has at minimum: text, anger, disgust, fear, joy, sadness, surprise

    Returns DataFrame with columns: text, anger, disgust, fear, joy, sadness, surprise
    """
    if emotions is None:
        emotions = EMOTIONS

    path = Path(base_path) / language / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Validate required columns exist
    required = ["text"] + emotions
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Keep 'id' column if present, plus required columns
    cols_to_keep = required[:]
    if "id" in df.columns:
        cols_to_keep = ["id"] + cols_to_keep
    df = df[cols_to_keep].copy()
    for emo in emotions:
        df[emo] = df[emo].astype(int)

    return df


def load_language(
    base_path: str,
    language: str,
    splits: Optional[list[str]] = None,
    emotions: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load all requested splits for a single language.

    Returns dict mapping split name to DataFrame.
    """
    if splits is None:
        splits = ["train", "dev", "test"]

    return {
        split: load_split(base_path, language, split, emotions)
        for split in splits
    }


def load_all_languages(
    base_path: str,
    languages: Optional[list[str]] = None,
    splits: Optional[list[str]] = None,
    emotions: Optional[list[str]] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load data for all languages.

    Returns nested dict: {language: {split: DataFrame}}
    """
    if languages is None:
        languages = LANGUAGES

    return {
        lang: load_language(base_path, lang, splits, emotions)
        for lang in languages
    }


def get_texts_and_labels(
    df: pd.DataFrame,
    emotions: Optional[list[str]] = None,
    include_ids: bool = False,
) -> tuple:
    """Extract texts and label dicts from a df"""
    if emotions is None:
        emotions = EMOTIONS

    texts = df["text"].tolist()
    labels = df[emotions].to_dict(orient="records")

    if not include_ids:
        return texts, labels

    if "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = [str(i) for i in df.index]

    return texts, labels, ids
