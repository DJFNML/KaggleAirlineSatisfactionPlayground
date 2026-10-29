from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "kagglehub" / "competitions" / "playground-series-s6e10"
ORIG_DIR = Path(__file__).parent / "kagglehub" / "datasets" / "airline-passenger-satisfaction"

TARGET = "satisfaction"
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DEP_DELAY = "Departure Delay in Minutes"
ARR_DELAY = "Arrival Delay in Minutes"
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]


def _clean(df, zero_ratings_as_nan, delay_thresholds):
    """Split off the target (if present), drop id, handle 0 ratings, add delay flags."""
    y = df[TARGET].astype(int) if TARGET in df else None  # True/False -> 1/0
    X = df.drop(columns=["id", TARGET], errors="ignore")

    # 0 = "not applicable" in the source survey, so mark it as missing
    if zero_ratings_as_nan:
        X[RATING_COLS] = X[RATING_COLS].replace(0, np.nan)

    # 1 if either the departure or arrival delay is over t minutes
    for t in delay_thresholds:
        X[f"delay_over_{t}"] = ((X[DEP_DELAY] > t) | (X[ARR_DELAY] > t)).astype(int)

    return X, y


def load_original(zero_ratings_as_nan=True):
    """Original (real) Airline Passenger Satisfaction data, train + test, cleaned like the competition data.

    Columns match the competition train set; "Inflight service" (not in the competition) is dropped.
    Categoricals are plain strings here; callers align categories.
    """
    orig = pd.concat([pd.read_csv(ORIG_DIR / "train.csv"), pd.read_csv(ORIG_DIR / "test.csv")],
                     ignore_index=True)
    orig[TARGET] = orig[TARGET] == "satisfied"
    columns = pd.read_csv(DATA / "train.csv", nrows=0).columns
    return _clean(orig[columns], zero_ratings_as_nan, delay_thresholds=())


def load_data(zero_ratings_as_nan=True, delay_thresholds=(), ratings="numeric", teacher=False):
    """Return X, y, X_test, test_ids ready for XGBoost (enable_categorical=True).

    ratings: "numeric" keeps ratings as numbers, "category" turns them into categories,
    "both" keeps the numbers and adds a categorical copy of each rating.
    teacher: add "teacher", the predicted probability from a model trained only on the
    original dataset (see teacher.py), as a feature.
    """
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    X, y = _clean(train, zero_ratings_as_nan, delay_thresholds)
    X_test, _ = _clean(test, zero_ratings_as_nan, delay_thresholds)

    # XGBoost needs integer/string categories; Int64 keeps NaN as missing
    cat_cols = list(CAT_COLS)
    if ratings == "category":
        for df in (X, X_test):
            df[RATING_COLS] = df[RATING_COLS].astype("Int64")
        cat_cols += RATING_COLS
    elif ratings == "both":
        for df in (X, X_test):
            for col in RATING_COLS:
                df[f"{col} (cat)"] = df[col].astype("Int64")
        cat_cols += [f"{col} (cat)" for col in RATING_COLS]
    elif ratings != "numeric":
        raise ValueError(f"Unknown ratings mode: {ratings}")

    # Native categorical support; share categories so train/test codes match
    for col in cat_cols:
        categories = pd.concat([X[col], X_test[col]]).astype("category").cat.categories
        X[col] = pd.Categorical(X[col], categories=categories)
        X_test[col] = pd.Categorical(X_test[col], categories=categories)

    if teacher:
        from teacher import teacher_predictions  # noqa: PLC0415 (teacher.py imports this module)
        X["teacher"], X_test["teacher"] = teacher_predictions(X, X_test, zero_ratings_as_nan)

    return X, y, X_test, test["id"]
