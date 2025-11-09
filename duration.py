#!/usr/bin/env python3
"""Compute durations of stays and durations outside from travel_history.tsv.

Outputs two tables to stdout:
 1) duration of stay for each Arrival -> next Departure (days)
 2) duration outside for each Departure -> next Arrival (days)

Assumptions:
 - The TSV has columns: Row, DATE, TYPE, LOCATION
 - TYPE is either 'Arrival' or 'Departure'
 - Dates are ISO format YYYY-MM-DD

The script handles unpaired final events by skipping incomplete pairs.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def parse_events(path: Path, sep: str = "\t") -> pd.DataFrame:
    """Read the TSV and return a pandas DataFrame with parsed dates and normalized columns.

    Columns returned: DATE (datetime64[ns]), TYPE (str), LOCATION (str)
    Sorted by DATE ascending.
    """
    df = pd.read_csv(path, sep=sep, dtype=str)
    # Keep only relevant columns and drop rows with missing values
    df = df[[c for c in ["DATE", "TYPE", "LOCATION"] if c in df.columns]].dropna()
    # parse dates
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()
    df["TYPE"] = df["TYPE"].str.strip()
    df["LOCATION"] = df["LOCATION"].str.strip()
    df = df.sort_values("DATE").reset_index(drop=True)

    # Validate alternation: once sorted, TYPE should alternate between Arrival and Departure
    # Normalize to lowercase for comparison
    types = df["TYPE"].str.lower().fillna("")
    if not types.isin(["arrival", "departure"]).all():
        bad = df.loc[~types.isin(["arrival", "departure"])][["DATE", "TYPE"]]
        raise ValueError(f"parse_events: found non Arrival/Departure TYPE rows:\n{bad}")

    is_arrival = types == "arrival"
    # adjacent values must differ (XOR)
    if len(is_arrival) >= 2:
        same_adjacent = is_arrival[:-1].values == is_arrival[1:].values
        if np.any(same_adjacent):
            # find first offending index (i where i and i+1 are same)
            idx = int(np.where(same_adjacent)[0][0])
            row1 = df.iloc[idx]
            row2 = df.iloc[idx + 1]
            raise ValueError(
                f"parse_events: TYPE must alternate Arrival/Departure after sorting by DATE. "
                f"Found same TYPE at rows {idx} and {idx+1}: {row1['DATE'].date()} {row1['TYPE']} "
                f"and {row2['DATE'].date()} {row2['TYPE']}"
            )

    return df


def compute_intervals(
    events: pd.DataFrame,
    from_type: str,
    to_type: str,
    from_label: str,
    to_label: str,
) -> pd.DataFrame:
    """Generalized interval computation.

    - from_type/to_type: 'arrival' or 'departure' (case-insensitive)
    - from_label/to_label: output column names for the from/to timestamps
    - location_label: output column name for location
    - location_source: 'left' to use the from-row's LOCATION, 'right' to use the to-row's LOCATION

    Returns a DataFrame with columns [from_label, to_label, location_label, 'Days']
    """
    df = events.copy()
    left = df[df["TYPE"].str.lower() == from_type.lower()].reset_index(drop=True)
    right = df[df["TYPE"].str.lower() == to_type.lower()].reset_index(drop=True)

    if left.empty or right.empty:
        return pd.DataFrame(columns=[from_label, to_label, "Locations", "Days"])

    left_sorted = left.sort_values("DATE").reset_index(drop=True)
    right_sorted = right.sort_values("DATE").reset_index(drop=True)

    # prepare right side with renamed columns for merge_asof
    right_prepped = right_sorted[["DATE", "LOCATION"]].rename(columns={"DATE": to_label, "LOCATION": "_right_LOCATION"})

    merged = pd.merge_asof(
        left_sorted,
        right_prepped,
        left_on="DATE",
        right_on=to_label,
        direction="forward",
    )

    merged = merged.dropna(subset=[to_label]).copy()

    # combine left/right locations into one column 'Locations' formatted 'LEFT -> RIGHT'
    left_loc = merged["LOCATION"].fillna("").astype(str)
    right_loc = merged["_right_LOCATION"].fillna("").astype(str)
    locations = left_loc + " -> " + right_loc

    result = pd.DataFrame(
        {
            from_label: merged["DATE"],
            to_label: merged[to_label],
            "Locations": locations,
            "Days": (merged[to_label] - merged["DATE"]).dt.days.astype(int),
        }
    )
    return result.reset_index(drop=True)


def compute_stays(events: pd.DataFrame) -> pd.DataFrame:
    return compute_intervals(
        events,
        from_type="arrival",
        to_type="departure",
        from_label="Arrival",
        to_label="Departure",
    )


def compute_outsides(events: pd.DataFrame) -> pd.DataFrame:
    return compute_intervals(
        events,
        from_type="departure",
        to_type="arrival",
        from_label="Departure",
        to_label="NextArrival",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute travel stay and outside durations from a travel history file")
    parser.add_argument("-i", "--input",
                        help="Path to input CSV/TSV file",
                        default=None)
    parser.add_argument("-f", "--format",
                        help="Input format: csv, tsv, or auto (default auto)",
                        choices=["csv", "tsv", "auto"], default="auto")
    args = parser.parse_args()
    p = Path(args.input)

    if not p.exists():
        raise SystemExit(f"Input file not found: {p}")

    # choose separator based on format
    if args.format == "csv" or (args.format == "auto" and p.suffix == ".csv"):
        sep = ","
    else:
        sep = "\t"

    events = parse_events(p, sep=sep)
    stays = compute_stays(events)
    outs = compute_outsides(events)
    print("Durations of stay (Arrival -> Departure):")
    print(stays)
    print("Durations outside (Departure -> next Arrival):")
    print(outs)


if __name__ == "__main__":
    main()
