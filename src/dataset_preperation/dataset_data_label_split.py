import numpy as np
import pandas as pd

INPUT_FILE = "train.csv"
HISTORY_SIZE = 3
LABEL_COLS = ["next_port_lat", "next_port_lon", "next_port_capacity"]

FEATURE_COLS = [
    "capacity",
    "destination_latitude",
    "destination_longitude",
    "vessel_type_enc",
    "previous_port_type",
    "current_port_type",
    "flag_continent_enc",
]
VESSEL_TYPES = [
    "Crude/Oil Products Tanker",
    "Products Tanker ",
    "Crude Oil Tanker",
    "Chemical/Oil Products Tanker",
    "Asphalt/Bitumen Tanker",
    "Bulk/Caustic Soda Carrier (CABU)",
    "Bulk/Oil Carrier (OBO)",
    "Chemical Tanker",
    "Bulk Carrier",
    "Ore/Oil Carrier",
    "LNG Tanker",
    "Offshore Tug/Supply Ship",
    "FSO (Floating, Storage, Offloading)",
    "FSO, Oil",
]

CONTINETS = [
    "Africa",
    "Asia",
    "North America",
    "South America",
    "Europe",
    "Ocenia",
    "Unknown",
]


def data_split(
    file: str,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(file)
    df = df.sort_values(["vessel_id", "Unnamed: 0"]).reset_index(drop=True)

    # Strip whitespace from categoricals and fill NaN flag_continent
    df["vessel_type"] = df["vessel_type"].str.strip()
    df["flag_continent"] = df["flag_continent"].fillna("Unknown")

    df["vessel_type_enc"] = pd.Categorical(
        df["vessel_type"], categories=VESSEL_TYPES
    ).codes / len(VESSEL_TYPES)

    df["flag_continet_enc"] = pd.Categorical(
        df["flag_continent"], categories=CONTINETS
    ).codes / len(CONTINETS)

    # Not enough time to investigate this
    df["capacity"] = df["capacity"].clip(upper=1)
    df["next_port_capacity"] = df["next_port_capacity"].clip(upper=1)

    df["desination_latitude"] = (
        df["desination_latitude"] + 90
    ) / 180  # [-90, 90] -> [0, 1]
    df["desination_longitude"] = (
        df["desination_longitude"] + 180
    ) / 360  # [-180, 180] -> [0, 1]

    df["next_port_lat"] = (df["next_port_lat"] + 90) / 180  # [-90, 90] -> [0, 1]
    df["next_port_lon"] = (df["next_port_lon"] + 180) / 360  # [-180, 180] -> [0, 1]

    df["previous_port_type"] = df["previous_port_type"] / 4  # 4 values
    df["current_port_type"] = df["current_port_type"] / 4  # 4 values

    X_list, y_list = [], []

    for _, group in df.groupby("vessel_id", sort=False):
        rows = group[FEATURE_COLS].values  # (n_rows, n_features)
        labels = group[LABEL_COLS].values  # (n_rows, 3)

        if len(rows) < history_size:
            continue

        # Slide a window of history_size across this vessel's rows.
        # Window [i : i+history_size] uses the label
        # from the last row i.e. index i+history_size-1
        for i in range(len(rows) - history_size + 1):
            X_list.append(rows[i : i + history_size])
            y_list.append(labels[i + history_size - 1])

    data = np.array(X_list, dtype=np.float32)  # (n_samples, 3, n_features)
    label = np.array(y_list, dtype=np.float32)  # (n_samples, 3)

    return data, label
