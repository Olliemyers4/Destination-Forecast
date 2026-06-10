from random import shuffle

import pandas as pd
import pycountry
import pycountry_convert as pc

ASSUMED_DENSITY = 0.9
CONTINENT_LOOKUP = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
}

MISSING_LOCATION_LOOKUP = {
    "Iran": [27.157481, 56.147944],  # Bandar Shahid Rajaee
    "Philippines": [14.62059, 120.94526],  # Manila Port
}


def country_name_to_continent(name):
    try:
        country = pycountry.countries.lookup(name)
        country_code = country.alpha_2

        continent_code = pc.country_alpha2_to_continent_code(country_code)

        return CONTINENT_LOOKUP[continent_code]
    except:  # noqa E722
        # Allowing bare except here as we want to catch anything this could throw
        return None


# Read all the datasets:
port_calls = pd.read_csv("port_calls_port_types.csv")

for key in MISSING_LOCATION_LOOKUP.keys():
    missing_locations = port_calls[
        (port_calls["destination"] == key)
        & (port_calls["destination_latitude"].isnull())
    ]
    for i, _ in missing_locations.iterrows():
        port_calls.loc[i, "destination_latitude"] = MISSING_LOCATION_LOOKUP[key][0]
        port_calls.loc[i, "destination_longitude"] = MISSING_LOCATION_LOOKUP[key][1]


vessels = pd.read_csv("vessels.csv")
trades = pd.read_csv("cleaned_trades.csv")

# Input Features
"""
Cargo capacity
    assuming a density of around 0.9t/m^3 take the port volume
    and calculate a mass and compare to vessel dead weight
Lat of current location
Lon of current location
vessel type
history of port in/outbound
flag continent
"""
# Output Features
"""
Lat of next port
Lon of next port
cargo capacity (to allow for consistent predicitions)
"""

calls_and_vessels = port_calls.merge(
    vessels, right_on="id", left_on="vessel_id", how="left"
)

calls_and_vessels["capacity"] = (
    calls_and_vessels["cargo_volume"] * ASSUMED_DENSITY
) / calls_and_vessels["dead_weight"]

calls_and_vessels["current_port_type"] = (
    1 + calls_and_vessels["is_trade_origin"] + (2 * calls_and_vessels["is_trade_dest"])
)

calls_and_vessels["previous_port_type"] = -1
calls_and_vessels["next_port_lat"] = -999.0
calls_and_vessels["next_port_lon"] = -999.0
calls_and_vessels["next_port_capacity"] = -1.0


for vessel in calls_and_vessels["vessel_id"].unique():
    count = 0
    previous_row = 0
    for i, row in calls_and_vessels[
        calls_and_vessels["vessel_id"] == vessel
    ].iterrows():
        if count == 0:
            previous_row = row["current_port_type"]
        else:
            calls_and_vessels.loc[i, "previous_port_type"] = previous_row
            previous_row = row["current_port_type"]
        count += 1

# Now we need to do it backwards to handle the 'next' fields
for vessel in calls_and_vessels["vessel_id"].unique():
    count = 0
    next_lat = 0
    next_long = 0
    next_capacity = 0
    for i, row in calls_and_vessels[calls_and_vessels["vessel_id"] == vessel][
        ::-1
    ].iterrows():
        if count == 0:
            next_lat = row["destination_latitude"]
            next_long = row["destination_longitude"]
            next_capacity = row["capacity"]
        else:
            calls_and_vessels.loc[i, "next_port_lat"] = next_lat
            calls_and_vessels.loc[i, "next_port_lon"] = next_long
            calls_and_vessels.loc[i, "next_port_capacity"] = next_long

            next_lat = row["destination_latitude"]
            next_long = row["destination_longitude"]
            next_capacity = row["capacity"]
        count += 1


calls_and_vessels["flag_continent"] = calls_and_vessels["flag_name"].apply(
    country_name_to_continent
)


# Now we drop rows
calls_and_vessels = calls_and_vessels[
    (calls_and_vessels["previous_port_type"] != -1)
    & (calls_and_vessels["next_port_capacity"] != -1)
]

# Split into train test and validation splits
# 70 15 15 split
unique_vessels = calls_and_vessels["vessel_id"].unique()
shuffle(unique_vessels)

full_len = len(unique_vessels)
split_at = int(full_len * 0.7)
split_at_2 = full_len + int((full_len - split_at) / 2)

train = unique_vessels[0:split_at]
test = unique_vessels[split_at:split_at_2]
validation = unique_vessels[split_at_2:]

calls_and_vessels = calls_and_vessels[
    [
        "vessel_id",
        "capacity",
        "destination_latitude",
        "destination_longitude",
        "vessel_type",
        "previous_port_type",
        "current_port_type",
        "flag_continent",
        "next_port_lat",
        "next_port_lon",
        "next_port_capacity",
    ]
]

calls_and_vessels[calls_and_vessels["vessel_id"].isin(train)].to_csv("train.csv")

calls_and_vessels[calls_and_vessels["vessel_id"].isin(test)].to_csv("test.csv")

calls_and_vessels[calls_and_vessels["vessel_id"].isin(validation)].to_csv(
    "validation.csv"
)
