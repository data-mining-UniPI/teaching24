from typing import Tuple, Optional, Sequence

import pandas


def average_position(cyclist: str, races: Optional[Sequence[str]], races_df: pandas.DataFrame) -> Tuple[float, float]:
    if races is None:
        cyclist_placements = races_df[races_df["cyclist"] == cyclist, "position"]
    else:
        cyclist_placements = races_df[
            (races_df["cyclist"] == cyclist) &
            (races_df["_url"].isin(races)),
            "position"
        ]

    return cyclist_placements.mean(), cyclist_placements.std()
