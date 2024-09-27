from typing import Sequence, Dict, Optional, Any

import numpy
import pandas


def remove_from_column(dataframe: pandas.DataFrame,
                       column: Sequence[str],
                       size: Optional[float | int] = None,
                       removal_index: Optional[pandas.core.series.Series] = None,
                       strategy: Optional[str] = "random",
                       replacing_value: Optional[Any] = None) -> pandas.DataFrame:
    """Remove values from `column` the given `dataframe`.

    Args:
        dataframe: The dataframe whose values to remove.
        column: What columns should be affected?
        size: How many rows should be affected? Ignored if `index` is provided. Either a float (proportion),
              integer (value equi-allocated among different columns), or a dictionary column => size.
        removal_index: Index of the rows to affect, required for `strategy` `index`. Defaults to None.
        strategy: Row selection strategy. One of:
            - "random": Random rows. Sizes given by `size`
            - "index": Uses `removal_index`
        replacing_value: The value replacing missing values.

    Returns:
        A copy of `dataframe` with values removed from columns defined in `axes`.
    """
    processed_dataset = dataframe.copy()

    if size is None:
        removal_size = sum(removal_index)
    elif isinstance(size, int):
        removal_size = size
    else:
        removal_size = int(dataframe.shape[0] * size)

    match strategy:
        case "random":
            rows_to_remove = numpy.random.choice(dataframe.index, removal_size, replace=False)

        case "index":
            # already given by removal index
            rows_to_remove = removal_index.copy()

    processed_dataset.loc[rows_to_remove, column] = replacing_value

    return processed_dataset


def remove_from_columns(
    dataframe: pandas.DataFrame,
    axes: Sequence[str],
    size_per_axis: Optional[float | int | Dict[str, float | int]],
    replacing_value: Any,
    strategy: str,
    removal_index: Optional[pandas.core.series.Series | Dict[str, pandas.core.series.Series]] = None
    ) -> pandas.DataFrame:
    """Remove values from the given `dataframe`.

    Args:
        dataframe: The dataframe whose values to remove.
        axes: What columns should be affected?
        size_per_axis: How many rows should be affected? Ignored if `index` is provided. Either a float (proportion),
              integer (value repeated for all columns), or a dictionary column => size.
        replacing_value: The value replacing missing values.
        removal_index: What rows should be affected? Either an array index, or a dictionary column => index.
        strategy: Row selection strategy. One of:
            - "random": Random rows. Sizes given by `size`
            - "index": Rows selected on a per-axis basis, as given by `index`

    Returns:
        A copy of `dataframe` with values removed from columns defined in `axes`.
    """
    processed_dataset = dataframe.copy()

    if size_per_axis is None:
        removal_sizes = {axis: None for axis in axes}
    elif isinstance(size_per_axis, int):
        removal_sizes = {axis: size_per_axis for axis in axes}
    else:
        dataframe_size = dataframe.shape[0]
        removal_sizes = {axis: int(size_per_axis * dataframe_size) for axis in axes}

    if removal_index is None:
        removal_indices = {axis: None for axis in axes}
    elif isinstance(removal_index, pandas.core.series.Series):
        removal_indices = {axis: removal_index for axis in axes}
    else:
        removal_indices = removal_index

    for axis in axes:
        processed_dataset = remove_from_column(processed_dataset,
                                               column=axis,
                                               size=removal_sizes[axis],
                                               removal_index=removal_indices[axis],
                                               strategy=strategy,
                                               replacing_value=replacing_value)

    return processed_dataset
