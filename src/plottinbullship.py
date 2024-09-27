import pandas


def correlation_long_format(dataset: pandas.DataFrame,
                            correlation_type_column: str = "correlation_type") -> pandas.DataFrame:
    correlations = dataset[correlation_type_column].unique()

    long_formats = list()
    for c in correlations:
        long_format = dataset[dataset[correlation_type_column] == c]\
            .drop(correlation_type_column, axis="columns")\
            .rename_axis(None)\
            .stack()\
            .reset_index()\
            .copy()
        long_format[correlation_type_column] = c
        long_formats.append(long_format)
    long_formats = pandas.concat(long_formats, axis="rows")

    return long_formats
