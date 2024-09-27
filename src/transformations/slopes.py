"""Compute slopes"""
import numpy
from sklearn.linear_model import LinearRegression


def bounds_slope(values) -> float:
    """Only consider first and last point"""
    return (values[-1] - values[0]) / len(values)

def regression_slope(values) -> float:
    """Compute a linear regression over `values`, return coefficient"""
    regression = LinearRegression().fit(
        numpy.arange(len(values)).reshape(-1, 1),  # assume equi-spaced values
        values
    )
    
    return regression.coef_
