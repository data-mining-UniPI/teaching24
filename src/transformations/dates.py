import numpy
import pandas


def numpydatetime_to_season(d: numpy.datetime64) -> str:
    match pandas.to_datetime(d).month:
        case 1 | 2 | 11 | 12:
            return "winter"
        case 3 | 4:
            return "spring"
        case 5 | 6 | 7 | 8 | 9:
            return "summer"
        case 10:
            return "fall"

def numpydatetime_to_year(d: numpy.datetime64) -> int:
    return pandas.to_datetime(d).year
