from __future__ import annotations
from typing import cast
from loguru import logger
import pandas as pd
from pandas import Index, Series
import numpy as np

from stochastic_volatility_models.config import MODULE_DIRECTORY


DEFAULT_CHUNKSIZE = 20000


def get_realised_measures(
	ticker: str,
	dates: Index,
	measure: str = "rk_parzen",
) -> Series:
	path: str = f"{MODULE_DIRECTORY}/data/realised_volatility/oxfordman.csv"

	realised_measures = pd.read_csv(path, index_col=0)
	ticker_realised_volatility = realised_measures.loc[(realised_measures.Symbol == f".{ticker.upper()}"), measure]
	ticker_realised_volatility.index = Index(((ticker_realised_volatility.index).values.astype("datetime64[ns]") + np.timedelta64(2, "h")).astype("datetime64[D]").astype(str))

	try:
		realised_volatility = cast(Series, np.sqrt(ticker_realised_volatility.loc[dates]))
	except KeyError:
		logger.warning("Dates provided not in range of data")
		realised_volatility = pd.Series(None, index=dates)

	return realised_volatility


def get_realised_measure_maximum(
	ticker: str,
	measure: str = "rk_parzen",
) -> float:
	path: str = f"{MODULE_DIRECTORY}/data/realised_volatility/oxfordman.csv"

	realised_measures = pd.read_csv(path, index_col=0)
	ticker_realised_volatility = realised_measures.loc[(realised_measures.Symbol == f".{ticker.upper()}"), measure]
	ticker_realised_volatility.index = Index(((ticker_realised_volatility.index).values.astype("datetime64[ns]") + np.timedelta64(2, "h")).astype("datetime64[D]").astype(str))

	maximum_realised_volatility = cast(Series, np.sqrt(ticker_realised_volatility)).max()

	return maximum_realised_volatility
