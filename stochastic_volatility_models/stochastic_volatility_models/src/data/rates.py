import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from stochastic_volatility_models.config import MODULE_DIRECTORY
from stochastic_volatility_models.src.utils.options.expiry import deannualise
from stochastic_volatility_models.src.utils.cache import np_cache
from stochastic_volatility_models.src.data.prices import DEFAULT_CHUNKSIZE


@np_cache(arg_num=1, arg_name="time_to_expiry")
def get_risk_free_interest_rate(
	time: np.datetime64,
	time_to_expiry: NDArray[np.float64],
	chunksize: int = DEFAULT_CHUNKSIZE,
) -> NDArray[np.float64]:
	path: str = f"{MODULE_DIRECTORY}/data/rates.csv"
	rates_iter = pd.read_csv(
		path,
		index_col=[0, 1],
		chunksize=chunksize,
	)

	key = np.datetime_as_string(time, unit="D")
	date_rates = pd.concat(
		[
			rates.xs(
				key,
				level=0,
			)
			for rates in rates_iter
			if key in rates.index.get_level_values(level=0)
		]
	).sort_index()

	if date_rates.index.empty:
		raise ValueError("Date not found")

	interpolation = CubicSpline(
		x=date_rates.index.values,
		y=date_rates["rate"].to_numpy(),
		bc_type="natural",
	)
	risk_free_interest_rate = interpolation(deannualise(time_to_expiry=time_to_expiry)) / 100

	return risk_free_interest_rate  # type: ignore
