import numpy as np
from numpy.typing import NDArray

TRADING_DAYS = 252
TOTAL_DAYS = 365
YEAR = np.timedelta64(TRADING_DAYS, "D")
DAYS = float(YEAR / np.timedelta64(1, "D"))


def annualise(
	delta: NDArray[np.float64],
) -> NDArray[np.float64]:
	return delta / DAYS


def deannualise(
	time_to_expiry: NDArray[np.float64],
) -> NDArray[np.float64]:
	return time_to_expiry * DAYS


def time_to_expiry(
	time: np.datetime64,
	option_expiries: NDArray[np.datetime64],
) -> NDArray[np.float64]:
	days = np.array(
		[
			np.sum(
				np.is_busday(
					dates=np.arange(
						start=time,
						stop=expiry,
						dtype=np.datetime64,
					)
				),
				dtype=np.int64,
			)
			for expiry in option_expiries
		]
	)

	return annualise(delta=days)
