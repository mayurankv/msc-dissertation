from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying


def find_closest_strikes(
	strikes: NDArray[np.int64],
	spot: float,
	n: int = 3,
) -> NDArray[np.int64]:
	strikes = np.sort(strikes)
	spot_idx = np.searchsorted(strikes, spot)
	strikes = np.concatenate([strikes[spot_idx - n : spot_idx], strikes[spot_idx : spot_idx + n]])

	return strikes


def moneyness(
	underlying: Underlying,
	strikes: NDArray[np.int64],
	time: np.datetime64,
	future: bool = False,
	log: bool = False,
) -> NDArray[np.float64]:
	moneyness = (underlying.future_price(time=time) if future else underlying.price(time=time)) / strikes
	if log:
		moneyness = np.log(moneyness)
	return moneyness
