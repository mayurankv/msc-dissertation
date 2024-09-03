from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from pandas import DataFrame

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.volatility_surface import JointVolatilitySurface
from stochastic_volatility_models.src.utils.options.strikes import moneyness, find_closest_strikes


def atm_skew(
	surface: DataFrame,
	joint_volatility_surface: JointVolatilitySurface,
	volatility: bool,
	time: np.datetime64,
) -> DataFrame:
	underlying = joint_volatility_surface.underlying if not volatility else joint_volatility_surface.volatility_underlying
	spot = underlying.price(time=time)
	strikes = joint_volatility_surface.strikes if not volatility else joint_volatility_surface.volatility_strikes
	expiries = joint_volatility_surface.expiries if not volatility else joint_volatility_surface.volatility_expiries
	indices = find_closest_strikes(
		strikes=strikes,
		spot=spot,
	)
	atm_skews = DataFrame(
		data=[
			np.polyfit(
				x=moneyness(underlying=underlying, strikes=indices, time=time, log=True),
				y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[indices, "Symbol"].to_numpy(),
				deg=1,
			)[0]
			for expiry in expiries
		],
		index=[expiry for expiry in expiries],
		columns=["Skew"],
	)

	return atm_skews
