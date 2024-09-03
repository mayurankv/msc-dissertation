from __future__ import annotations
from typing import TYPE_CHECKING, cast
import pandas as pd
from pandas import DataFrame, Index, Series
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
from loguru import logger

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying
from stochastic_volatility_models.config import MODULE_DIRECTORY
from stochastic_volatility_models.src.utils.options.expiry import time_to_expiry
from stochastic_volatility_models.src.data.rates import get_risk_free_interest_rate
from stochastic_volatility_models.src.data.prices import DEFAULT_CHUNKSIZE


@lru_cache
def get_atm_prices(
	ticker: str,
	spot: float,
	time: np.datetime64,
	strikes: tuple[int] | None,
	monthly: bool = True,
	chunksize: int = DEFAULT_CHUNKSIZE,
) -> DataFrame:
	reference = spot * 1000

	def get_atm_strikes(
		strike_df: DataFrame,
	) -> DataFrame:
		strike_df.index = strike_df.index.droplevel([0, 1])
		if strikes is not None:
			strike_df = strike_df.loc[strike_df.index.intersection(Index(strikes))]
		exactmatch = strike_df[strike_df.index == reference]
		if not exactmatch.empty:
			strike = exactmatch.index
		else:
			strike = upperneighbour_ind if (upperneighbour_ind := strike_df[strike_df.index > reference].index.min()) + (lowerneighbour_ind := strike_df[strike_df.index < reference].index.max()) <= 2 * reference else lowerneighbour_ind

		atm_strikes = strike_df.loc[Index([strike])]
		atm_strikes.index.name = "strike_price"

		return atm_strikes

	path: str = f"{MODULE_DIRECTORY}/data/options/{ticker.lower()}.csv"
	option_prices_iter = pd.read_csv(
		path,
		usecols=["symbol", "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"],
		index_col=[0, 1],
		chunksize=chunksize,
	)

	option_strikes = (
		pd.concat(
			[
				(
					prices := option_prices.xs(
						date_key,
						level=1,
					)
				).loc[(prices.index.get_level_values(0).str.startswith(ticker + (" " if monthly else "W "))), ["exdate", "cp_flag", "strike_price", "best_bid", "best_offer"]]
				for option_prices in option_prices_iter
				if (date_key := np.datetime_as_string(time, unit="D")) in option_prices.index.get_level_values(level=1)
			]
		)
		.reset_index()
		.drop(labels=["symbol"], axis=1)
		.set_index(keys=["exdate", "cp_flag", "strike_price"])
	)

	logger.trace(f"Found expiries: {set(option_strikes.index.get_level_values(level=0).unique())}")

	option_strikes["Mid"] = (option_strikes["best_bid"] + option_strikes["best_offer"]) / 2
	option_strikes = option_strikes.drop(["best_bid", "best_offer"], axis=1)

	atm_prices = option_strikes.groupby(level=[0, 1]).apply(func=get_atm_strikes)["Mid"].unstack(level=1).reset_index().set_index(keys="exdate").sort_index()

	return atm_prices


def get_option_future_price(
	underlying: Underlying,
	time: np.datetime64,
	strikes: NDArray[np.float64],
	expiries: NDArray[np.datetime64],
	monthly: bool = True,
	chunksize: int = DEFAULT_CHUNKSIZE,
) -> NDArray[np.float64]:
	atm_prices = get_atm_prices(
		ticker=underlying.ticker,
		spot=underlying.price(time=time),
		time=time,
		strikes=tuple(strikes * 1000),
		monthly=monthly,
		chunksize=chunksize,
	)
	t2x = time_to_expiry(time=time, option_expiries=expiries)
	r = get_risk_free_interest_rate(time=time, time_to_expiry=t2x)
	print(atm_prices.shape, np.exp(-r * t2x).shape)
	forward_prices = (atm_prices["C"] - atm_prices["P"]) * np.exp(-r * t2x) + (atm_prices["strike_price"] / 1000)
	forward_prices = cast(Series, forward_prices).to_numpy()

	return forward_prices
