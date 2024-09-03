from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict
from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pandas import Index

from stochastic_volatility_models.src.core.model import StochasticVolatilityModel, NUM_PATHS, SEED
from stochastic_volatility_models.src.data.rates import get_risk_free_interest_rate
from stochastic_volatility_models.src.data.dividends import interpolate_dividend_yield
from stochastic_volatility_models.src.models.rough_bergomi.simulation import simulate
from stochastic_volatility_models.src.utils.options.expiry import DAYS

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying


class RoughBergomiParameters(TypedDict):
	hurst_index: float  # H
	volatility_of_volatility: float  # eta
	wiener_correlation: float  # rho


ROUGH_BERGOMI_BOUNDS = {
	"hurst_index": (0.0, 0.5),
	# "volatility_of_volatility": (0, np.inf),
	"volatility_of_volatility": (0, 10),
	"wiener_correlation": (-1, 1),
}


class RoughBergomi(StochasticVolatilityModel):
	def __init__(
		self,
		parameters: RoughBergomiParameters,
	) -> None:
		self.name = "Rough Bergomi"
		self.parameters: RoughBergomiParameters = parameters
		self.bounds = tuple([ROUGH_BERGOMI_BOUNDS[parameter] for parameter in self.parameters.keys()])

	def integrated_volatility(
		self,
		underlying: Underlying,
		time: np.datetime64,
	) -> float:
		# TODO (@mayurankv): Finish
		return 0

	def volatility(
		self,
		underlying: Underlying,
		time: np.datetime64,
	) -> float:
		# TODO (@mayurankv): Finish
		# TODO (@mayurankv): Distribution?
		return 0

	def simulate_path(
		self,
		underlying: Underlying,
		volatility_underlying: Underlying,
		time: np.datetime64,
		simulation_length: float = 1.0,
		steps_per_year: int = int(DAYS),
		num_paths: int = NUM_PATHS,
		monthly: bool = True,
		use_drift: bool = False,
		seed: int = SEED,
	) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		# TODO (@mayurankv): Is this right?
		initial_variance = (underlying.realised_volatility(Index([time.astype(str)])).iloc[0] ** 2) * DAYS
		(volatility_underlying.price(time) / 100) ** 2  # 0.2885 ** 2
		spot = underlying.price(time)
		risk_free_rates = (
			get_risk_free_interest_rate(
				time=time,
				time_to_expiry=np.linspace(start=0, stop=simulation_length, num=1 + int(steps_per_year * simulation_length)),
			)
			if use_drift
			else np.zeros(1 + int(steps_per_year * simulation_length))
		)
		dividend_yields = (
			interpolate_dividend_yield(
				ticker=underlying.ticker,
				spot=spot,
				time=time,
				strikes=None,
				time_to_expiries=np.linspace(start=0, stop=simulation_length, num=1 + int(steps_per_year * simulation_length)),
				monthly=monthly,
			)
			if use_drift
			else np.zeros(1 + int(steps_per_year * simulation_length))
		)

		logger.trace("Simulate paths")
		price_process, variance_process, _ = simulate(
			ticker=underlying.ticker,
			spot=spot,
			time=time,
			initial_variance=initial_variance,
			**self.parameters,
			simulation_length=simulation_length,
			steps_per_year=steps_per_year,
			num_paths=num_paths,
			monthly=monthly,
			risk_free_rates=risk_free_rates,
			dividend_yields=dividend_yields,
			seed=seed,
		)

		return price_process, variance_process
