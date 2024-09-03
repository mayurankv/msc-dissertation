from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Optional
import numpy as np
from pandas import Index
from numpy.typing import NDArray
from loguru import logger

from stochastic_volatility_models.src.core.model import StochasticVolatilityModel, NUM_PATHS, SEED
from stochastic_volatility_models.src.models.heston.analytic_pricing import analytic_prices, DEFAULT_LG_DEGREE
from stochastic_volatility_models.src.models.heston.simulation import simulate
from stochastic_volatility_models.src.utils.options.expiry import time_to_expiry, DAYS
from stochastic_volatility_models.src.data.rates import get_risk_free_interest_rate
from stochastic_volatility_models.src.data.dividends import get_dividend_yield, interpolate_dividend_yield

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying


class HestonParameters(TypedDict):
	initial_variance: float  # v_0
	long_term_variance: float  # theta
	volatility_of_volatility: float  # eta
	mean_reversion_rate: float  # kappa
	wiener_correlation: float  # rho


HESTON_BOUNDS = {
	# "initial_variance": (0, np.inf),
	# "long_term_variance": (0, np.inf),
	# "volatility_of_volatility": (0, np.inf),
	# "mean_reversion_rate": (0, np.inf),
	"initial_variance": (0, 15),
	"long_term_variance": (0, 15),
	"volatility_of_volatility": (0, 15),
	"mean_reversion_rate": (0, 15),
	"wiener_correlation": (-1, 1),
}


class Heston(StochasticVolatilityModel):
	def __init__(
		self,
		parameters: HestonParameters,
	) -> None:
		self.name = "Heston"
		self.parameters: HestonParameters = parameters
		self.bounds = tuple([HESTON_BOUNDS[parameter] for parameter in self.parameters.keys()])

	def constraints(
		self,
	) -> float:
		# return np.maximum((self.parameters["volatility_of_volatility"] ** 2) - 2 * self.parameters["mean_reversion_rate"] * self.parameters["long_term_variance"], 0)
		return 0

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
		# TODO (@mayurankv): Distribution?
		# TODO (@mayurankv): Is this right?: `np.sqrt(self.parameters["long_term_variance"])`
		return 0

	def analytic_price(
		self,
		underlying: Underlying,
		time: np.datetime64,
		types: NDArray[str],  # type: ignore
		strikes: NDArray[np.int64],
		expiries: NDArray[np.datetime64],
		monthly: bool,
		legendre_gauss_degree: Optional[int] = DEFAULT_LG_DEGREE,
	) -> NDArray[np.float64]:
		logger.trace("Extracting parameters for Heston model pricing")
		spot = underlying.price(time=time)
		time_to_expiries = time_to_expiry(
			time=time,
			option_expiries=expiries,
		)
		logger.trace("Extracting risk free rates")
		risk_free_rates = get_risk_free_interest_rate(
			time=time,
			time_to_expiry=time_to_expiries,
		)
		logger.trace("Extracting dividend yields")
		dividend_yields = get_dividend_yield(
			underlying=underlying,
			time=time,
			strikes=strikes,
			expiries=expiries,
			monthly=monthly,
		)

		prices = analytic_prices(
			spot=spot,
			types=types,
			strikes=strikes,
			time_to_expiries=time_to_expiries,
			risk_free_rates=risk_free_rates,
			dividend_yields=dividend_yields,
			legendre_gauss_degree=legendre_gauss_degree,
			**self.parameters,
		)

		return prices

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
		logger.trace("Simulate paths")
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
		price_process, variance_process = simulate(
			spot=spot,
			time=time,
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

	def convert_variance_to_volatility_index(
		self,
		variance_process: NDArray[np.float64],
		volatility_index_window: float,
	) -> NDArray[np.float64]:
		delta_t = volatility_index_window
		a = self.parameters["mean_reversion_rate"] * self.parameters["long_term_variance"]
		b = -self.parameters["mean_reversion_rate"]
		volatility_index_process = 100 * np.sqrt((np.exp(b * delta_t) - 1) / (b * delta_t) * variance_process + (a * (np.exp(b * delta_t) - 1 - b * delta_t)) / (b**2 * delta_t))

		return volatility_index_process

	def convert_parameters_to_market_measure(
		self,
		time: np.datetime64,
		underlying: Underlying,
		volatility_underlying: Underlying,
	) -> tuple[float, float]:
		cir_market_price_volatility_risk_mean_reversion_rate = self.parameters["volatility_of_volatility"] * (1 - ((volatility_underlying.price(time) / 100) / np.sqrt(22) / underlying.realised_volatility(Index([time.astype(str)])).iloc[0]))
		cir_market_price_volatility_risk_long_term_variance = 0.0
		new_market_reversion_rate = self.parameters["mean_reversion_rate"] - cir_market_price_volatility_risk_mean_reversion_rate
		self.parameters["long_term_variance"] = (self.parameters["mean_reversion_rate"] * self.parameters["long_term_variance"] + cir_market_price_volatility_risk_mean_reversion_rate * cir_market_price_volatility_risk_long_term_variance) / new_market_reversion_rate
		self.parameters["mean_reversion_rate"] = new_market_reversion_rate

		return cir_market_price_volatility_risk_mean_reversion_rate, cir_market_price_volatility_risk_long_term_variance

	def fix_initial_variance(
		self,
		initial_daily_volatility: float,
	) -> None:
		initial_variance = (initial_daily_volatility**2) * DAYS
		self.bounds = ((initial_variance, initial_variance), *self.bounds[1:])
		self.parameters["initial_variance"] = initial_variance

	def set_upper_long_term_variance(
		self,
		max_daily_volatility: float,
	) -> None:
		upper_variance = (max_daily_volatility**2) * DAYS
		# self.parameters["volatility_of_volatility"] = np.sqrt(2 * self.parameters["long_term_variance"] * self.parameters["mean_reversion_rate"]) / 2
		# self.bounds = (self.bounds[0], (self.bounds[1][0], upper_variance), (self.bounds[2][0], np.sqrt(2 * upper_variance * self.bounds[3][1])), *self.bounds[3:])
		self.bounds = (self.bounds[0], (self.bounds[1][0], upper_variance), *self.bounds[2:])
