from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import os
import csv
from pandas import DataFrame, MultiIndex, to_datetime
import numpy as np
from loguru import logger
from abc import ABC, abstractmethod
from typing import Mapping
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from numba import njit, prange
import hashlib

from stochastic_volatility_models.config import MODULE_DIRECTORY
from stochastic_volatility_models.src.core.calibration import DEFAULT_COST_FUNCTION_WEIGHTS, minimise_cost_function, cost_function
from stochastic_volatility_models.src.utils.options.parameters import get_options_parameters_transpose
from stochastic_volatility_models.src.utils.options.expiry import DAYS, time_to_expiry
from stochastic_volatility_models.src.utils.cache import model_lru_cache
from stochastic_volatility_models.src.data.prices import get_forecast_dates
# from stochastic_volatility_models.src.data.rates import get_risk_free_interest_rate
# from stochastic_volatility_models.src.data.dividends import get_dividend_yield

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying
	from stochastic_volatility_models.src.core.pricing_models import PricingModel
	from stochastic_volatility_models.src.core.volatility_surface import JointVolatilitySurface
	from stochastic_volatility_models.src.core.calibration import CostFunctionWeights


SEED = 343
NUM_PATHS = 2**18
MIN_TICK_SIZE = 0.05
VOLATILITY_MIN_TICK_SIZE = 0.01
VOLATILITY_WINDOW = 1 / 12
MIN_BOUND_RESOLUTION = 0.001
CSV_LOG_HEADERS = ["volatility_index", "skew", "iterations", "RMSE error"]


@njit(parallel=True, cache=True)
def pricing_index_options(
	price_process: NDArray[np.float64],
	flags: NDArray[np.int64],
	strikes: NDArray[np.int64],
	time_to_expiries: NDArray[np.float64],
	# risk_free_rates: NDArray[np.float64],
	# dividend_yields: NDArray[np.float64],
	steps_per_year: int,
) -> NDArray[np.float64]:
	prices = np.zeros_like(strikes, dtype=np.float64)
	for idx in prange(strikes.shape[0]):
		prices[idx] = np.mean(
			np.maximum(
				(price_process[:, int(time_to_expiries[idx] * steps_per_year)] - strikes[idx]) * flags[idx],
				0.0,
			)
		)

	return prices


@njit(parallel=True, cache=True)
def pricing_volatility_index_options_from_variance_paths(
	variance_process: NDArray[np.float64],
	flags: NDArray[np.int64],
	strikes: NDArray[np.int64],
	time_to_expiries: NDArray[np.float64],
	# risk_free_rates: NDArray[np.float64],
	# dividend_yields: NDArray[np.float64],
	steps_per_year: int,
) -> NDArray[np.float64]:
	dt = 1 / steps_per_year
	prices = np.zeros_like(strikes, dtype=np.float64)
	for idx in prange(strikes.shape[0]):
		prices[idx] = np.mean(
			np.maximum(
				(
					100
					* np.sqrt(
						np.sum(
							variance_process[
								:,
								int(time_to_expiries[idx] * steps_per_year) : int((time_to_expiries[idx] + VOLATILITY_WINDOW) * steps_per_year),
							],
							axis=1,
						)
						* dt
						/ VOLATILITY_WINDOW
					)
					- strikes[idx]
				)
				* flags[idx],
				0.0,
			)
		)

	return prices


@njit(parallel=True, cache=True)
def pricing_volatility_index_options_from_price_paths(
	price_process: NDArray[np.float64],
	flags: NDArray[np.int64],
	strikes: NDArray[np.int64],
	time_to_expiries: NDArray[np.float64],
	# risk_free_rates: NDArray[np.float64],
	# dividend_yields: NDArray[np.float64],
	steps_per_year: int,
) -> NDArray[np.float64]:
	prices = np.zeros_like(strikes, dtype=np.float64)
	for idx in prange(strikes.shape[0]):
		prices[idx] = np.mean(
			np.maximum(
				(
					100
					* np.sqrt(
						np.maximum(
							-(2 / VOLATILITY_WINDOW)
							* (
								np.log(
									price_process[
										:,
										int((time_to_expiries[idx] + VOLATILITY_WINDOW) * steps_per_year),
									]
								)
								- np.log(
									price_process[
										:,
										int(time_to_expiries[idx] * steps_per_year),
									]
								)
							),
							0,
						)
					)
					- strikes[idx]
				)
				* flags[idx],
				0.0,
			)
		)

	return prices


class StochasticVolatilityModel(ABC):
	def __init__(
		self,
		parameters: Mapping,
	) -> None:
		self.name = ""
		self.parameters = parameters
		self.bounds = tuple()

	def constraints(
		self,
	) -> float:
		return 0

	@abstractmethod
	def integrated_volatility(
		self,
		underlying: Underlying,
		time: np.datetime64,
	) -> float:
		# TODO (@mayurankv): Distribution?
		pass

	@abstractmethod
	def volatility(
		self,
		underlying: Underlying,
		time: np.datetime64,
	) -> float:
		# TODO (@mayurankv): Distribution?
		pass

	def price(
		self,
		underlying: Underlying,
		volatility_underlying: Underlying,
		time: np.datetime64,
		types: NDArray[str],  # type: ignore
		strikes: NDArray[np.int64],
		expiries: NDArray[np.datetime64],
		volatility_types: NDArray[str],  # type: ignore
		volatility_strikes: NDArray[np.int64],
		volatility_expiries: NDArray[np.datetime64],
		monthly: bool = True,
		steps_per_year: int = int(DAYS),
		num_paths: int = NUM_PATHS,
		seed: int = SEED,
	) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		logger.trace("Extracting parameters for Rough Bergomi model pricing")
		time_to_expiries = time_to_expiry(
			time=time,
			option_expiries=expiries,
		)
		volatility_time_to_expiries = time_to_expiry(
			time=time,
			option_expiries=volatility_expiries,
		)
		# logger.trace("Extracting risk free rates")
		# risk_free_rates = get_risk_free_interest_rate(
		# 	time=time,
		# 	time_to_expiry=time_to_expiries,
		# )
		# logger.trace("Extracting dividend yields")
		# dividend_yields = get_dividend_yield(
		# 	underlying=underlying,
		# 	time=time,
		# 	expiries=expiries,
		# 	monthly=monthly,
		# )

		price_process, variance_process = self.simulate_path(
			underlying=underlying,
			volatility_underlying=volatility_underlying,
			time=time,
			simulation_length=time_to_expiries.max(),
			steps_per_year=steps_per_year,
			num_paths=num_paths,
			seed=seed,
			monthly=monthly,
		)

		logger.trace("Price options")
		prices = pricing_index_options(
			price_process=price_process,
			flags=(types == "C") * 2 - 1,
			strikes=strikes,
			time_to_expiries=time_to_expiries,
			steps_per_year=steps_per_year,
		)

		logger.trace("Price volatility options")
		volatility_prices = (
			pricing_index_options(
				price_process=self.convert_variance_to_volatility_index(  # type: ignore
					variance_process=variance_process,
					volatility_index_window=VOLATILITY_WINDOW,
				),
				flags=(volatility_types == "C") * 2 - 1,
				strikes=volatility_strikes,
				time_to_expiries=volatility_time_to_expiries,
				steps_per_year=steps_per_year,
			)
			if callable(getattr(self, "convert_variance_to_volatility_index", None))
			else pricing_volatility_index_options_from_variance_paths(
				variance_process=variance_process,
				flags=(volatility_types == "C") * 2 - 1,
				strikes=volatility_strikes,
				time_to_expiries=volatility_time_to_expiries,
				steps_per_year=steps_per_year,
			)
		)

		return prices, volatility_prices

	@model_lru_cache(maxsize=4)
	def price_surface(
		self,
		underlying: Underlying,
		volatility_underlying: Underlying,
		time: np.datetime64,
		symbols: tuple[str, ...],  # type: ignore
		volatility_symbols: tuple[str, ...],  # type: ignore
		monthly: bool = True,
		*args,
		**kwargs,
	) -> tuple[DataFrame, DataFrame]:
		options_parameters_transpose = get_options_parameters_transpose(
			ticker=underlying.ticker,
			symbols=symbols,
		)
		volatility_options_parameters_transpose = get_options_parameters_transpose(
			ticker=volatility_underlying.ticker,
			symbols=volatility_symbols,
		)

		prices_array, volatility_prices_array = self.price(
			underlying=underlying,
			volatility_underlying=volatility_underlying,
			time=time,
			types=np.array(options_parameters_transpose["type"]),
			strikes=np.array(options_parameters_transpose["strike"]),
			expiries=np.array(options_parameters_transpose["expiry"]),
			volatility_types=np.array(volatility_options_parameters_transpose["type"]),
			volatility_strikes=np.array(volatility_options_parameters_transpose["strike"]),
			volatility_expiries=np.array(volatility_options_parameters_transpose["expiry"]),
			monthly=monthly,
			*args,
			**kwargs,
		)
		prices = (
			DataFrame(
				data=prices_array,
				index=symbols,
				columns=["Mid"],
			)
			.fillna(0)
			.clip(lower=MIN_TICK_SIZE / 2)
		)
		volatility_prices = (
			DataFrame(
				data=volatility_prices_array,
				index=volatility_symbols,
				columns=["Mid"],
			)
			.fillna(0)
			.clip(lower=VOLATILITY_MIN_TICK_SIZE / 2)
		)

		return prices, volatility_prices

	@abstractmethod
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
		pass

	def fit(
		self,
		joint_volatility_surface: JointVolatilitySurface,
		time: np.datetime64,
		empirical_pricing_model: PricingModel,
		model_pricing_model: PricingModel,
		volatility_empirical_pricing_model: PricingModel,
		volatility_model_pricing_model: PricingModel,
		weights: CostFunctionWeights = DEFAULT_COST_FUNCTION_WEIGHTS,
		out_the_money: bool = True,
		volatility_out_the_money: bool = True,
		call: Optional[bool] = None,
		volatility_call: Optional[bool] = True,
		niter: int = 5,
	) -> dict:
		results = differential_evolution(
			func=minimise_cost_function,
			bounds=tuple((lb + MIN_BOUND_RESOLUTION * diff, ub - MIN_BOUND_RESOLUTION * diff) if ((diff := ub - lb) == 0) else (lb, ub) for (lb, ub) in self.bounds),
			x0=np.array(list(self.parameters.values())),
			args=(
				joint_volatility_surface,
				time,
				self,
				empirical_pricing_model,
				model_pricing_model,
				volatility_empirical_pricing_model,
				volatility_model_pricing_model,
				weights,
				out_the_money,
				volatility_out_the_money,
				call,
				volatility_call,
			),
			strategy="best1bin",
			mutation=(0.01, 0.05),
			maxiter=niter,
			recombination=0.2,
			tol=0.01,
		)
		self.parameters: Mapping = {parameter_key: parameter for parameter_key, parameter in zip(self.parameters.keys(), results["x"])}

		path: str = f"{MODULE_DIRECTORY}/logs/minimisation/{self.name.lower().replace(' ', '_')}/{time}_otm_{out_the_money}_call_{call}_{'Y' if joint_volatility_surface.monthly else 'N'}monthly_{joint_volatility_surface.underlying.ticker}_{int(hashlib.md5(str(tuple(joint_volatility_surface.strikes)).replace(' ','').encode("utf8")).hexdigest(), 16)}_{int(hashlib.md5(str(tuple(joint_volatility_surface.expiries.astype(str))).replace(' ','').encode("utf8")).hexdigest(), 16)}_{joint_volatility_surface.volatility_underlying.ticker}_{int(hashlib.md5(str(tuple(joint_volatility_surface.volatility_strikes)).replace(' ','').encode("utf8")).hexdigest(), 16)}_{int(hashlib.md5(str(tuple(joint_volatility_surface.volatility_expiries.astype(str))).replace(' ','').encode("utf8")).hexdigest(), 16)}_MPM_{model_pricing_model.model.lower().replace(' ','_').replace('-','_')}_EPM_{empirical_pricing_model.model.lower().replace(' ','_').replace('-','_')}.csv"
		with open(file=path, mode="a") as file:
			parameters = sorted(self.parameters.keys())
			writer = csv.writer(file)
			if not os.path.isfile(path):
				writer.writerow(CSV_LOG_HEADERS + parameters)
			writer.writerow(
				[
					weights["volatility_index"],
					weights["skew"],
					niter,
					results["fun"],
				]
				+ [self.parameters[parameter] for parameter in parameters],
			)

		return self.parameters

	def evaluate_fit(
		self,
		joint_volatility_surface: JointVolatilitySurface,
		time: np.datetime64,
		empirical_pricing_model: PricingModel,
		model_pricing_model: PricingModel,
		volatility_empirical_pricing_model: PricingModel,
		volatility_model_pricing_model: PricingModel,
		weights: CostFunctionWeights = DEFAULT_COST_FUNCTION_WEIGHTS,
		out_the_money: bool = True,
		volatility_out_the_money: bool = True,
		call: Optional[bool] = None,
		volatility_call: Optional[bool] = True,
	) -> float:
		cost = cost_function(
			joint_volatility_surface=joint_volatility_surface,
			time=time,
			model=self,
			empirical_pricing_model=empirical_pricing_model,
			model_pricing_model=model_pricing_model,
			volatility_empirical_pricing_model=volatility_empirical_pricing_model,
			volatility_model_pricing_model=volatility_model_pricing_model,
			weights=weights,
			out_the_money=out_the_money,
			volatility_out_the_money=volatility_out_the_money,
			call=call,
			volatility_call=volatility_call,
		)

		return cost

	def forecast_volatility(
		self,
		underlying: Underlying,
		volatility_underlying: Underlying,
		time: np.datetime64,
		simulation_length: float = 1.0,
		steps_per_year: int = int(DAYS),
		num_paths: int = NUM_PATHS,
		monthly: bool = True,
		forecast_confidences: list[float] = [0.95, 0.99],
		use_drift: bool = True,
		seed: int = SEED,
	) -> DataFrame:
		_, variance_process = self.simulate_path(
			underlying=underlying,
			volatility_underlying=volatility_underlying,
			time=time,
			simulation_length=simulation_length,
			steps_per_year=steps_per_year,
			num_paths=num_paths,
			monthly=monthly,
			use_drift=use_drift,
			seed=seed,
		)

		forecast_dates = to_datetime(
			get_forecast_dates(
				underlying.ticker,
				time,
				simulation_length,
			)
		)

		forecast_confidences_arr = np.array([round((confidence + 1) / 2, 4) for confidence in forecast_confidences])
		annualised_volatility_process = np.sqrt(variance_process)
		annualised_volatility_forecast = DataFrame(
			data=0.0,
			index=forecast_dates,
			columns=MultiIndex.from_tuples([("Forecast", "Mean")] + [("Lower", confidence) for confidence in forecast_confidences_arr] + [("Upper", confidence) for confidence in forecast_confidences_arr]),
		)
		annualised_volatility_forecast[("Forecast", "Mean")] = np.mean(annualised_volatility_process, axis=0)
		annualised_volatility_forecast["Upper"] = np.quantile(annualised_volatility_process, q=forecast_confidences_arr, axis=0).T
		annualised_volatility_forecast["Lower"] = np.quantile(annualised_volatility_process, q=1 - forecast_confidences_arr, axis=0).T

		volatility_forecast = annualised_volatility_forecast / np.sqrt(252)

		return volatility_forecast
