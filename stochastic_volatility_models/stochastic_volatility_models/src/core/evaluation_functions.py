from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.volatility_surface import JointVolatilitySurface
	from stochastic_volatility_models.src.core.model import StochasticVolatilityModel
	from stochastic_volatility_models.src.core.pricing_models import PricingModel
from stochastic_volatility_models.src.utils.metrics import Metrics, METRICS
from stochastic_volatility_models.src.utils.options.skew import atm_skew


def surface_evaluation(
	joint_volatility_surface: JointVolatilitySurface,
	volatility: bool,
	time: np.datetime64,
	model: StochasticVolatilityModel,
	empirical_pricing_model: Optional[PricingModel] = None,
	model_pricing_model: Optional[PricingModel] = None,
	prices: bool = False,
	metric: Metrics = "MSE",
	out_the_money: bool = True,
	call: Optional[bool] = None,
) -> float:
	if not prices and (empirical_pricing_model is None or model_pricing_model is None):
		raise ValueError("If evaluating volatilities, pricing models must be provided")
	empirical_kwargs = {} if prices else {"pricing_model": empirical_pricing_model}
	model_kwargs = {} if prices else {"pricing_model": model_pricing_model}
	loss = METRICS[metric](
		joint_volatility_surface.surface_quantities(
			time=time,
			quantity_method="empirical_price" if prices else "empirical_pricing_implied_volatility",
			price_types=["Mid"],
			out_the_money=out_the_money,
			call=call,
			**empirical_kwargs,
		)[int(volatility)][0].values,
		joint_volatility_surface.surface_quantities(
			time=time,
			quantity_method="model_price" if prices else "model_pricing_implied_volatility",
			price_types=["Mid"],
			out_the_money=out_the_money,
			model=model,
			call=call,
			**model_kwargs,
		)[int(volatility)][0].values,
	)

	return loss


def surface_atm_skew(
	joint_volatility_surface: JointVolatilitySurface,
	volatility: bool,
	time: np.datetime64,
	model: StochasticVolatilityModel,
	empirical_pricing_model: PricingModel,
	model_pricing_model: PricingModel,
	metric: Metrics = "RMSE",
	out_the_money: bool = True,
	call: Optional[bool] = None,
) -> float:
	surfaces = [
		joint_volatility_surface.surface_quantities(
			time=time,
			quantity_method="empirical_pricing_implied_volatility",
			price_types=["Mid"],
			out_the_money=out_the_money,
			call=call,
			pricing_model=empirical_pricing_model,
		)[int(volatility)][0],
		joint_volatility_surface.surface_quantities(
			time=time,
			quantity_method="model_pricing_implied_volatility",
			price_types=["Mid"],
			out_the_money=out_the_money,
			call=call,
			model=model,
			pricing_model=model_pricing_model,
		)[int(volatility)][0],
	]

	atm_skews = [
		atm_skew(
			surface=surface,
			joint_volatility_surface=joint_volatility_surface,
			volatility=volatility,
			time=time,
		).values
		for surface in surfaces
	]

	loss = METRICS[metric](*atm_skews)

	return loss
