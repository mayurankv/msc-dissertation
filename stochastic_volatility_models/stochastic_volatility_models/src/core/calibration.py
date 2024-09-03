from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Optional
import numpy as np
from numpy.typing import NDArray
from loguru import logger
import gc

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.volatility_surface import JointVolatilitySurface
	from stochastic_volatility_models.src.core.model import StochasticVolatilityModel
	from stochastic_volatility_models.src.core.pricing_models import PricingModel
from stochastic_volatility_models.src.core.evaluation_functions import surface_evaluation


class CostFunctionWeights(TypedDict):
	volatility_index: float
	skew: float


DEFAULT_COST_FUNCTION_WEIGHTS: CostFunctionWeights = {
	"volatility_index": 1.0,
	"skew": 0.01,
}
CONSTRAINT_LOSS = 10


def cost_function(
	joint_volatility_surface: JointVolatilitySurface,
	time: np.datetime64,
	model: StochasticVolatilityModel,
	empirical_pricing_model: PricingModel,
	model_pricing_model: PricingModel,
	volatility_empirical_pricing_model: PricingModel,
	volatility_model_pricing_model: PricingModel,
	weights: CostFunctionWeights = DEFAULT_COST_FUNCTION_WEIGHTS,
	out_the_money: bool = True,
	volatility_out_the_money: bool = True,
	call: Optional[bool] = None,
	volatility_call: Optional[bool] = None,
) -> float:
	cost = np.sqrt(
		(
			surface_evaluation(
				joint_volatility_surface=joint_volatility_surface,
				volatility=False,
				time=time,
				model=model,
				empirical_pricing_model=empirical_pricing_model,
				model_pricing_model=model_pricing_model,
				out_the_money=out_the_money,
				call=call,
			)
			+ weights["volatility_index"]
			* surface_evaluation(
				joint_volatility_surface=joint_volatility_surface,
				volatility=True,
				time=time,
				model=model,
				empirical_pricing_model=volatility_empirical_pricing_model,
				model_pricing_model=volatility_model_pricing_model,
				out_the_money=volatility_out_the_money,
				call=volatility_call,
			)
			# + weights["skew"]
			# * surface_atm_skew(
			# 	joint_volatility_surface=joint_volatility_surface,
			# 	volatility=False,
			# 	time=time,
			# 	model=model,
			# 	empirical_pricing_model=empirical_pricing_model,
			# 	model_pricing_model=model_pricing_model,
			# 	out_the_money=out_the_money,
			# 	call=call,
			# )
			# + weights["skew"]
			# * weights["volatility_index"]
			# * surface_atm_skew(
			# 	joint_volatility_surface=joint_volatility_surface,
			# 	volatility=True,
			# 	time=time,
			# 	model=model,
			# 	empirical_pricing_model=volatility_empirical_pricing_model,
			# 	model_pricing_model=volatility_model_pricing_model,
			# 	out_the_money=volatility_out_the_money,
			# 	call=volatility_call,
			# )
		)
		/ (1 + weights["volatility_index"])
	)

	return cost


def minimise_cost_function(
	parameters: NDArray[np.float64],
	joint_volatility_surface: JointVolatilitySurface,
	time: np.datetime64,
	model: StochasticVolatilityModel,
	empirical_pricing_model: PricingModel,
	model_pricing_model: PricingModel,
	volatility_empirical_pricing_model: PricingModel,
	volatility_model_pricing_model: PricingModel,
	weights: CostFunctionWeights,
	out_the_money: bool = True,
	volatility_out_the_money: bool = True,
	call: Optional[bool] = None,
	volatility_call: Optional[bool] = False,
) -> float:
	model.parameters = {parameter_key: parameter for parameter_key, parameter in zip(model.parameters.keys(), parameters)}

	cost = (
		cost_function(
			joint_volatility_surface=joint_volatility_surface,
			time=time,
			model=model,
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
		+ CONSTRAINT_LOSS * model.constraints()
	)

	logger.trace(f"Cost is {cost} with parameters {model.parameters}")
	logger.debug(f"Cost is {cost} with parameters {model.parameters}")

	gc.collect()

	return cost
