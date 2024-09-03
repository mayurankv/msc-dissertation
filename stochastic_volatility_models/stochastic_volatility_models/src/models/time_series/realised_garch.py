from __future__ import annotations
import numpy as np
from rpy2.robjects import r, numpy2ri
import rpy2.robjects as ro
from numpy.typing import NDArray
from loguru import logger

from stochastic_volatility_models.src.core.underlying import Underlying
from stochastic_volatility_models.src.utils.options.expiry import DAYS


class ReGARCH:
	"""
	ReGARCH

	Realised GARCH model

	source: https://www.r-bloggers.com/2014/01/the-realized-garch-model/
	"""

	def __init__(
		self,
		include_mean: bool = True,
		ar_order: int = 1,
		ma_order: int = 1,
		garch_ar_order: int = 1,
		garch_ex_order: int = 1,
	) -> None:
		self.include_mean = include_mean
		self.ar_order = ar_order
		self.ma_order = ma_order
		self.garch_ar_order = garch_ar_order
		self.garch_ex_order = garch_ex_order
		logger.trace("Initialise rpy2 functions")
		with open("/Users/mayurankv/Documents/Mayuran/Programming/Projects/Academic/Imperial College London/MSc Statistics/Dissertation/Project/modules/stochastic_volatility_models/stochastic_volatility_models/src/models/time_series/realised_garch.r", "r") as file:
			r(file.read())

	def fit(
		self,
		time: np.datetime64,
		underlying: Underlying,
		period: float = 1,
	) -> float:
		logger.trace("Fit RealGARCH")
		self.time = time
		returns = underlying.returns(
			time=time,
			period=period,
		)
		realised_volatility = underlying.realised_volatility(
			dates=returns.index,
		)
		self.fitting_dates = returns.index

		numpy2ri.activate()
		ro.globalenv["index"] = returns.index.to_numpy()
		ro.globalenv["returns"] = returns.to_numpy()
		ro.globalenv["realised_vol"] = realised_volatility.to_numpy()
		numpy2ri.deactivate()
		r(f"model <- fit_model({str(self.include_mean).upper()}, {self.ar_order}, {self.ma_order}, {self.garch_ar_order}, {self.garch_ex_order}, returns, realised_vol, index)")

		self.model = r("model")
		self.unconditional_variance = list(r("unconditional_variance(model)"))[0]  # type: ignore

		return self.unconditional_variance

	def forecast(
		self,
		forecast_period: float = 1,
		simulations: int = 100000,
	) -> NDArray[np.float64]:
		logger.trace("Forecast RealGARCH")
		steps = int(forecast_period * DAYS)
		ro.globalenv["model"] = self.model
		r(f"forecast <- forecast_model(model, {steps}, {simulations})")

		self.volatility_forecast = np.array(r("forecast$forecast"))
		self.volatility_distribution = np.array(r("forecast$distribution"))

		return self.volatility_forecast

	def quantiles(
		self,
		probs: list[float] = [0.95, 0.99],
	) -> NDArray[np.float64]:
		logger.trace("Get RealGARCH quantiles")
		self.probs = np.array([[round(1 - prob, ndigits=3), prob] for prob in probs]).ravel()
		self.volatility_quantiles = np.quantile(
			self.volatility_distribution,
			q=self.probs,
			axis=1,
		)

		return self.volatility_quantiles


# print(base.sqrt(rugarch.uncvariance(model)))
