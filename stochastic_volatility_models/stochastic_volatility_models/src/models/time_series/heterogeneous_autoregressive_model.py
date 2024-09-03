from __future__ import annotations
import numpy as np
from loguru import logger
from pandas import DataFrame, MultiIndex, IndexSlice, to_datetime
import statsmodels.api as sm
from scipy.stats import norm

from stochastic_volatility_models.src.core.underlying import Underlying
from stochastic_volatility_models.src.utils.options.expiry import DAYS
from stochastic_volatility_models.src.data.prices import get_forecast_dates


class HAR:
	def __init__(
		self,
	) -> None:
		return

	def fit(
		self,
		time: np.datetime64,
		underlying: Underlying,
		fitting_period: float = 1,
	):
		logger.trace("Fit HAR")
		self.ticker = underlying.ticker
		self.time = time
		fitting_days = int(fitting_period * DAYS)
		returns = underlying.returns(
			time=time,
			period=fitting_period + ((1 + 22) / DAYS),
		)
		realised_volatility = underlying.realised_volatility(
			dates=returns.index,
		)
		self.fitting_dates = returns.index[-fitting_days:]

		realised_variance = DataFrame(data=np.square(realised_volatility.to_numpy()), index=realised_volatility.index, columns=["RV_t"])
		realised_variance["RV_daily"] = realised_variance["RV_t"].shift(1)
		weekly_realised_variance = realised_variance["RV_t"].rolling(window=5).mean()
		realised_variance["RV_weekly"] = weekly_realised_variance.shift(1)
		monthly_realised_variance = realised_variance["RV_t"].rolling(window=22).mean()
		realised_variance["RV_monthly"] = monthly_realised_variance.shift(1)
		realised_variance = realised_variance.dropna().tail(fitting_days)

		log_realised_variance = np.log(realised_variance)
		self.current_log_realised_variances = (np.log(realised_variance["RV_t"].iloc[-1]), np.log(weekly_realised_variance.iloc[-1]), np.log(monthly_realised_variance.iloc[-1]))  # type: ignore
		self.past_predictions = np.log(realised_variance["RV_t"].iloc[-22:].to_numpy())

		self.model = sm.OLS(endog=log_realised_variance["RV_t"], exog=sm.add_constant(log_realised_variance[["RV_daily", "RV_weekly", "RV_monthly"]])).fit()

		return self.model.summary()

	def forecast(
		self,
		forecast_period: float = 1,
		forecast_confidences: list[float] = [0.95, 0.99],
	) -> DataFrame:
		logger.trace("Forecast HAR")
		self.steps = int(forecast_period * DAYS)
		forecast_confidences_arr = np.array([round((confidence + 1) / 2, 4) for confidence in forecast_confidences])
		volatility_forecast = DataFrame(data=0.0, index=list(range(self.steps + 1)), columns=MultiIndex.from_tuples([("Forecast", "Mean")] + [("Lower", prob) for prob in forecast_confidences_arr] + [("Upper", prob) for prob in forecast_confidences_arr]))
		ols_sample_variance = self.model.resid.var()
		past_predictions = self.past_predictions
		past_variances = np.zeros(22)
		coefficients = np.empty((3, 22))
		coefficients[2] = self.model.params["RV_monthly"] / 22
		coefficients[1, 0:5] = self.model.params["RV_weekly"] / 5
		coefficients[0, 0] = self.model.params["RV_daily"]
		coefficients = np.sum(coefficients, axis=0)
		coefficient_variances = np.empty((3, 22))
		coefficient_variances[2] = (self.model.bse["RV_monthly"] / 22) ** 2
		coefficient_variances[1, 0:5] = (self.model.bse["RV_weekly"] / 5) ** 2
		coefficient_variances[0, 0] = (self.model.bse["RV_daily"]) ** 2
		coefficient_variances = np.sum(coefficient_variances, axis=0)

		current_log_realised_variances = self.current_log_realised_variances
		volatility_forecast.iloc[0] = [np.exp(current_log_realised_variances[0] / 2)] * len(volatility_forecast.columns)  # type: ignore

		for i in range(1, self.steps + 1):
			current_log_realised_variances = np.array([1, *current_log_realised_variances])

			prediction = self.model.predict(current_log_realised_variances)[0]
			volatility_forecast.at[i, ("Forecast", "Mean")] = np.exp((prediction / 2) + (ols_sample_variance / 8))
			variance = (self.model.bse["const"] ** 2) + np.sum((coefficients**2) * past_variances) + np.sum(coefficient_variances * past_variances) + np.sum(coefficient_variances * (past_predictions**2)) + ols_sample_variance
			past_variances[1:] = past_variances[:-1]
			past_variances[0] = variance
			past_predictions[1:] = past_predictions[:-1]
			past_predictions[0] = prediction
			volatility_forecast.loc[i, IndexSlice["Upper"]] = np.exp((np.sqrt(variance) * norm.ppf(q=forecast_confidences_arr) + prediction) / 2)  # type: ignore
			volatility_forecast.loc[i, IndexSlice["Lower"]] = np.exp((np.sqrt(variance) * norm.ppf(q=1 - forecast_confidences_arr) + prediction) / 2)  # type: ignore

			current_log_realised_variances = (prediction, (current_log_realised_variances[1] * 4 + prediction) / 5, (current_log_realised_variances[2] * 21 + prediction) / 22)

		self.forecast_dates = get_forecast_dates(
			self.ticker,
			self.time,
			forecast_period,
		)
		volatility_forecast.index = to_datetime(self.forecast_dates)

		self.volatility_forecast = volatility_forecast

		return self.volatility_forecast
