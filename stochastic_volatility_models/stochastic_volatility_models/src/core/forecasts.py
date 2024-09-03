from __future__ import annotations
from typing import TYPE_CHECKING
from pandas import DataFrame
import numpy as np

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying


def forecast_performance(
	underlying: Underlying,
	volatility_forecast: DataFrame,
) -> tuple[dict[str, float], dict[float, float]]:
	volatility_forecast = volatility_forecast.iloc[1:]
	realised_volatility = underlying.realised_volatility(
		dates=volatility_forecast.index.astype(str),
	)
	realised_volatility.index = volatility_forecast.index

	performance = {
		"MAE": np.mean(np.abs(volatility_forecast[("Forecast", "Mean")] - realised_volatility)),
		# "MAPE": np.mean(np.abs((volatility_forecast[("Forecast", "Mean")] - realised_volatility) / realised_volatility)),
		"RMSE": np.sqrt(np.mean(np.square(volatility_forecast[("Forecast", "Mean")] - realised_volatility))),
		# "RMSPE": np.sqrt(np.mean(np.square((volatility_forecast[("Forecast", "Mean")] - realised_volatility) / realised_volatility))),
	}

	prediction_interval_accuracies = {round(confidence * 2 - 1, ndigits=4): ((volatility_forecast["Lower", confidence] < realised_volatility) & (volatility_forecast["Upper", confidence] > realised_volatility)).sum() / volatility_forecast.index.shape[0] for confidence in volatility_forecast["Upper"].columns}

	return performance, prediction_interval_accuracies
