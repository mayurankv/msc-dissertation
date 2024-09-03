from __future__ import annotations
from typing import TYPE_CHECKING
from pandas import DataFrame, to_datetime
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure


if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying


def plot_forecast(
	underlying: Underlying,
	volatility_forecast: DataFrame,
	historical_period: float,
) -> Figure:
	realised_volatility_past = underlying.realised_volatility(
		dates=underlying.returns(
			time=np.datetime64(volatility_forecast.index.values[0]).astype(dtype="datetime64[D]"),
			period=historical_period,
		).index
	)
	realised_volatility_past.index = to_datetime(realised_volatility_past.index)

	realised_volatility_future = underlying.realised_volatility(
		dates=volatility_forecast.index.astype(str),
	)
	realised_volatility_future.index = volatility_forecast.index

	fig = go.Figure()
	fig.add_traces(
		[
			go.Scatter(
				x=realised_volatility_past.index,
				y=realised_volatility_past,
				mode="lines",
				name="Historical RV",
				line=dict(color="blue"),
			),
			go.Scatter(
				x=realised_volatility_future.index,
				y=realised_volatility_future,
				mode="lines",
				name="True RV",
				line=dict(color="red"),
			),
			go.Scatter(
				x=volatility_forecast.index,
				y=volatility_forecast[("Forecast", "Mean")],
				mode="lines",
				name="Forecast",
				line=dict(color="purple"),
			),
		]
	)
	for confidence in np.array(volatility_forecast["Upper"].columns, dtype=np.float64):
		fig.add_trace(
			go.Scatter(
				x=volatility_forecast.index.to_numpy(),
				y=volatility_forecast[("Lower", confidence)].to_numpy(),
				mode="lines",
				line=dict(width=0),
				fillcolor="rgba(128, 0, 128, 0.15)",
				fill="tonexty",
				showlegend=False,
				legendgroup="Prediction Interval",
				legendgrouptitle_text="Prediction Interval",
				name=f"{round(confidence * 2 - 1,ndigits=4) * 100}% Prediction Interval Lower Bound",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=volatility_forecast.index.to_numpy(),
				y=volatility_forecast[("Upper", confidence)].to_numpy(),
				mode="lines",
				line=dict(width=0),
				fillcolor="rgba(128, 0, 128, 0.15)",
				fill="tonexty",
				showlegend=False,
				legendgroup="Prediction Interval",
				legendgrouptitle_text="Prediction Interval",
				name=f"{round(confidence * 2 - 1,ndigits=4) * 100}% Prediction Interval Upper Bound",
			)
		)

	fig.update_layout(
		# title="Forecasted Realised Volatility with Prediction Intervals",
		xaxis_title="Time",
		yaxis_title="Realised Volatility",
		legend=dict(x=0, y=1),
		template="plotly_white",
		margin=dict(l=0, r=0, t=0, b=0),
	)

	return fig


def plot_forecast_comparison(
	underlying: Underlying,
	volatility_forecast_1: DataFrame,
	volatility_forecast_2: DataFrame,
	historical_period: float,
	name_1: str = "Forecast 1",
	name_2: str = "Forecast 2",
) -> Figure:
	realised_volatility_past = underlying.realised_volatility(
		dates=underlying.returns(
			time=np.datetime64(volatility_forecast_1.index.values[0]).astype(dtype="datetime64[D]"),
			period=historical_period,
		).index
	)
	realised_volatility_past.index = to_datetime(realised_volatility_past.index)

	realised_volatility_future = underlying.realised_volatility(
		dates=volatility_forecast_1.index.astype(str),
	)
	realised_volatility_future.index = volatility_forecast_1.index

	fig = go.Figure()
	fig.add_traces(
		[
			go.Scatter(
				x=realised_volatility_past.index,
				y=realised_volatility_past,
				mode="lines",
				name="Historical RV",
				line=dict(color="blue"),
			),
			go.Scatter(
				x=realised_volatility_future.index,
				y=realised_volatility_future,
				mode="lines",
				name="True RV",
				line=dict(color="red"),
			),
			go.Scatter(
				x=volatility_forecast_1.index,
				y=volatility_forecast_1[("Forecast", "Mean")],
				mode="lines",
				name=name_1,
				line=dict(color="#da00ff"),
			),
			go.Scatter(
				x=volatility_forecast_2.index,
				y=volatility_forecast_2[("Forecast", "Mean")],
				mode="lines",
				name=name_2,
				line=dict(color="#007cff"),
			),
		]
	)

	fig.update_layout(
		# title="Forecasted Realised Volatility with Prediction Intervals",
		xaxis_title="Time",
		yaxis_title="Realised Volatility",
		legend=dict(x=0, y=1),
		template="plotly_white",
		margin=dict(l=0, r=0, t=0, b=0),
	)

	return fig
