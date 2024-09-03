from pandas import DataFrame
import numpy as np
from typing import cast
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
import copy

from stochastic_volatility_models.src.core.pricing_models import PricingModel
from stochastic_volatility_models.src.core.underlying import Underlying
from stochastic_volatility_models.src.core.volatility_surface import JointVolatilitySurface
from stochastic_volatility_models.visualisations.volatility_surface import plot_volatility_surface
from stochastic_volatility_models.visualisations.forecasts import plot_forecast
from stochastic_volatility_models.src.utils.options.expiry import time_to_expiry, DAYS
from stochastic_volatility_models.src.utils.options.strikes import find_closest_strikes
from stochastic_volatility_models.src.core.forecasts import forecast_performance


HESTON_OPTIMAL_PARAMETERS = {
	"2013-07-11": {
		"SPX": {"initial_variance": 0.01417449066157833, "long_term_variance": 0.0976082312303378, "volatility_of_volatility": 0.8599397768293384, "mean_reversion_rate": 0.24175166704773687, "wiener_correlation": -0.4525348016611488},
		"VIX": {"initial_variance": 0.01417449066157833, "long_term_variance": 0.03548883512598244, "volatility_of_volatility": 0.8434979043416542, "mean_reversion_rate": 8.670975033964442, "wiener_correlation": 0.7103820237758731},
	},
	"2015-09-01": {
		"SPX": {"initial_variance": 0.0895791230480412, "long_term_variance": 0.021357008010883253, "volatility_of_volatility": 2.969506343913128, "mean_reversion_rate": 9.443471596806445, "wiener_correlation": -0.9128602034942876},
		"VIX": {"initial_variance": 0.0895791230480412, "long_term_variance": 0.020655909614676057, "volatility_of_volatility": 2.6699215610053937, "mean_reversion_rate": 7.572874088738013, "wiener_correlation": -0.7119178902681154},
	},
	"2017-11-30": {
		"SPX": {"initial_variance": 0.007215276601791483, "long_term_variance": 0.017100228562753395, "volatility_of_volatility": 1.645667164191087, "mean_reversion_rate": 7.777980093495778, "wiener_correlation": -0.9149983849009664},
		"VIX": {"initial_variance": 0.007215276601791483, "long_term_variance": 0.0363301387547702, "volatility_of_volatility": 0.6813592590835942, "mean_reversion_rate": 1.6177774153739044, "wiener_correlation": -0.2851318745267004},
	},
}


def get_expiry_slider(
	traces,
	expiries,
	t2x,
):
	steps = []
	for idx in range(len(expiries)):
		step = dict(
			method="update",
			args=[
				{"visible": [j // traces == idx for j in range(len(expiries) * traces)]},  # Group visibility by expiry
				{"title": f"T: {expiries[idx]} - t2x: {t2x[idx]}"},
			],
			label=str(expiries[idx]),
		)
		steps.append(step)

	sliders = [dict(active=0, currentvalue={"prefix": "Expiry: "}, pad={"t": 50}, steps=steps)]
	return sliders


def get_strike_slider(
	traces,
	strikes,
):
	steps = []
	for idx in range(len(strikes)):
		step = dict(
			method="update",
			args=[
				{"visible": [j // traces == idx for j in range(len(strikes) * traces)]},  # Group visibility by expiry
				{"title": f"K: {strikes[idx]}"},
			],
			label=str(strikes[idx]),
		)
		steps.append(step)

	sliders = [dict(active=0, currentvalue={"prefix": "Strike: "}, pad={"t": 50}, steps=steps)]
	return sliders


class Notebook:
	def __init__(
		self,
		model=None,
		to_fit=False,
		time="2022-03-03",
		use_optimal_spx_params: bool = True,
		use_optimal_vix_params: bool = True,
	) -> None:
		self.model = model
		if model is not None:
			if model.name == "Heston" and time in HESTON_OPTIMAL_PARAMETERS:
				if use_optimal_spx_params and HESTON_OPTIMAL_PARAMETERS[time]["SPX"] is not None:
					model.parameters = copy.deepcopy(HESTON_OPTIMAL_PARAMETERS[time]["SPX"])
				elif use_optimal_vix_params and HESTON_OPTIMAL_PARAMETERS[time]["VIX"] is not None:
					model.parameters = copy.deepcopy(HESTON_OPTIMAL_PARAMETERS[time]["VIX"])
			else:
				pass

		self.to_fit = to_fit and model is not None
		self.ticker = "SPX"
		self.volatility_ticker = "VIX"
		self.underlying = Underlying(self.ticker)
		self.volatility_underlying = Underlying(self.volatility_ticker)

		self.pricing_model = PricingModel("Black-76 EMM" if self.model is not None else "Black-Scholes-Merton")
		self.volatility_pricing_model = PricingModel("Black-76 EMM" if self.model is not None else "Black-Scholes")
		self.empirical_pricing_model = PricingModel()
		self.volatility_empirical_pricing_model = PricingModel("Black-Scholes")

		self.time = np.datetime64(time)
		expiries = np.array(
			{
				"2022-03-03": ["2022-03-04", "2022-03-09", "2022-03-11", "2022-03-18", "2022-03-23", "2022-03-25", "2022-03-30", "2022-03-31", "2022-04-01", "2022-04-08", "2022-04-14", "2022-04-22", "2022-04-29", "2022-05-20", "2022-05-31", "2022-06-17", "2022-06-30", "2022-07-15", "2022-07-29", "2022-08-31"],
				"2017-01-03": ["2017-01-20", "2017-02-17", "2017-03-17", "2017-04-21", "2017-06-16", "2017-09-15", "2017-12-15", "2018-01-19", "2018-06-15", "2018-12-21", "2019-12-20"],
				"2015-07-01": ["2015-07-17", "2015-08-21", "2015-09-18", "2015-10-16", "2015-12-19", "2016-01-15", "2016-03-18", "2016-06-17", "2016-12-16", "2017-06-16", "2017-12-15"],
				"2015-08-21": ["2015-09-18", "2015-10-16", "2015-11-20", "2015-12-19", "2016-01-15", "2016-03-18", "2016-06-17", "2016-12-16", "2017-01-20", "2017-06-16", "2017-12-15"],
				"2015-09-01": ["2015-09-18", "2015-10-16", "2015-11-20", "2015-12-19", "2016-01-15", "2016-03-18", "2016-06-17", "2016-12-16", "2017-01-20", "2017-06-16", "2017-12-15"],
				"2013-07-11": ["2013-07-20", "2013-08-17", "2013-09-21", "2013-10-19", "2013-12-21", "2014-01-18", "2014-03-22", "2014-06-21", "2014-12-20", "2015-01-17", "2015-12-19"],
				"2017-11-30": ["2017-12-15", "2018-01-19", "2018-02-16", "2018-03-16", "2018-06-15", "2018-09-21", "2018-12-21", "2019-01-18", "2019-06-21", "2019-12-20"],
			}[time],
			dtype=np.datetime64,
		)
		strikes = np.array(
			{
				"2022-03-03": [2200, 2400, 2600, 2800, 3000, 3200, 3400, 3500, 3600, 3700, 3800, 3850, 3900, 3950, 3975, 4000, 4025, 4040, 4050, 4060, 4070, 4075, 4080, 4090, 4100, 4110, 4120, 4125, 4130, 4140, 4150, 4160, 4170, 4175, 4180, 4190, 4200, 4210, 4220]
				+ [4225, 4230, 4240, 4250, 4260, 4270, 4275, 4280, 4290, 4300, 4310, 4320, 4325, 4330, 4340, 4350, 4360, 4370, 4375, 4380, 4390, 4400, 4410, 4420, 4425, 4430, 4440, 4450, 4460, 4470, 4475, 4480, 4490, 4500, 4510, 4525, 4550, 4600, 4650, 4700, 4800, 5000, 5200, 5400],
				"2017-01-03": [1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2350, 2400, 2450],
				"2015-07-01": [500, 600, 700, 800, 900, 1000, 1050, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700]
				+ [1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2350, 2400, 2500],
				"2015-08-21": [500, 600, 700, 800, 850, 900, 950, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575]
				+ [1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2350, 2400, 2500],
				"2015-09-01": [500, 600, 700, 800, 850, 900, 950, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400]
				+ [1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2350, 2400, 2500],
				"2013-07-11": [500, 600, 650, 700, 750, 800, 850, 900, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1900],
				"2017-11-30": [1100, 1200, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050]
				+ [2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2475, 2500, 2525, 2550, 2575, 2600, 2625, 2650, 2675, 2700, 2725, 2750, 2775, 2800, 2825, 2900, 3000],
			}[time]
		)
		volatility_expiries = np.array(
			{
				"2022-03-03": ["2022-03-09", "2022-03-23", "2022-03-30", "2022-04-06"],
				"2017-01-03": ["2017-01-04", "2017-01-11", "2017-01-18", "2017-01-25", "2017-02-01", "2017-02-15", "2017-03-22", "2017-04-19", "2017-05-17", "2017-06-21"],
				"2015-07-01": ["2015-07-22", "2015-08-19", "2015-09-16", "2015-10-21", "2015-11-18", "2015-12-16"],
				"2015-08-21": ["2015-09-16", "2015-10-21", "2015-11-18", "2015-12-16", "2016-01-20", "2016-02-17"],
				"2015-09-01": ["2015-09-16", "2015-10-21", "2015-11-18", "2015-12-16", "2016-01-20", "2016-02-17"],
				"2013-07-11": ["2013-07-17", "2013-08-21", "2013-09-18", "2013-10-16", "2013-11-20", "2013-12-18"],
				"2017-11-30": ["2017-12-06", "2017-12-13", "2017-12-20", "2017-12-27", "2018-01-03", "2018-01-17", "2018-02-14", "2018-03-21", "2018-04-18", "2018-05-16"],
			}[time],
			dtype=np.datetime64,
		)
		volatility_strikes = np.array(
			{
				"2022-03-03": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
				"2017-01-03": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45],
				"2015-07-01": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 55, 60, 65, 70],
				"2015-08-21": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 55, 60, 65, 70],
				"2015-09-01": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 55, 60, 65, 70],
				"2013-07-11": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 55, 60, 65, 70],
				"2017-11-30": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45],
			}[time]
		)
		self.joint_volatility_surface = JointVolatilitySurface(
			underlying=self.underlying,
			expiries=expiries,
			strikes=strikes,
			volatility_underlying=self.volatility_underlying,
			volatility_expiries=volatility_expiries,
			volatility_strikes=volatility_strikes,
			monthly=time in ["2017-01-03", "2015-07-01", "2015-09-01", "2013-07-11", "2017-11-30"],
		)

	def price(
		self,
	) -> tuple[DataFrame, DataFrame]:
		if self.model is not None:
			return self.joint_volatility_surface.model_price(time=self.time, model=self.model)
		else:
			return self.joint_volatility_surface.empirical_price(time=self.time)

	def plot_surfaces(
		self,
		out_the_money=True,
		call=None,
		volatility=False,
		price=False,
	) -> go.Figure:
		pricing_model = self.pricing_model if not volatility else self.volatility_pricing_model

		if price:
			if self.model is not None:
				fig = plot_volatility_surface(
					time=self.time,
					joint_volatility_surface=self.joint_volatility_surface,
					quantity_method="model_price",
					model=self.model,
					plot_parameters={"moneyness": False, "time_to_expiry": False, "log_moneyness": False, "mid_price": True},
					out_the_money=out_the_money,
					call=call,
					volatility=volatility,
				)
			else:
				fig = plot_volatility_surface(
					time=self.time,
					joint_volatility_surface=self.joint_volatility_surface,
					quantity_method="empirical_price",
					plot_parameters={"moneyness": False, "time_to_expiry": False, "log_moneyness": False, "mid_price": True},
					out_the_money=out_the_money,
					call=call,
					volatility=volatility,
				)
		else:
			if self.model is not None:
				fig = plot_volatility_surface(
					time=self.time,
					joint_volatility_surface=self.joint_volatility_surface,
					quantity_method="model_pricing_implied_volatility",
					pricing_model=pricing_model,
					model=self.model,
					plot_parameters={"moneyness": False, "time_to_expiry": False, "log_moneyness": False, "mid_price": True},
					out_the_money=out_the_money,
					call=call,
					volatility=volatility,
				)
			else:
				fig = plot_volatility_surface(
					time=self.time,
					joint_volatility_surface=self.joint_volatility_surface,
					quantity_method="empirical_pricing_implied_volatility",
					pricing_model=pricing_model,
					plot_parameters={"moneyness": False, "time_to_expiry": False, "log_moneyness": False, "mid_price": True},
					out_the_money=out_the_money,
					call=call,
					volatility=volatility,
				)

		return fig

	def plot_put_call_iv(
		self,
		plot_closeup=False,
		volatility=False,
	) -> go.Figure:
		pricing_model = self.pricing_model if not volatility else self.volatility_pricing_model

		if self.model is not None:
			surface_c = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=True,
				call=True,
				pricing_model=pricing_model,
				model=self.model,
			)[int(volatility)][0]
			surface_p = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=True,
				call=False,
				pricing_model=pricing_model,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface_c = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=True,
				call=True,
				pricing_model=pricing_model,
			)[int(volatility)][0]
			surface_p = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=True,
				call=False,
				pricing_model=pricing_model,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		t2x = time_to_expiry(self.time, expiries)
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
		)

		fig = go.Figure()

		for idx, expiry in enumerate(expiries):
			if not plot_closeup:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface_c.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface_p.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="red"),
						marker=dict(color="red"),
					)
				)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=indices,
					y=cast(DataFrame, surface_c.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
					name=f"T: {expiry}",
					mode="markers" if not plot_closeup else "lines+markers",
					marker=dict(color="purple"),
					line=dict(color="purple"),
				)
			)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=indices,
					y=cast(DataFrame, surface_p.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
					name=f"T: {expiry}",
					mode="markers" if not plot_closeup else "lines+markers",
					marker=dict(color="orange"),
					line=dict(color="orange"),
				)
			)
			fig.add_vline(
				x=spot,
				line_dash="dash",
				line_color="green",
			)

		fig.update_layout(
			sliders=get_expiry_slider(
				traces=4 - 2 * plot_closeup,
				expiries=expiries,
				t2x=t2x,
			),
			showlegend=False,
			title_text=f"T: {expiries[0]} - t2x: {t2x[0]}",
		)

		return fig

	def plot_iv(
		self,
		plot_closeup=False,
		out_the_money=True,
		call=None,
		volatility=False,
	) -> go.Figure:
		pricing_model = self.pricing_model if not volatility else self.volatility_pricing_model

		if self.model is not None:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				pricing_model=pricing_model,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				pricing_model=pricing_model,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		t2x = time_to_expiry(self.time, expiries)
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
		)

		fig = go.Figure()

		for idx, expiry in enumerate(expiries):
			cs = CubicSpline(
				x=indices,
				y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
				bc_type="natural",
			)

			x = np.linspace(indices.min(), indices.max(), 100)
			slope, intercept = np.polyfit(indices, cast(DataFrame, surface.xs(key=expiry, level=1)).loc[indices, "Symbol"].to_numpy(), 1)

			if not plot_closeup:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=indices,
					y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
					name=f"T: {expiry}",
					mode="lines+markers",
					line=dict(color="red"),
					marker=dict(color="red"),
				)
			)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=[spot],
					y=[cs(spot)],
					name=f"T: {expiry}",
					mode="markers",
					marker=dict(color="darkgreen"),
				)
			)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=x,
					y=slope * x + intercept,
					name=f"T: {expiry}",
					mode="lines",
					line=dict(color="orange"),
				)
			)
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=x,
					y=cs(x),
					name=f"T: {expiry}",
					mode="lines",
					line=dict(color="purple"),
				)
			)
			fig.add_vline(
				x=spot,
				line_dash="dash",
				line_color="green",
			)

		fig.update_layout(
			sliders=get_expiry_slider(
				traces=5 - plot_closeup,
				expiries=expiries,
				t2x=t2x,
			),
			showlegend=False,
			title_text=f"T: {expiries[0]} - t2x: {t2x[0]}",
		)

		return fig

	def plot_strike_iv(
		self,
		out_the_money=True,
		call=None,
		volatility=False,
	) -> go.Figure:
		pricing_model = self.pricing_model if not volatility else self.volatility_pricing_model

		if self.model is not None:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				pricing_model=pricing_model,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_pricing_implied_volatility",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				pricing_model=pricing_model,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
		)

		fig = go.Figure()

		for idx, strike in enumerate(strikes):
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=expiries,
					y=cast(DataFrame, surface.xs(key=strike, level=0)).loc[expiries, "Symbol"].values,  # type: ignore
					name=f"K: {strike}",
					mode="lines+markers",
					line=dict(color="blue" if strike not in indices else "purple"),
					marker=dict(color="blue" if strike not in indices else "purple"),
				)
			)

		fig.update_layout(
			sliders=get_strike_slider(
				traces=1,
				strikes=strikes,
			),
			showlegend=False,
			title_text=f"K: {strikes[0]}",
		)

		return fig

	def plot_price(
		self,
		plot_closeup=False,
		out_the_money=True,
		call=None,
		volatility=False,
	) -> go.Figure:
		if self.model is not None:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		t2x = time_to_expiry(self.time, expiries)
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
			n=8,
		)

		fig = go.Figure()

		for idx, expiry in enumerate(expiries):
			if not plot_closeup:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
			else:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=indices,
						y=cast(DataFrame, surface.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
			fig.add_vline(
				x=spot,
				line_dash="dash",
				line_color="green",
			)

		fig.update_layout(
			sliders=get_expiry_slider(
				traces=1,
				expiries=expiries,
				t2x=t2x,
			),
			showlegend=False,
			title_text=f"T: {expiries[0]} - t2x: {t2x[0]}",
		)

		return fig

	def plot_strike_price(
		self,
		out_the_money=True,
		call=None,
		volatility=False,
	) -> go.Figure:
		if self.model is not None:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=call,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
		)

		fig = go.Figure()

		for idx, strike in enumerate(strikes):
			fig.add_trace(
				trace=go.Scatter(
					visible=idx == 0,  # Only the first trace is visible initially
					x=expiries,
					y=cast(DataFrame, surface.xs(key=strike, level=0)).loc[expiries, "Symbol"].values,  # type: ignore
					name=f"K: {strike}",
					mode="lines+markers",
					line=dict(color="blue" if strike not in indices else "purple"),
					marker=dict(color="blue" if strike not in indices else "purple"),
				)
			)

		fig.update_layout(
			sliders=get_strike_slider(
				traces=1,
				strikes=strikes,
			),
			showlegend=False,
			title_text=f"K: {strikes[0]}",
		)

		return fig

	def plot_joint_price(
		self,
		plot_closeup=False,
		out_the_money=True,
		volatility=False,
	) -> go.Figure:
		if self.model is not None:
			surface_c = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=True,
				model=self.model,
			)[int(volatility)][0]
			surface_p = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="model_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=False,
				model=self.model,
			)[int(volatility)][0]
		else:
			surface_c = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=True,
			)[int(volatility)][0]
			surface_p = self.joint_volatility_surface.surface_quantities(
				time=self.time,
				quantity_method="empirical_price",
				price_types=["Mid"],
				out_the_money=out_the_money,
				call=False,
			)[int(volatility)][0]

		spot = self.joint_volatility_surface.underlying.price(time=self.time) if not volatility else self.joint_volatility_surface.volatility_underlying.price(time=self.time)
		expiries = self.joint_volatility_surface.expiries if not volatility else self.joint_volatility_surface.volatility_expiries
		strikes = self.joint_volatility_surface.strikes if not volatility else self.joint_volatility_surface.volatility_strikes
		t2x = time_to_expiry(self.time, expiries)
		indices = find_closest_strikes(
			strikes=strikes,
			spot=spot,
			n=8,
		)

		fig = go.Figure()

		for idx, expiry in enumerate(expiries):
			if not plot_closeup:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface_c.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=strikes,
						y=cast(DataFrame, surface_p.xs(key=expiry, level=1)).loc[strikes, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="red"),
						marker=dict(color="red"),
					)
				)
			else:
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=indices,
						y=cast(DataFrame, surface_c.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="blue"),
						marker=dict(color="blue"),
					)
				)
				fig.add_trace(
					trace=go.Scatter(
						visible=idx == 0,  # Only the first trace is visible initially
						x=indices,
						y=cast(DataFrame, surface_p.xs(key=expiry, level=1)).loc[indices, "Symbol"].values,
						name=f"T: {expiry}",
						mode="lines+markers",
						line=dict(color="red"),
						marker=dict(color="red"),
					)
				)
			fig.add_vline(
				x=spot,
				line_dash="dash",
				line_color="green",
			)

		fig.update_layout(
			sliders=get_expiry_slider(
				traces=2,
				expiries=expiries,
				t2x=t2x,
			),
			showlegend=False,
			title_text=f"T: {expiries[0]} - t2x: {t2x[0]}",
		)

		return fig

	def fit(
		self,
		skew_weight=0.0,
		vol_weight=0.0,
	) -> dict:
		if self.model is not None:
			if self.to_fit:
				parameters: dict = self.model.fit(
					joint_volatility_surface=self.joint_volatility_surface,
					time=self.time,
					empirical_pricing_model=self.empirical_pricing_model,
					model_pricing_model=self.pricing_model,
					volatility_empirical_pricing_model=self.volatility_empirical_pricing_model,
					volatility_model_pricing_model=self.volatility_pricing_model,
					weights={
						"volatility_index": vol_weight,
						"skew": skew_weight,
					},
				)

				return parameters
			else:
				cost = self.evaluate_fit(skew_weight=skew_weight, vol_weight=vol_weight)

				print(f"Cost: {cost}")

				return {}
		else:
			return {}

	def evaluate_fit(
		self,
		skew_weight=0.0,
		vol_weight=0.0,
	) -> float:
		if self.model is not None:
			cost: float = self.model.evaluate_fit(
				joint_volatility_surface=self.joint_volatility_surface,
				time=self.time,
				empirical_pricing_model=self.empirical_pricing_model,
				model_pricing_model=self.pricing_model,
				volatility_empirical_pricing_model=self.volatility_empirical_pricing_model,
				volatility_model_pricing_model=self.volatility_pricing_model,
				weights={
					"volatility_index": vol_weight,
					"skew": skew_weight,
				},
			)

			return cost
		return 0

	def paths(
		self,
		simulation_length=1,
		num_paths=2**14,
		steps_per_year=DAYS,
		seed=1,
	):
		if self.model is not None:
			price_process, variance_process = self.model.simulate_path(
				underlying=self.underlying,
				volatility_underlying=self.volatility_underlying,
				time=self.time,
				simulation_length=simulation_length,
				steps_per_year=steps_per_year,
				num_paths=num_paths,
				monthly=False,
				seed=seed,
			)
			return price_process, variance_process
		return None, None

	def plot_paths(
		self,
		simulation_length=1,
		num_paths=2**14,
		num_show: int | None = None,
		choice_seed: int | None = None,
		seed=1,
	) -> None:
		if self.model is not None:
			price_process, variance_process = self.model.simulate_path(
				underlying=self.underlying,
				volatility_underlying=self.volatility_underlying,
				time=self.time,
				simulation_length=simulation_length,
				num_paths=num_paths,
				monthly=False,
				seed=seed,
			)

			if num_show is None or num_show == num_paths:
				price = price_process
				variance = variance_process
			else:
				rng = np.random.default_rng(seed=choice_seed)
				selected_elements = rng.choice(num_paths, size=num_show, replace=False)
				price = price_process[selected_elements]
				variance = variance_process[selected_elements]

			DataFrame(price.T).plot(legend=False)
			plt.show()

			DataFrame(variance.T).plot(legend=False)
			plt.show()

	def plot_forecast(
		self,
		simulation_length: float = 1,
		num_paths: int = 2**14,
		steps_per_year: int = int(DAYS),
		historical_period: float = 1,
		forecast_confidences: list[float] = [0.95, 0.99],
		use_drift: bool = False,
		config: dict = {},
		seed=1,
	):
		if self.model is not None:
			self.volatility_forecast = self.model.forecast_volatility(
				underlying=self.underlying,
				volatility_underlying=self.volatility_underlying,
				time=self.time,
				simulation_length=simulation_length,
				steps_per_year=steps_per_year,
				num_paths=num_paths,
				monthly=False,
				forecast_confidences=forecast_confidences,
				use_drift=use_drift,
				seed=seed,
			)

			plot_forecast(
				underlying=self.underlying,
				volatility_forecast=self.volatility_forecast,
				historical_period=historical_period,
			).show(config=config)

			return forecast_performance(self.underlying, self.volatility_forecast)

	def check_variance(
		self,
		simulation_length=1,
		strike: int | float | None = None,
		t2x: float | None = None,
		power_rounds: int = 6,
		path_powers: int = 8,
	):
		spot = self.underlying.price(time=self.time)

		if self.model is None:
			return None, None
		if t2x is None:
			t2x = 0.24383561643835616
		if strike is None:
			strike = spot

		steps_per_year = 365

		num_rounds = 2**power_rounds
		diffs = np.zeros((path_powers + 1, num_rounds))

		for power in range(path_powers + 1):
			num_paths = 2**power
			for idx in range(num_rounds):
				price_process, variance_process = self.paths(
					simulation_length=simulation_length,
					num_paths=num_paths,
					steps_per_year=steps_per_year,
					seed=idx,
				)
				assert price_process is not None and variance_process is not None
				call_price = np.mean(
					np.maximum(
						# (price_process[:, int(time_to_expiry * steps_per_year)] - strike) * np.exp((dividend_yield - risk_free_rate) * time_to_expiry) * flag,
						(price_process[:, int(t2x * steps_per_year)] - strike) * 1,
						0,
					)
				)
				put_price = np.mean(
					np.maximum(
						# (price_process[:, int(time_to_expiry * steps_per_year)] - strike) * np.exp((dividend_yield - risk_free_rate) * time_to_expiry) * flag,
						(price_process[:, int(t2x * steps_per_year)] - strike) * -1,
						0,
					)
				)
				diffs[power, idx] = put_price - call_price + spot - strike

		mean = np.mean(diffs, axis=1)
		variance = np.var(diffs, axis=1)

		return mean, variance
