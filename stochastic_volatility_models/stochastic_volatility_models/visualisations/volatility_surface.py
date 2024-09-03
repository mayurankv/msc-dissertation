from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.volatility_surface import QuantityMethod, PriceTypes, JointVolatilitySurface
from stochastic_volatility_models.src.utils.options.expiry import time_to_expiry
from stochastic_volatility_models.src.utils.options.strikes import moneyness


class SurfacePlotParameters(TypedDict):
	mid_price: bool
	time_to_expiry: bool
	moneyness: bool
	log_moneyness: bool


DEFAULT_PLOT_PARAMETERS: SurfacePlotParameters = {
	"mid_price": True,
	"time_to_expiry": True,
	"moneyness": True,
	"log_moneyness": True,
}


# TODO (@mayurankv): Prettify plot
def plot_volatility_surface(
	time: np.datetime64,
	joint_volatility_surface: JointVolatilitySurface,
	quantity_method: QuantityMethod,
	plot_parameters: SurfacePlotParameters = DEFAULT_PLOT_PARAMETERS,
	volatility: bool = False,
	opacity: float = 1.0,
	**kwargs,
) -> Figure:
	plot_type = "Model" if "model" in kwargs else "Empirical"

	plot_parameters = DEFAULT_PLOT_PARAMETERS | plot_parameters
	underlying = joint_volatility_surface.underlying if not volatility else joint_volatility_surface.volatility_underlying
	expiries = joint_volatility_surface.expiries if not volatility else joint_volatility_surface.volatility_expiries
	strikes = joint_volatility_surface.strikes if not volatility else joint_volatility_surface.volatility_strikes

	x = expiries if not plot_parameters["time_to_expiry"] else time_to_expiry(time=time, option_expiries=expiries)
	y = strikes if not plot_parameters["moneyness"] else moneyness(underlying=underlying, strikes=strikes, time=time, log=plot_parameters["log_moneyness"])
	price_types: list[PriceTypes] = ["Bid", "Ask"] if not plot_parameters["mid_price"] else ["Mid"]
	moneyness_title = f"{'log-' if plot_parameters["log_moneyness"] else ''}Moneyness"
	fig = go.Figure(
		data=[
			go.Surface(
				x=x,
				y=y,
				z=surface_quantities.unstack(0).to_numpy().transpose(),  # TODO (@mayurankv): Label
				name=f"{plot_type} Implied Volatility surface",
				opacity=opacity,
			)
			for surface_quantities in joint_volatility_surface.surface_quantities(time=time, quantity_method=quantity_method, price_types=price_types, **kwargs)[int(volatility)]
		],
		layout=dict(
			# title="Volatility Surface",  # TODO (@mayurankv): Opacity and layout
			scene=dict(
				aspectmode="cube",
				xaxis=dict(title="Expiries" if not plot_parameters["time_to_expiry"] else "Time to Expiry (Yr)"),
				yaxis=dict(title="Strikes" if not plot_parameters["moneyness"] else moneyness_title),
				zaxis=dict(title=f"Implied {quantity_method.split(sep="_")[-1].title()}"),
			),
			margin=dict(l=0, r=0, t=0, b=0),
		),
	)  # .update_traces(
	# 	contours_z=dict(
	# 		show=True,
	# 		usecolormap=True,
	# 		highlightcolor="limegreen",
	# 		project_z=True,
	# 	),
	# )
	return fig


def plot_volatility_surface_comparison(
	model_fig: Figure,
	empirical_fig: Figure,
	opacity: float = 0.4,
) -> Figure:
	for trace in model_fig.data:
		if isinstance(trace, go.Surface):
			model_z = np.array(trace.z)
	for trace in empirical_fig.data:
		if isinstance(trace, go.Surface):
			empirical_z = np.array(trace.z)
			x = trace.x
			y = trace.y

	fig = (
		go.Figure(
			data=[
				go.Surface(
					x=x,
					y=y,
					z=model_z,
					opacity=opacity,
					name="Model",
					showlegend=True,
					colorscale=[[0, "orange"], [1, "red"]],
					colorbar=None,
					showscale=False,
				),
				go.Surface(
					x=x,
					y=y,
					z=empirical_z,
					opacity=opacity,
					name="Market",
					showlegend=True,
					colorscale=[[0, "purple"], [1, "blue"]],
					colorbar=None,
					showscale=False,
				),
			],
			layout=model_fig.layout,
		).update_coloraxes(showscale=False)
		# .update_layout(dict(title="Volatility Surface Comparison"))
	)

	return fig


def plot_volatility_surface_error_comparison(
	model_fig: Figure,
	empirical_fig: Figure,
	opacity: float = 1.0,
) -> Figure:
	model_z = np.array([])
	empirical_z = np.array([])

	for trace in model_fig.data:
		if isinstance(trace, go.Surface):
			model_z = np.array(trace.z)
	for trace in empirical_fig.data:
		if isinstance(trace, go.Surface):
			empirical_z = np.array(trace.z)
			x = trace.x
			y = trace.y

	fig = go.Figure(
		data=[
			go.Surface(
				x=x,
				y=y,
				z=100 * np.abs((empirical_z - model_z) / empirical_z),
				opacity=opacity,
				name="Percentage Error in Model Implied Volatility surface",
			),
		],
		layout=model_fig.layout,
	).update_layout(
		dict(
			# title="Volatility Surface Comparison",
			scene=dict(
				zaxis=dict(
					title="Absolute % Error in Volatility",
				)
			),
		)
	)

	return fig
