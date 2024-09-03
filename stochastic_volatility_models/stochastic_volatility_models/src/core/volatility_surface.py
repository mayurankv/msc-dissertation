from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict, Literal, Optional, cast
import numpy as np
from pandas import DataFrame, MultiIndex
from numpy.typing import NDArray

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying
	from stochastic_volatility_models.src.core.pricing_models import PricingModel
	from stochastic_volatility_models.src.core.model import StochasticVolatilityModel
from stochastic_volatility_models.src.data.prices import get_option_prices
from stochastic_volatility_models.src.utils.options.parameters import get_option_symbol

PriceTypes = Literal["Bid", "Ask", "Mid"]
OptionTypes = Literal["C", "P"]
QuantityMethod = Literal["empirical_price", "empirical_pricing_implied_volatility", "model_price", "model_pricing_implied_volatility"]


class OptionParameters(TypedDict):
	type: OptionTypes
	strike: int
	expiry: np.datetime64
	monthly: bool


class JointVolatilitySurface:
	def __init__(
		self,
		underlying: Underlying,
		expiries: NDArray[np.datetime64],
		strikes: NDArray[np.int64],
		volatility_underlying: Underlying,
		volatility_expiries: NDArray[np.datetime64],
		volatility_strikes: NDArray[np.int64],
		monthly: bool = True,
	) -> None:
		self.underlying = underlying
		self.expiries = np.sort(expiries)
		self.strikes = np.sort(strikes)
		self.volatility_underlying = volatility_underlying
		self.volatility_expiries = np.sort(volatility_expiries)
		self.volatility_strikes = np.sort(volatility_strikes)
		self.monthly = monthly
		self.options = DataFrame("", index=MultiIndex.from_product([self.strikes, self.expiries], names=["Strike", "Expiry"]), columns=["C", "P"])
		self.volatility_options = DataFrame("", index=MultiIndex.from_product([self.volatility_strikes, self.volatility_expiries], names=["Strike", "Expiry"]), columns=["C", "P"])
		self.options[["C", "P"]] = [
			[
				get_option_symbol(
					ticker=self.underlying.ticker,
					option_type="C",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
				get_option_symbol(
					ticker=self.underlying.ticker,
					option_type="P",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
			]
			for strike, expiry in self.options.index
		]
		self.volatility_options[["C", "P"]] = [
			[
				get_option_symbol(
					ticker=self.volatility_underlying.ticker,
					option_type="C",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
				get_option_symbol(
					ticker=self.volatility_underlying.ticker,
					option_type="P",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
			]
			for strike, expiry in self.volatility_options.index
		]

	def surface_symbols(
		self,
		time: np.datetime64,
		out_the_money: bool = True,
		call: Optional[bool] = None,
	) -> tuple[DataFrame, DataFrame]:
		spot = self.underlying.price(time=time)
		volatility_spot = self.volatility_underlying.price(time=time)

		def call_condition(
			index: tuple[np.int64, np.datetime64],
			spot: float,
		) -> bool:
			if call is not None:
				return call
			else:
				return bool((index[0] >= spot and out_the_money) or (index[0] < spot and not out_the_money))

		surface = DataFrame(
			data=[self.options.at[index, "C" if call_condition(index, spot) else "P"] for index in self.options.index],
			index=self.options.index,
			columns=["Symbol"],
		)
		volatility_surface = DataFrame(
			data=[self.volatility_options.at[index, "C" if call_condition(index, volatility_spot) else "P"] for index in self.volatility_options.index],
			index=self.volatility_options.index,
			columns=["Symbol"],
		)

		return surface, volatility_surface

	@lru_cache()
	def empirical_price(
		self,
		time: np.datetime64,
	) -> tuple[DataFrame, DataFrame]:
		empirical_prices = get_option_prices(
			ticker=self.underlying.ticker,
			time=time,
			symbols=self.options.values.ravel(),
		)
		empirical_volatility_prices = get_option_prices(
			ticker=self.volatility_underlying.ticker,
			time=time,
			symbols=self.volatility_options.values.ravel(),
		)

		return empirical_prices, empirical_volatility_prices

	def empirical_pricing_implied_volatility(
		self,
		time: np.datetime64,
		pricing_model: PricingModel,
	) -> tuple[DataFrame, DataFrame]:
		empirical_prices, empirical_volatility_prices = self.empirical_price(
			time=time,
		)
		empirical_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=empirical_prices,
			time=time,
			underlying=self.underlying,
			monthly=self.monthly,
		)
		empirical_volatility_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=empirical_volatility_prices,
			time=time,
			underlying=self.volatility_underlying,
			monthly=self.monthly,
		)

		return empirical_pricing_implied_volatilities, empirical_volatility_pricing_implied_volatilities

	def model_price(
		self,
		time: np.datetime64,
		model: StochasticVolatilityModel,
		*args,
		**kwargs,
	) -> tuple[DataFrame, DataFrame]:
		model_prices, model_volatility_prices = model.price_surface(
			time=time,
			underlying=self.underlying,
			volatility_underlying=self.volatility_underlying,
			symbols=tuple(self.options.values.ravel()),
			volatility_symbols=tuple(self.volatility_options.values.ravel()),
			monthly=self.monthly,
			*args,
			**kwargs,
		)

		return model_prices, model_volatility_prices

	def model_pricing_implied_volatility(
		self,
		time: np.datetime64,
		model: StochasticVolatilityModel,
		pricing_model: PricingModel,
		*args,
		**kwargs,
	) -> tuple[DataFrame, DataFrame]:
		model_prices, model_volatility_prices = self.model_price(
			time=time,
			model=model,
			*args,
			**kwargs,
		)
		model_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=model_prices,
			time=time,
			underlying=self.underlying,
			monthly=self.monthly,
		)
		model_volatility_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=model_volatility_prices,
			time=time,
			underlying=self.volatility_underlying,
			monthly=self.monthly,
		)

		return model_pricing_implied_volatilities, model_volatility_pricing_implied_volatilities

	def surface_quantities(
		self,
		time: np.datetime64,
		quantity_method: QuantityMethod,
		price_types: list[PriceTypes],
		out_the_money: bool = True,
		call: Optional[bool] = None,
		*args,
		**kwargs,
	) -> tuple[list[DataFrame], list[DataFrame]]:
		surface_symbols, volatility_surface_symbols = self.surface_symbols(time=time, out_the_money=out_the_money, call=call)
		quantities, volatility_quantities = cast(tuple[DataFrame, DataFrame], getattr(self, quantity_method)(time=time, *args, **kwargs))
		surfaces = [surface_symbols["Symbol"].map(quantities[price_type]).to_frame() for price_type in price_types]
		volatility_surfaces = [volatility_surface_symbols["Symbol"].map(volatility_quantities[price_type]).to_frame() for price_type in price_types]

		return surfaces, volatility_surfaces


class VolatilitySurface:
	def __init__(
		self,
		underlying: Underlying,
		expiries: NDArray[np.datetime64],
		strikes: NDArray[np.int64],
		monthly: bool = True,
	) -> None:
		self.underlying = underlying
		self.expiries = np.sort(expiries)
		self.strikes = np.sort(strikes)
		self.monthly = monthly
		self.options = DataFrame("", index=MultiIndex.from_product([self.strikes, self.expiries], names=["Strike", "Expiry"]), columns=["C", "P"])
		self.options[["C", "P"]] = [
			[
				get_option_symbol(
					ticker=self.underlying.ticker,
					option_type="C",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
				get_option_symbol(
					ticker=self.underlying.ticker,
					option_type="P",
					expiry=np.datetime64(expiry),
					strike=strike,
					monthly=self.monthly,
				),
			]
			for strike, expiry in self.options.index
		]

	def surface_symbols(
		self,
		time: np.datetime64,
		out_the_money: bool = True,
		call: Optional[bool] = None,
	) -> DataFrame:
		def call_condition(index: tuple[np.int64, np.datetime64]):
			if call is not None:
				return call
			else:
				return (index[0] >= self.underlying.price(time=time) and out_the_money) or (index[0] < self.underlying.price(time=time) and not out_the_money)

		surface = DataFrame(
			data=[self.options.at[index, "C" if call_condition(index) else "P"] for index in self.options.index],
			index=self.options.index,
			columns=["Symbol"],
		)

		return surface

	def empirical_price(
		self,
		time: np.datetime64,
	) -> DataFrame:
		empirical_prices = get_option_prices(
			ticker=self.underlying.ticker,
			time=time,
			symbols=self.options.values.ravel(),
		)

		return empirical_prices

	def empirical_pricing_implied_volatility(
		self,
		time: np.datetime64,
		pricing_model: PricingModel,
	) -> DataFrame:
		empirical_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=self.empirical_price(
				time=time,
			),
			time=time,
			underlying=self.underlying,
			monthly=self.monthly,
		)

		return empirical_pricing_implied_volatilities

	def model_price(
		self,
		time: np.datetime64,
		model: StochasticVolatilityModel,
		*args,
		**kwargs,
	) -> DataFrame:
		model_prices, _ = model.price_surface(
			time=time,
			underlying=self.underlying,
			symbols=tuple(self.options.values.ravel()),
			volatility_symbols=tuple(),  # TODO (@mayurankv):
			monthly=self.monthly,
			*args,
			**kwargs,
		)

		return model_prices

	def model_pricing_implied_volatility(
		self,
		model: StochasticVolatilityModel,
		pricing_model: PricingModel,
		time: np.datetime64,
		*args,
		**kwargs,
	) -> DataFrame:
		model_pricing_implied_volatilities = pricing_model.price_implied_volatility(
			prices=self.model_price(
				time=time,
				model=model,
				*args,
				**kwargs,
			),
			time=time,
			underlying=self.underlying,
			monthly=self.monthly,
		)

		return model_pricing_implied_volatilities

	def surface_quantities(
		self,
		time: np.datetime64,
		quantity_method: QuantityMethod,
		price_types: list[PriceTypes],
		out_the_money: bool = True,
		call: Optional[bool] = None,
		*args,
		**kwargs,
	) -> list[DataFrame]:
		surface_symbols = self.surface_symbols(time=time, out_the_money=out_the_money, call=call)
		quantities: DataFrame = getattr(self, quantity_method)(time=time, *args, **kwargs)
		surfaces = [surface_symbols["Symbol"].map(quantities[price_type]).to_frame() for price_type in price_types]

		return surfaces
