from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import numpy as np
from pandas import DataFrame
import warnings
from py_vollib_vectorized import vectorized_black_scholes as bs_price
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as bs_iv
from py_vollib_vectorized import vectorized_black as b_price
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility_black as iv
from py_vollib_vectorized import vectorized_black_scholes_merton as bsm_price

if TYPE_CHECKING:
	from stochastic_volatility_models.src.core.underlying import Underlying
from stochastic_volatility_models.src.data.option_implied_prices import get_option_future_price
from stochastic_volatility_models.src.data.rates import get_risk_free_interest_rate
from stochastic_volatility_models.src.data.dividends import get_dividend_yield
from stochastic_volatility_models.src.utils.options.expiry import time_to_expiry
from stochastic_volatility_models.src.utils.options.parameters import get_options_parameters_transpose


PricingModels = Literal["Black-Scholes-Merton", "Black-Scholes", "Black-76", "Black-76 EMM"]


class PricingModel:
	def __init__(
		self,
		model: PricingModels = "Black-Scholes-Merton",
		use_market_future: bool = False,
	) -> None:
		if model in ["Black-Scholes-Merton", "Black-Scholes", "Black-76", "Black-76 EMM"]:
			self.model = model
		else:
			raise ValueError("Pricing Model unknown")

		self.use_market_future = use_market_future

	def price_implied_volatility(
		self,
		underlying: Underlying,
		prices: DataFrame,
		time: np.datetime64,
		monthly: bool = True,
	) -> DataFrame:
		warnings.filterwarnings(
			action="ignore",
			category=UserWarning,
		)
		implied_volatilities = DataFrame(
			data=None,
			index=prices.index,
			columns=prices.columns,
		)
		options_parameters_transpose = get_options_parameters_transpose(
			ticker=underlying.ticker,
			symbols=tuple(implied_volatilities.index.values),
		)
		for column in implied_volatilities.columns:
			if self.model == "Black-Scholes-Merton":
				implied_volatilities[column] = bs_iv(
					S=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					price=prices[column].values,
					q=get_dividend_yield(
						underlying=underlying,
						time=time,
						strikes=np.array(options_parameters_transpose["strike"]),
						expiries=np.array(options_parameters_transpose["expiry"]),
						monthly=monthly,
					),
					return_as="numpy",
					model="black_scholes_merton",
				)
			elif self.model == "Black-Scholes":
				implied_volatilities[column] = bs_iv(
					S=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					price=prices[column].values,
					return_as="numpy",
					model="black_scholes",
				)
			elif self.model == "Black-76":
				implied_volatilities[column] = iv(
					F=underlying.future_price(time=time)
					if self.use_market_future
					else get_option_future_price(
						underlying=underlying,
						time=time,
						strikes=np.array(options_parameters_transpose["strike"]),
						expiries=np.array(options_parameters_transpose["expiry"]),
						monthly=monthly,
					),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					price=prices[column].values,  # TODO (@mayurankv): Discount?
					return_as="numpy",
				)
			elif self.model == "Black-76 EMM":
				implied_volatilities[column] = iv(
					F=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=0,
					price=prices[column].values,
					return_as="numpy",
				)

		implied_volatilities = implied_volatilities.fillna(value=0)

		return implied_volatilities

	def volatility_implied_price(
		self,
		underlying: Underlying,
		volatilities: DataFrame,
		time: np.datetime64,
		monthly: bool = True,
	) -> DataFrame:
		warnings.filterwarnings(
			action="ignore",
			category=UserWarning,
		)
		implied_prices = DataFrame(
			data=None,
			index=volatilities.index,
			columns=volatilities.columns,
		)
		options_parameters_transpose = get_options_parameters_transpose(
			ticker=underlying.ticker,
			symbols=tuple(implied_prices.index.values),
		)
		for column in implied_prices.columns:
			if self.model == "Black-Scholes-Merton":
				implied_prices[column] = bsm_price(
					S=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					q=get_dividend_yield(
						underlying=underlying,
						time=time,
						strikes=np.array(options_parameters_transpose["strike"]),
						expiries=np.array(options_parameters_transpose["expiry"]),
						monthly=monthly,
					),
					sigma=volatilities[column].values,
					return_as="numpy",
				)
			elif self.model == "Black-Scholes":
				implied_prices[column] = bs_price(
					S=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					sigma=volatilities[column].values,
					return_as="numpy",
				)
			elif self.model == "Black-76":
				implied_prices[column] = b_price(
					F=underlying.future_price(time=time)
					if self.use_market_future
					else get_option_future_price(
						underlying=underlying,
						time=time,
						strikes=np.array(options_parameters_transpose["strike"]),
						expiries=np.array(options_parameters_transpose["expiry"]),
						monthly=monthly,
					),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					sigma=volatilities[column].values,  # TODO (@mayurankv): Un-discount price?
					return_as="numpy",
				)
			elif self.model == "Black-76 EMM":
				implied_prices[column] = b_price(
					F=underlying.price(time=time),
					flag=np.char.lower(options_parameters_transpose["type"]),
					K=options_parameters_transpose["strike"],
					t=time_to_expiry(
						time=time,
						option_expiries=np.array(options_parameters_transpose["expiry"]),
					),
					r=get_risk_free_interest_rate(
						time=time,
						time_to_expiry=time_to_expiry(
							time=time,
							option_expiries=np.array(options_parameters_transpose["expiry"]),
						),
					),
					sigma=volatilities[column].values,  # TODO (@mayurankv): Un-discount price?
					return_as="numpy",
				)

		implied_prices = implied_prices.fillna(value=0)

		return implied_prices
