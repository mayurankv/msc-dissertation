import os
import pandas as pd
from pandas import DataFrame
from stochastic_volatility_models.config import MODULE_DIRECTORY
from stochastic_volatility_models.src.data.pull.wrds_pull_data import PLURAL

INDEX_KEYS = {
	"option": ["symbol", "date"],
	"future": ["expiration", "date", "amsettlement"],
	"security": ["date"],
}


def format_wrds_data(
	ticker: str,
	year: int,
	asset_type: str,
) -> DataFrame:
	path = f"{MODULE_DIRECTORY}/data/wrds/{PLURAL[asset_type]}/{ticker.lower()}/{year}/{asset_type}_prices_{ticker.lower()}_{year}.csv"
	option_prices = pd.read_csv(path).drop(columns="Unnamed: 0")
	option_prices = option_prices.set_index(keys=INDEX_KEYS[asset_type])
	if not option_prices.index.is_unique:
		raise IndexError("Index is not unique")

	return option_prices


def format_data(
	ticker: str,
	asset_type: str,
) -> None:
	prices = pd.concat(
		objs=[
			format_wrds_data(
				ticker=ticker,
				year=int(year),
				asset_type=asset_type,
			)
			for year in os.listdir(f"{MODULE_DIRECTORY}/data/wrds/{PLURAL[asset_type]}/{ticker.lower()}/")
		]
	).sort_index()

	if not prices.index.is_unique:
		raise IndexError("Index is not unique")

	prices.to_csv(f"{MODULE_DIRECTORY}/data/{PLURAL[asset_type]}/{ticker.lower()}.csv")


def main() -> None:
	ticker = input("Input Ticker: ")
	asset_type: str = input("Pull 'option', 'future', or 'security': ")

	format_data(
		ticker=ticker,
		asset_type=asset_type,
	)


if __name__ == "__main__":
	main()
