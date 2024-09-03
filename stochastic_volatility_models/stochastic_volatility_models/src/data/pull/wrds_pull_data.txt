import os
import json
import wrds
from pandas import DataFrame, Index
from stochastic_volatility_models.config import MODULE_DIRECTORY

SOURCE = {
	"option": "opprcd",
	"future": "fwdprd",
	"security": "secprd",
}
PLURAL = {
	"option": "options",
	"future": "futures",
	"security": "securities",
}


def pull_data(
	ticker: str,
	year: int,
	asset_type: str,
) -> None:
	ticker_path = f"{MODULE_DIRECTORY}/data/wrds/{PLURAL[asset_type]}/{ticker.lower()}"
	path = f"{ticker_path}/{year}"
	if not os.path.exists(ticker_path):
		os.mkdir(ticker_path)
	if not os.path.exists(path):
		os.mkdir(path)

	csv_path = f"{path}/{asset_type}_prices_{ticker.lower()}_{year}.csv"
	json_path = f"{path}/{asset_type}_prices_{ticker.lower()}_{year}_metadata.json"
	if os.path.exists(csv_path):
		print("Data already pulled. Pull again?")
		if input("Continue (Y/N): ") != "Y":
			return

	with wrds.Connection(wrds_username="mayurankv") as db:
		query = f"""
		SELECT secid FROM optionm_all.secnmd WHERE ticker = '{ticker}'
		"""

		secids = db.raw_sql(sql=query)
		assert isinstance(secids, DataFrame)
		secid = int(secids["secid"].unique()[0])

		query = f"""
		SELECT *
		FROM optionm_all.{SOURCE[asset_type]}{year}
		WHERE  secid = '{secid}'
		"""

		prices = db.raw_sql(sql=query)

	assert isinstance(prices, DataFrame)
	columns = prices.columns
	unique_columns = prices.loc[:, (prices != prices.iloc[0]).any()].columns
	metadata = {
		"ticker": ticker,
		"year": year,
	}
	for column in columns.difference(unique_columns):
		metadata[column] = prices.iloc[0][column]

	if asset_type == "option":
		unique_columns = Index([*unique_columns, "am_settlement"])

	prices = prices[unique_columns]
	prices.to_csv(csv_path)
	with open(
		file=json_path,
		mode="w",
	) as file:
		json.dump(metadata, fp=file)


def main() -> None:
	ticker = input("Input Ticker: ")
	year = int(input("Input Year: "))
	asset_type: str = input("Pull 'option', 'future', or 'security': ")
	if asset_type not in SOURCE.keys():
		raise KeyError(f"Invalid type '{asset_type}' given")

	pull_data(
		ticker=ticker,
		year=year,
		asset_type=asset_type,
	)


if __name__ == "__main__":
	main()
