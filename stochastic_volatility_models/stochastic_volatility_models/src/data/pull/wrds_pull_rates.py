import os
import wrds
from pandas import DataFrame
from stochastic_volatility_models.config import MODULE_DIRECTORY


def pull_rates(
	year: int,
	set_rates: bool = True,
) -> None:
	csv_path = f"{MODULE_DIRECTORY}/data/wrds/rates/rates_post_{year}.csv"
	if os.path.exists(csv_path):
		print("Data already pulled. Pull again?")
		if input("Continue (Y/N): ") != "Y":
			print("Rates not set")
			return

	with wrds.Connection(wrds_username="mayurankv") as db:
		query = f"""
		SELECT *
		FROM optionm_all.zerocd
		WHERE date >= '{year}-01-01'
		"""

		rates = db.raw_sql(sql=query)

	assert isinstance(rates, DataFrame)
	rates["date"] = rates["date"].astype(str)
	rates = rates.set_index(["date", "days"])
	rates.to_csv(csv_path)
	if set_rates:
		rates.to_csv(f"{MODULE_DIRECTORY}/data/rates.csv")


def main() -> None:
	year = int(input("Input starting year: "))
	set_rates = input("Set rates for use? (Y/N): ") == "Y"

	pull_rates(
		year=year,
		set_rates=set_rates,
	)


if __name__ == "__main__":
	main()
