# Imports
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Constants
PARENTS = Path(__file__).resolve().parents
THESIS_DIRECTORY = PARENTS[4]
PROJECT_DIRECTORY = PARENTS[2]
MODULE_DIRECTORY = PARENTS[0]
MODULE_NAME = PARENTS[0].name


# Initialisation
def initialise() -> None:
	load_dotenv()

	try:
		from tqdm import tqdm

		logger.remove(0)
		logger.add(
			lambda msg: tqdm.write(msg, end=""),
			colorize=True,
		)
	except ModuleNotFoundError:
		pass

	logger.info(f"Loaded module `{MODULE_NAME}` from project directory path `{PROJECT_DIRECTORY}`")


# Guarded Script
if __name__ == "__main__":
	raise RuntimeError("Config should not be run directly")
