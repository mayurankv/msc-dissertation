from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray


Metrics = Literal["MSE", "RMSE", "MAE"]


def mse(
	quantities_1: NDArray,
	quantities_2: NDArray,
) -> float:
	return ((quantities_1 - quantities_2) ** 2).mean()


def rmse(
	quantities_1: NDArray,
	quantities_2: NDArray,
) -> float:
	return np.sqrt(
		mse(
			quantities_1=quantities_1,
			quantities_2=quantities_2,
		)
	)


def mae(
	quantities_1: NDArray,
	quantities_2: NDArray,
) -> float:
	return np.abs(quantities_1 - quantities_2).mean()


METRICS: dict[Metrics, Callable[[NDArray, NDArray], float]] = {
	"MSE": mse,
	"RMSE": rmse,
	"MAE": mae,
}
