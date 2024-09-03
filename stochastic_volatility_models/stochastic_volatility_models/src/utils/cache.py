from __future__ import annotations
from functools import wraps
from pandas import DataFrame
import numpy as np
from copy import deepcopy
from collections import OrderedDict


class LRUCache:
	def __init__(
		self,
		maxsize: int = 128,
	) -> None:
		self.maxsize = maxsize
		self.cache = OrderedDict()

	def get(
		self,
		key: tuple,
	):
		if key in self.cache:
			self.cache.move_to_end(key)
			return self.cache[key]
		return None

	def set(
		self,
		key: tuple,
		value,
	) -> None:
		if key in self.cache:
			self.cache.move_to_end(key)
		elif len(self.cache) >= self.maxsize:
			self.cache.popitem(last=False)
		self.cache[key] = value


def np_cache(
	*,
	arg_num: int = 0,
	arg_name: str = "symbols",
):
	cache = {}

	def decorator(function):
		@wraps(wrapped=function)
		def wrapper(
			*args,
			**kwargs,
		):
			args = list(args)
			indexing_arguments = kwargs.pop(arg_name, None)
			if indexing_arguments is None:
				# Assuming target_arg is the first positional argument
				if len(args) > 0:
					indexing_arguments = args.pop(arg_num)
					kwarg = False
				else:
					raise TypeError("Not enough arguments")
			else:
				kwarg = True

			keys = [(indexing_argument, *tuple(args), *kwargs.items()) for indexing_argument in indexing_arguments]
			evaluate_indexing_arguments = np.array([key[0] for key in keys if key not in cache])

			eval_args = deepcopy(args)
			eval_kwargs = deepcopy(kwargs)
			if kwarg:
				eval_kwargs[arg_name] = evaluate_indexing_arguments
			else:
				eval_args.insert(arg_num, evaluate_indexing_arguments)

			if len(evaluate_indexing_arguments) > 0:
				new_values = function(
					*eval_args,
					**eval_kwargs,
				)

				for indexing_argument, new_value in zip(evaluate_indexing_arguments, new_values):
					cache[(indexing_argument, *tuple(args), *kwargs.items())] = new_value

			result = np.array([cache[(indexing_argument, *tuple(args), *kwargs.items())] for indexing_argument in indexing_arguments])

			return result

		return wrapper

	return decorator


def np_multiple_cache():
	cache = {}

	def decorator(function):
		@wraps(wrapped=function)
		def wrapper(
			*args,
			**kwargs,
		):
			args_indices = tuple(i for i, arg in enumerate(args) if isinstance(arg, np.ndarray))
			kwargs_keys = tuple(k for k, v in kwargs.items() if isinstance(v, np.ndarray))
			num_elements = args[args_indices[0]].size if args_indices else kwargs[kwargs_keys[0]].size

			keys = [tuple([args_indices, kwargs_keys]) + tuple(arg[idx] if i in args_indices else arg for i, arg in enumerate(args)) + tuple(v[idx] if k in kwargs_keys else v for k, v in kwargs.items()) for idx in range(num_elements)]

			evaluate_keys = np.array([key not in cache for key in keys])
			eval_args = [arg if i not in args_indices else arg[evaluate_keys] for i, arg in enumerate(args)]
			eval_kwargs = {k: (kwarg if k not in kwargs_keys else kwarg[evaluate_keys]) for k, kwarg in kwargs.items()}

			if evaluate_keys.sum() > 0:
				new_values = function(
					*eval_args,
					**eval_kwargs,
				)

				for key, new_value in zip([key for key, evaluate in zip(keys, evaluate_keys) if evaluate], new_values):
					cache[key] = new_value

			result = np.array([cache[key] for key in keys])

			return result

		return wrapper

	return decorator


def df_cache(
	*,
	arg_num: int = 0,
	arg_name: str = "symbols",
):
	columns = []
	cache = {}

	def decorator(function):
		@wraps(wrapped=function)
		def wrapper(
			*args,
			**kwargs,
		) -> DataFrame:
			args = list(args)
			indexing_arguments = kwargs.pop(arg_name, None)
			if indexing_arguments is None:
				# Assuming target_arg is the first positional argument
				if len(args) > 0:
					indexing_arguments = args.pop(arg_num)
					kwarg = False
				else:
					raise TypeError("Not enough arguments")
			else:
				kwarg = True

			keys = [(indexing_argument, *tuple(args), *kwargs.items()) for indexing_argument in indexing_arguments]
			evaluate_indexing_arguments = np.array([key[0] for key in keys if key not in cache])

			eval_args = deepcopy(args)
			eval_kwargs = deepcopy(kwargs)
			if kwarg:
				eval_kwargs[arg_name] = evaluate_indexing_arguments
			else:
				eval_args.insert(1, evaluate_indexing_arguments)

			if len(evaluate_indexing_arguments) > 0:
				new_df = function(
					*eval_args,
					**eval_kwargs,
				)

				for indexing_argument in new_df.index:
					cache[(indexing_argument, *tuple(args), *kwargs.items())] = tuple(new_df.loc[indexing_argument])

				if columns == []:
					for column in new_df.columns:
						columns.append(column)

			result = DataFrame(
				data=[cache[(indexing_argument, *tuple(args), *kwargs.items())] for indexing_argument in indexing_arguments],
				index=[indexing_argument for indexing_argument in indexing_arguments],
				columns=columns,
			)

			return result

		return wrapper

	return decorator


def make_model_cache_key(
	self,
	*args,
	**kwargs,
) -> tuple:
	parameters_key = tuple(sorted(self.parameters.items()))
	return (parameters_key, args, tuple(sorted(kwargs.items())))


def model_lru_cache(maxsize=128):
	def decorator(func):
		cache = LRUCache(maxsize=maxsize)

		@wraps(func)
		def wrapper(self, *args, **kwargs):
			cache_key = make_model_cache_key(self, *args, **kwargs)

			cached_result = cache.get(key=cache_key)
			if cached_result is not None:
				return cached_result
			result = func(self, *args, **kwargs)
			cache.set(key=cache_key, value=result)
			return result

		return wrapper

	return decorator
