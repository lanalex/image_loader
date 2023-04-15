from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import pickle
import tempfile

class LazyDataFrame:
    """
    A DataFrame-like class that lazily evaluates complex columns when accessed.
    """

    def __init__(self, df: Union[pd.DataFrame, Any], cache_lazy: bool = False, in_memory: bool = False, **kwargs: Any):
        """
        Initializes a LazyDataFrame.

        Args:
            df (Union[pd.DataFrame, Any]): The input DataFrame or any object that can be converted to a DataFrame.
            cache_lazy (bool, optional): Flag to cache the results of complex column evaluations. Defaults to False.
            in_memory (bool, optional): Flag to store the cache in memory or in a temporary file. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the pandas.DataFrame constructor.
        """
        self.df = pd.DataFrame(df, **kwargs)
        self.cache_lazy = cache_lazy
        self.in_memory = in_memory
        self.temp_files: Dict[Tuple[str, int, str], Union[str, Any]] = {}
        self.complex_columns: Dict[str, bool] = self.infer_complex_columns()

    def infer_complex_columns(self) -> Dict[str, bool]:
        """
        Infers complex columns based on the presence of callables in the first row.

        Returns:
            Dict[str, bool]: A dictionary with column names as keys and True as values for complex columns.
        """
        complex_columns = {}
        for col in self.df.columns:
            if callable(self.df[col].iloc[0]):
                complex_columns[col] = True
        return complex_columns

    def __getitem__(self, key: Union[str, List[str]]) -> pd.DataFrame:
        """
        Retrieves one or multiple columns from the LazyDataFrame.

        Args:
            key (Union[str, List[str]]): The column name or a list of column names to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the requested columns.
        """
        if isinstance(key, list):
            keys = key
        else:
            keys = [key]

        results = []
        for key in keys:
            if key in self.complex_columns:
                result = []
                indexes = []
                for idx in range(len(self.df)):
                    func = self.df.loc[idx, key]
                    indexes.append(idx)
                    value = self.lazy_eval(func, self.df.loc[idx], idx, key)
                    result.append(value)
                results.append(pd.Series(name=key, data=result, index=indexes))
            else:
                results.append(self.df[key])

        return pd.DataFrame(results).T

    def lazy_eval(self, func: Callable, row: pd.Series, index: int, key: str) -> Any:
        """
        Lazily evaluates the function for the given row.

        Args:
            func (Callable): The function to be evaluated.
            row (pd.Series): The row of the DataFrame.
            index (int): The index of the row.
            key (str): The name of the complex column.

        Returns:
            Any: The result of the function evaluation.
        """
        cache_key = (func.__name__, index, key)
        if self.cache_lazy and cache_key in self.temp_files:
            if self.in_memory:
                return self.temp_files[cache_key]
            else:
                with open(self.temp_files[cache_key], 'rb') as f:
                    return pickle.load(f)

        result = func(row)

        if self.cache_lazy:
            if self.in_memory:
                self.temp_files[cache_key] = result
            else:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    pickle.dump(result, f)
                    self.temp_files[cache_key] = f.name

            return result

    def __getattr__(self, attr: str) -> Any:
        """
        Gets attributes from the underlying DataFrame if not found in the LazyDataFrame.

        Args:
            attr (str): The attribute name.

        Returns:
            Any: The value of the attribute.
        """
        return getattr(self.df, attr)
