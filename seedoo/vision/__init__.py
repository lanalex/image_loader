from typing import Any, Dict, Optional, Tuple, Type
def pandas():
    """
    Monkey patch pandas to add render function. Similar to the tqdm.pandas() approach
    :return:
    """
    from pandas.core.frame import DataFrame
    from seedoo.io.pandas.lazy import LazyDataFrame

    def to_html_wrapper(self, perform_dispaly = True):
        from IPython.display import HTML, display
        html = self.to_html(escape=False)
        if perform_dispaly:
            display(HTML(html))
        else:
            return html

    DataFrame.render = to_html_wrapper

    # Save the original constructor
    _original_dataframe_constructor = DataFrame.__new__

    def _lazy_dataframe_constructor(cls: Type[pd.DataFrame], *args: Tuple, **kwargs: Dict[str, Any]) -> LazyDataFrame:
        """
        Custom constructor for DataFrame, which creates a LazyDataFrame wrapper.

        Args:
            cls (Type[pd.DataFrame]): The pandas DataFrame class.
            *args: Positional arguments for the DataFrame constructor.
            **kwargs: Keyword arguments for the DataFrame constructor.

        Returns:
            LazyDataFrame: A LazyDataFrame wrapper with the inner DataFrame created by the original constructor.
        """
        inner_df = _original_dataframe_constructor(cls, *args, **kwargs)
        return LazyDataFrame(inner_df)

    # Monkey-patch the DataFrame constructor
    pd.DataFrame.__new__ = _lazy_dataframe_constructor



pandas()
