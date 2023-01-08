def pandas():
    """
    Monkey patch pandas to add render function. Similar to the tqdm.pandas() approach
    :return:
    """
    from pandas.core.frame import DataFrame
    def to_html_wrapper(self, perform_dispaly = True):
        from IPython.display import HTML, display
        html = self.to_html(escape=False, index=False)
        if perform_dispaly:
            display(HTML(html))
        else:
            return html

    DataFrame.render = to_html_wrapper

pandas()
