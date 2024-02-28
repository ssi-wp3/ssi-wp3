from typing import List
import plotly.express as px
import pandas as pd


def sunburst_coicop_levels(dataframe: pd.DataFrame,
                           sunburst_columns: List[str],
                           amount_column: str,
                           filename: str):
    fig = px.sunburst(dataframe, path=sunburst_columns, values=amount_column)
    fig.write_html(filename)
