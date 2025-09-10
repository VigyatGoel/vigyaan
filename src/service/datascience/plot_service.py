from functools import wraps

import numpy as np
import pandas as pd
from fastapi import HTTPException

from ...datascience.plots.plots import Plot


def handle_exceptions(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except HTTPException as e:
            # Let HTTPException propagate with its intended status code
            raise e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return wrapper


class PlotService:
    def __init__(self, file_path: str):
        self.plot = Plot(file_path)

    # --- Helpers -----------------------------------------------------------------
    @staticmethod
    def _df_to_json_lists(
        df: pd.DataFrame, ndigits: int | None = None, reset_index: bool = False
    ) -> dict:
        if ndigits is not None:
            df = df.round(ndigits)
        if reset_index:
            df = df.reset_index(drop=True)

        # Ensure Python-native types and JSON-safe values
        df = df.astype(object)
        df = df.replace([np.inf, -np.inf], None)
        df = df.where(pd.notna(df), None)
        return df.to_dict(orient="list")

    @staticmethod
    def _series_to_json_list(series: pd.Series, ndigits: int | None = None) -> list:
        if ndigits is not None:
            series = series.round(ndigits)
        series = series.astype(object)
        series = series.replace([np.inf, -np.inf], None)
        series = series.where(pd.notna(series), None)
        return series.tolist()

    @handle_exceptions
    async def get_scatter_plot_data_service(self, feature1, feature2):
        scatter_plot_data = await self.plot.get_scatter_plot_data(feature1, feature2)
        return self._df_to_json_lists(scatter_plot_data, ndigits=2)

    @handle_exceptions
    async def get_histogram_plot_data_service(self):
        histogram_plot_data = await self.plot.get_histogram_plot_data()
        return self._df_to_json_lists(histogram_plot_data, ndigits=2)

    @handle_exceptions
    async def get_line_plot_data_service(self):
        line_plot_data = await self.plot.get_line_plot_data()
        # Empty numeric selection => provide clear 422 error
        if line_plot_data.empty:
            raise HTTPException(
                status_code=422,
                detail="No numeric columns available to render a line plot.",
            )
        return self._df_to_json_lists(line_plot_data, ndigits=2, reset_index=True)

    @handle_exceptions
    async def get_correlation_matrix_data_service(self):
        correlation_matrix_data = await self.plot.get_correlation_matrix_data()
        if correlation_matrix_data.empty:
            raise HTTPException(
                status_code=422,
                detail="No numeric columns available to calculate correlation.",
            )
        return self._df_to_json_lists(correlation_matrix_data, ndigits=4)

    @handle_exceptions
    async def get_box_plot_data_service(self, feature1):
        boxplot_data = await self.plot.get_box_plot_data(feature1)
        return {feature1: self._series_to_json_list(boxplot_data, ndigits=2)}

    @handle_exceptions
    async def get_pair_plot_data_service(self):
        pair_plot_data = await self.plot.get_pair_plot_data()
        return self._df_to_json_lists(pair_plot_data, ndigits=2)

    @handle_exceptions
    async def get_area_plot_data_service(self, feature1):
        area_plot_data = await self.plot.get_area_plot_data(feature1)
        return {feature1: self._series_to_json_list(area_plot_data, ndigits=2)}
