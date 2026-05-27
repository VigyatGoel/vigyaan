import asyncio
import os

import pandas as pd


class Plot:
    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.file_path = file_path
        self._df = None

    async def load_data(self, force_reload=False):
        if self._df is None or force_reload:
            self._df = await asyncio.to_thread(pd.read_csv, self.file_path)

    async def get_df(self) -> pd.DataFrame:
        await self.load_data()
        return self._df

    @staticmethod
    async def execute_parallel(func, df, *args):
        return await asyncio.to_thread(func, df, *args)

    @staticmethod
    def _scatter_plot(df, feature1, feature2):
        return df[[feature1, feature2]]

    async def get_scatter_plot_data(self, feature1, feature2):
        return await self.execute_parallel(
            self._scatter_plot, await self.get_df(), feature1, feature2
        )

    @staticmethod
    def _histogram_plot(df):
        return df.select_dtypes(include=["number"])

    async def get_histogram_plot_data(self):
        return await self.execute_parallel(self._histogram_plot, await self.get_df())

    @staticmethod
    def _line_plot(df):
        return df.select_dtypes(include=["number"])

    async def get_line_plot_data(self):
        return await self.execute_parallel(self._line_plot, await self.get_df())

    @staticmethod
    def _correlation_matrix(df):
        return df.select_dtypes(include=["number"]).corr()

    async def get_correlation_matrix_data(self):
        return await self.execute_parallel(
            self._correlation_matrix, await self.get_df()
        )

    @staticmethod
    def _single_box_plot(df, feature1):
        return df[feature1]

    async def get_box_plot_data(self, feature1):
        return await self.execute_parallel(
            self._single_box_plot, await self.get_df(), feature1
        )

    @staticmethod
    def _pair_plot(df):
        return df.select_dtypes(include=["number"])

    async def get_pair_plot_data(self):
        return await self.execute_parallel(self._pair_plot, await self.get_df())

    @staticmethod
    def _area_plot(df, feature1):
        return df[feature1]

    async def get_area_plot_data(self, feature1):
        return await self.execute_parallel(
            self._area_plot, await self.get_df(), feature1
        )
