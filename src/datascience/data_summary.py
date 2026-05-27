import asyncio
import os
from io import StringIO

import pandas as pd


class DataSummary:
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
    async def execute_parallel(func, df):
        return await asyncio.to_thread(func, df)

    async def get_file_info(self):
        return os.path.basename(self.file_path), round(
            os.path.getsize(self.file_path) / (1024 * 1024), 4
        )

    async def get_data_description(self):
        df = await self.get_df()
        return df.describe()

    async def get_data_info(self):
        df = await self.get_df()
        buffer = StringIO()
        await asyncio.to_thread(df.info, buf=buffer)
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def _get_data_types(df):
        return df.dtypes.astype(str)

    async def get_data_types(self):
        return await self.execute_parallel(self._get_data_types, await self.get_df())

    @staticmethod
    def _get_categorical_columns_count(df):
        return (
            df.select_dtypes(include=["object", "category"])
            .apply(pd.Series.value_counts)
            .fillna("none")
        )

    async def get_categorical_columns_count(self):
        return await self.execute_parallel(
            self._get_categorical_columns_count, await self.get_df()
        )

    async def get_row_col_count(self):
        df = await self.get_df()
        return await asyncio.to_thread(lambda: df.shape)

    async def get_null_val_count(self):
        df = await self.get_df()
        missing_values = df.isnull().sum()
        return missing_values, (missing_values / len(df)) * 100

    async def get_all_stats(self):
        return await asyncio.gather(
            self.get_file_info(),
            self.get_row_col_count(),
            self.get_null_val_count(),
            self.get_data_description(),
            self.get_data_info(),
            self.get_data_types(),
            self.get_categorical_columns_count(),
        )
