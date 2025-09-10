import asyncio
import os
import tempfile

import pandas as pd
import pytest

from src.service.datascience.plot_service import PlotService
from fastapi import HTTPException


def make_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def test_scatter_returns_json_safe_values():
    df = pd.DataFrame(
        {
            "Col_1": [1.0, 2.0, float("nan"), 4.0, float("inf")],
            "Col_3": [float("nan"), 1.5, 2.5, float("-inf"), 5.0],
            "Cat": ["a", "b", "c", "d", "e"],
        }
    )
    csv_path = make_temp_csv(df)
    try:

        async def run():
            svc = PlotService(csv_path)
            data = await svc.get_scatter_plot_data_service("Col_1", "Col_3")
            assert set(data.keys()) == {"Col_1", "Col_3"}
            # Ensure NaN/Inf became None
            assert any(v is None for v in data["Col_1"])  # NaN/Inf -> None
            assert any(v is None for v in data["Col_3"])  # NaN/Inf -> None
            # No remaining float NaNs should remain
            assert not any(isinstance(v, float) and v != v for v in data["Col_1"])
            assert not any(isinstance(v, float) and v != v for v in data["Col_3"])

        asyncio.run(run())
    finally:
        os.unlink(csv_path)


def test_line_plot_json_safe_and_not_empty_with_numeric_cols():
    df = pd.DataFrame(
        {
            "Col_1": [1.0, 2.0, float("nan"), 4.0, float("inf")],
            "Col_2": [10, 20, 30, 40, 50],
            "Col_3": [float("nan"), 1.5, 2.5, float("-inf"), 5.0],
            "Cat": ["a", "b", "c", "d", "e"],
        }
    )
    csv_path = make_temp_csv(df)
    try:

        async def run():
            svc = PlotService(csv_path)
            data = await svc.get_line_plot_data_service()
            # Should only include numeric columns
            assert set(data.keys()) == {"Col_1", "Col_2", "Col_3"}
            # Lengths preserved
            assert len(data["Col_1"]) == len(df)
            # NaN/Inf -> None
            assert any(v is None for v in data["Col_1"]) and any(
                v is None for v in data["Col_3"]
            )
            # No raw NaN floats remain
            for col in ("Col_1", "Col_2", "Col_3"):
                assert not any(isinstance(v, float) and v != v for v in data[col])

        asyncio.run(run())
    finally:
        os.unlink(csv_path)


def test_line_plot_no_numeric_columns_raises_422():
    df = pd.DataFrame({"A": ["x", "y", "z"], "B": ["m", "n", "o"]})
    csv_path = make_temp_csv(df)
    try:

        async def run():
            svc = PlotService(csv_path)
            with pytest.raises(HTTPException) as exc:
                await svc.get_line_plot_data_service()
            assert exc.value.status_code == 422
            assert "No numeric columns" in str(exc.value.detail)

        asyncio.run(run())
    finally:
        os.unlink(csv_path)
