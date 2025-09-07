import os
import tempfile

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def make_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def test_scatter_plot_endpoint_json_safe():
    df = pd.DataFrame(
        {
            "Col_1": [1.0, 2.0, float("nan"), 4.0],
            "Col_3": [float("nan"), 1.5, 2.5, 5.0],
        }
    )
    path = make_temp_csv(df)
    try:
        resp = client.get(
            "/data_science/scatter_plot",
            params={"csv_file": path, "feature1": "Col_1", "feature2": "Col_3"},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert set(payload.keys()) == {"Col_1", "Col_3"}
        assert any(v is None for v in payload["Col_1"])  # NaN -> None
        assert any(v is None for v in payload["Col_3"])  # NaN -> None
    finally:
        os.unlink(path)


def test_line_plot_endpoint_no_numeric_columns_422():
    df = pd.DataFrame({"A": ["x", "y", "z"], "B": ["m", "n", "o"]})
    path = make_temp_csv(df)
    try:
        resp = client.get("/data_science/line_plot", params={"csv_file": path})
        assert resp.status_code == 422
        assert "No numeric columns" in resp.json()["detail"]
    finally:
        os.unlink(path)
