import datetime as dt
import pytest

import src.module_1.module_1_meteo_api as meteo
from src.module_1.module_1_meteo_api import main


@pytest.fixture(scope="module")
def fake_meteo_response():
    """Reusable valid response object for tests (fast + deterministic)."""
    daily = meteo.MeteoDaily(
        time=[dt.date(2010, 1, 1), dt.date(2010, 1, 2)],
        temperature_2m_mean=[10.0, 12.0],
        precipitation_sum=[1.0, None],
        wind_speed_10m_max=[5.0, 6.0],
    )
    return meteo.MeteoResponse(daily=daily)


def test_main(monkeypatch, capsys, fake_meteo_response):
    """Unit-test main() by isolating network, plotting, and sleeping."""

    # avoid waiting
    monkeypatch.setattr(meteo.time, "sleep", lambda *_: None)

    # capture plot inputs
    captured = {"precip": None, "temp": None}

    def fake_plot_precip(city_to_rows):
        captured["precip"] = city_to_rows

    def fake_plot_temp(city_to_rows):
        captured["temp"] = city_to_rows

    monkeypatch.setattr(
        meteo, "plot_annual_total_precipitation_by_city", fake_plot_precip
    )
    monkeypatch.setattr(meteo, "plot_annual_mean_temperature_by_city", fake_plot_temp)

    # avoid real API calls
    def fake_get_data(city, start_date, end_date):
        return fake_meteo_response

    monkeypatch.setattr(meteo, "get_data_meteo_api", fake_get_data)

    # run
    main()

    # verify it printed something per city
    out = capsys.readouterr().out
    for city in meteo.COORDINATES.keys():
        assert city in out

    # verify both plotting functions were called with data for all cities
    assert captured["precip"] is not None
    assert captured["temp"] is not None
    assert set(captured["precip"].keys()) == set(meteo.COORDINATES.keys())
    assert set(captured["temp"].keys()) == set(meteo.COORDINATES.keys())

    # verify normalized rows look correct
    for rows in captured["precip"].values():
        assert len(rows) == 2
        assert rows[0]["date"] == dt.date(2010, 1, 1)
        for v in meteo.VARIABLES:
            assert v in rows[0]


def test_normalize_daily_accepts_string_dates():
    daily = {
        "time": ["2010-01-01", "2010-01-02"],
        "temperature_2m_mean": [10.0, 11.0],
        "precipitation_sum": [1.0, 2.0],
        "wind_speed_10m_max": [5.0, 6.0],
    }
    rows = meteo.normalize_daily(daily, meteo.VARIABLES)
    assert rows[0]["date"] == dt.date(2010, 1, 1)
    assert rows[1]["temperature_2m_mean"] == 11.0


def test_annual_total_precipitation_ignores_none():
    rows = [
        {"date": dt.date(2010, 1, 1), "precipitation_sum": 1.0},
        {"date": dt.date(2010, 1, 2), "precipitation_sum": None},
        {"date": dt.date(2010, 1, 3), "precipitation_sum": 2.5},
    ]
    out = meteo.annual_total_precipitation(rows)
    assert out == {2010: 3.5}


def test_annual_mean_temperature_computes_mean_and_ignores_none():
    rows = [
        {"date": dt.date(2010, 1, 1), "temperature_2m_mean": 10.0},
        {"date": dt.date(2010, 1, 2), "temperature_2m_mean": None},
        {"date": dt.date(2010, 1, 3), "temperature_2m_mean": 14.0},
    ]
    out = meteo.annual_mean_temperature(rows)
    assert out == {2010: 12.0}


def test_meteo_request_rejects_unknown_city():
    with pytest.raises(Exception):
        meteo.MeteoRequest(city="Paris", start_date="2010-01-01", end_date="2010-01-02")


def test_meteo_request_rejects_bad_date_order():
    with pytest.raises(Exception):
        meteo.MeteoRequest(
            city="Madrid", start_date="2010-01-02", end_date="2010-01-01"
        )
