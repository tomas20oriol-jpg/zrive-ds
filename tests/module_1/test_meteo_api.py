import datetime as dt
import pytest
from unittest.mock import MagicMock

import src.module_1.module_1_meteo_api as meteo


@pytest.fixture(scope="module")
def fake_meteo_response():
    daily = meteo.MeteoDaily(
        time=[dt.date(2010, 1, 1), dt.date(2010, 1, 2)],
        temperature_2m_mean=[10.0, 12.0],
        precipitation_sum=[1.0, None],
        wind_speed_10m_max=[5.0, 6.0],
    )
    return meteo.MeteoResponse(daily=daily)


def make_mock_session(status_code=200, json_data=None, side_effect=None):
    session = MagicMock()
    if side_effect:
        session.get.side_effect = side_effect
    else:
        session.get.return_value = MagicMock(status_code=status_code)
        if json_data is not None:
            session.get.return_value.json.return_value = json_data
    return session


# --- call_api ---

def test_call_api_returns_meteo_response_on_success(monkeypatch, fake_meteo_response):
    """call_api returns a MeteoResponse when the API responds correctly."""
    monkeypatch.setattr(meteo, "get_retry_session", lambda: make_mock_session(200, fake_meteo_response.model_dump()))

    assert isinstance(meteo.call_api("Madrid", "2010-01-01", "2010-01-02"), meteo.MeteoResponse)


def test_call_api_returns_none_on_bad_status(monkeypatch):
    """call_api returns None when the API returns a non-200 status code."""
    monkeypatch.setattr(meteo, "get_retry_session", lambda: make_mock_session(500))

    assert meteo.call_api("Madrid", "2010-01-01", "2010-01-02") is None


def test_call_api_returns_none_on_timeout(monkeypatch):
    """call_api returns None and doesn't raise when the request times out."""
    monkeypatch.setattr(meteo, "get_retry_session", lambda: make_mock_session(side_effect=meteo.requests.exceptions.Timeout))

    assert meteo.call_api("Madrid", "2010-01-01", "2010-01-02") is None


def test_call_api_returns_none_on_invalid_schema(monkeypatch):
    """call_api returns None when the response doesn't match MeteoResponse schema."""
    monkeypatch.setattr(meteo, "get_retry_session", lambda: make_mock_session(200, {"unexpected_field": 123}))

    assert meteo.call_api("Madrid", "2010-01-01", "2010-01-02") is None


# --- get_data_meteo_api ---

def test_get_data_meteo_api_returns_response_on_valid_input(monkeypatch, fake_meteo_response):
    """get_data_meteo_api returns data when city and dates are valid."""
    monkeypatch.setattr(meteo, "call_api", lambda *_: fake_meteo_response)

    assert meteo.get_data_meteo_api("Madrid", "2010-01-01", "2010-01-02") == fake_meteo_response


def test_get_data_meteo_api_returns_none_on_invalid_city():
    """get_data_meteo_api returns None and doesn't raise for an unknown city."""
    assert meteo.get_data_meteo_api("Paris", "2010-01-01", "2010-01-02") is None


def test_get_data_meteo_api_returns_none_on_bad_date_order():
    """get_data_meteo_api returns None when end_date is before start_date."""
    assert meteo.get_data_meteo_api("Madrid", "2010-01-02", "2010-01-01") is None