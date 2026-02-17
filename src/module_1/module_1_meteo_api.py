import datetime as dt
import requests
import time
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import matplotlib.pyplot as plt

API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def get_data_meteo_api(city: str, start_date: str, end_date: str):
    """
    Validates the request (city + date range) and retrieves weather data for a city.
    """
    try:
        req = MeteoRequest(city=city, start_date=start_date, end_date=end_date)
    except ValidationError as e:
        print(e)
        return None

    return call_api(req.city, req.start_date.isoformat(), req.end_date.isoformat())


def call_api(city: str, start_date: str, end_date: str):
    """
    Builds the API URL, calls the external API, validates the response schema, and returns structured data.
    """
    try:
        url = (
            API_URL
            + "&latitude="
            + str(COORDINATES[city]["latitude"])
            + "&longitude="
            + str(COORDINATES[city]["longitude"])
            + "&start_date="
            + start_date
            + "&end_date="
            + end_date
            + "&daily="
            + ",".join(VARIABLES)
        )

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for city {city}")
            return None

        raw = response.json()

        try:
            validated = MeteoResponse.model_validate(raw)
        except ValidationError as e:
            print(f"Schema validation failed for city {city}:")
            print(e)
            return None

        return validated

    except Exception as e:
        print(f"An error occurred while fetching data for city {city}: {e}")
        return None


class MeteoRequest(BaseModel):
    """
    Validates input parameters (known city + valid date range).
    """

    city: str
    start_date: dt.date
    end_date: dt.date

    @field_validator("city")
    @classmethod
    def city_must_be_known(cls, v: str):
        if v not in COORDINATES:
            raise ValueError(
                f"Invalid city, '{v}' is not in COORDINATES: {list(COORDINATES.keys())}"
            )
        return v

    @model_validator(mode="after")
    def check_date_after(self):
        if self.end_date < self.start_date:
            raise ValueError(
                f"Invalid range of dates end_date ({self.end_date}) must be on or after start_date ({self.start_date})"
            )
        return self


class MeteoDaily(BaseModel):
    """
    Validates the daily data section and ensures required variables exist and have matching lengths.
    """

    time: list[dt.date]  # parses "YYYY-MM-DD" -> dt.date
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_requested_variables(self):
        n = len(self.time)

        for var in VARIABLES:
            if not hasattr(self, var):
                raise ValueError(f"Missing daily variable '{var}' in API response")

            values = getattr(self, var)

            if not isinstance(values, list):
                raise ValueError(f"daily['{var}'] must be a list")

            if len(values) != n:
                raise ValueError(
                    f"Length mismatch: daily.time has {n} but daily.{var} has {len(values)}"
                )

        return self


class MeteoResponse(BaseModel):
    """
    Wraps the validated API response (only 'daily' is required).
    """

    daily: MeteoDaily
    model_config = {"extra": "allow"}


def normalize_daily(daily: dict, variables: list[str]):
    """
    Converts the daily response dict into a list of row dicts (one per date).
    """
    rows = []
    n = len(daily["time"])

    for i in range(n):
        row = {
            "date": dt.date.fromisoformat(daily["time"][i])
            if isinstance(daily["time"][i], str)
            else daily["time"][i]
        }

        for var in variables:
            row[var] = daily[var][i]

        rows.append(row)

    return rows


def annual_total_precipitation(rows: list[dict]):
    """
    Aggregates daily precipitation into a dict: {year: total_precipitation}.
    """
    result = {}

    for row in rows:
        year = row["date"].year
        rain = row["precipitation_sum"]

        if year not in result:
            result[year] = 0.0

        if rain is not None:
            result[year] += rain

    return result


def plot_annual_total_precipitation_by_city(city_to_rows: dict[str, list[dict]]):
    """
    Plots annual total precipitation for each city from normalized daily rows.
    """
    plt.figure()

    for city in city_to_rows:
        rows = city_to_rows[city]
        data = annual_total_precipitation(rows)

        years = sorted(data.keys())
        values = [data[y] for y in years]

        plt.plot(years, values, label=city)

    plt.title("Annual Total Precipitation by City")
    plt.xlabel("Year")
    plt.ylabel("Total precipitation (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def annual_mean_temperature(rows: list[dict]):
    """
    Aggregates daily temperatures into a dict: {year: mean_temperature}.
    """
    temps_by_year = {}

    for row in rows:
        year = row["date"].year
        temp = row["temperature_2m_mean"]

        if temp is None:
            continue

        if year not in temps_by_year:
            temps_by_year[year] = []

        temps_by_year[year].append(temp)

    result = {}
    for year in temps_by_year:
        values = temps_by_year[year]
        result[year] = sum(values) / len(values)

    return result


def plot_annual_mean_temperature_by_city(city_to_rows: dict[str, list[dict]]):
    """
    Plots annual mean temperature for each city from normalized daily rows.
    """
    plt.figure()

    for city in city_to_rows:
        rows = city_to_rows[city]
        data = annual_mean_temperature(rows)

        years = sorted(data.keys())
        values = [data[y] for y in years]

        plt.plot(years, values, label=city)

    plt.title("Annual Mean Temperature by City")
    plt.xlabel("Year")
    plt.ylabel("Mean temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Runs the full workflow: fetch -> normalize -> store per city -> plot results.
    """
    city_to_rows = {}

    for city in COORDINATES:
        data = get_data_meteo_api(city, start_date="2010-01-01", end_date="2020-12-31")
        if data is None:
            continue

        rows = normalize_daily(data.daily.model_dump(), VARIABLES)
        city_to_rows[city] = rows

        print(city, len(rows), rows[0])
        time.sleep(1)

    plot_annual_total_precipitation_by_city(city_to_rows)
    plot_annual_mean_temperature_by_city(city_to_rows)


if __name__ == "__main__":
    main()
