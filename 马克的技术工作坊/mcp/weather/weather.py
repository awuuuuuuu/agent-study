from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("weather", log_level="ERROR")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g., CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)
    if not data or "features" not in data:
        return "Unable to fetch alerts for the specified state."

    if not data["features"]:
        return "No alerts found for this state."

    alerts = [format_alert(feature) for feature in data["features"] ]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """
    Get a weather forecast for a specific latitude and longitude.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.

    Returns:
        A string containing the weather forecast.
    """
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for the specified location."

    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch forecast data for the specified location."

    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods:
        forecast = f"""
Period: {period.get('name', 'Unknown')}
Temperature: {period.get('temperature', 'Unknown')}°{period.get('temperatureUnit', 'Unknown')}
Wind: {period.get('windSpeed', 'Unknown')} {period.get('windDirection', 'Unknown')}
Forecast: {period.get('forecast', 'No forecast available')}
Description: {period.get('detailedDescription', 'No description available')}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    mcp.run(transport='stdio')