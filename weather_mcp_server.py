# ==========================================================
# Weather MCP Server: the "tool provider"
# ==========================================================
# This file holds the tools and nothing else: no model, no API key.
# An MCP client (weather_agent_with_mcp.py, Claude Desktop, Cursor...)
# starts this file, asks which tools it has, and calls them.
#
# Every MCP server has three parts:
#   1. Create the server     -> mcp = FastMCP("Weather")
#   2. Register the tools    -> @mcp.tool() above each function
#   3. Run the server        -> mcp.run(transport="stdio")
#
# Don't use print() in this file: with stdio, the terminal output is
# the channel the server uses to talk to the client, so a print would
# get mixed into its messages and break the connection.
# ==========================================================

from mcp.server.fastmcp import FastMCP  # The MCP server class (mcp version 1)
import openmeteo_requests               # Reads Open-Meteo's weather replies
import httpx                            # Sends plain web requests (used for geocoding)

# The Open-Meteo client for get_weather. Open-Meteo needs no API key.
client = openmeteo_requests.Client()

# Part 1: Create the server. "Weather" is the name the server gives
# itself; it doesn't have to match the name the agent uses for it.
mcp = FastMCP("Weather")


# Part 2: Register the tools.
# @mcp.tool() puts the function on the server. The server sends each tool's
# name, inputs (from the type hints) and docstring to the client, and the
# model reads them to decide which tool to call, so keep docstrings clear.

@mcp.tool()
def get_coordinates(city: str) -> str:
    """Look up the latitude and longitude of a city by its name. Use this before get_weather."""
    # Ask Open-Meteo's geocoding API for the best match (count=1) for this name
    response = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=20)
    data = response.json()  # Turn the JSON reply into a Python dictionary

    # When no city matches, the reply has no "results" key at all
    if "results" not in data:
        return f"Could not find a city called {city}"

    # "results" is a list of matches; [0] is the first (and only) one
    place = data["results"][0]
    return f"{place['name']}, {place['country']}: latitude {place['latitude']}, longitude {place['longitude']}"


@mcp.tool()
def get_weather(latitude: float, longitude: float) -> str:
    """Get the current temperature at a latitude and longitude."""
    # Ask for current conditions at these coordinates. The values come back
    # in the same order as the "current" list: 0 = temperature, 1 = wind speed.
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast",
        params={"latitude": latitude, "longitude": longitude, "current": ["temperature_2m", "wind_speed_10m"]})

    # responses[0] is the first (only) location; Variables(0) is temperature_2m
    temperature = responses[0].Current().Variables(0).Value()
    return f"{round(temperature, 1)} C degrees"


# Part 3: Run the server.
# Only runs when this file is started directly (the agent starts it this way).
# The server then waits for requests over stdio (standard input/output).
# If you run it yourself it looks frozen: that's it waiting. Stop it with Ctrl+C.
if __name__ == "__main__":
    mcp.run(transport="stdio")
