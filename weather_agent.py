# ==========================================================
# Weather Agent (tools written in this file)
# ==========================================================
# An AI agent that answers weather questions:
#   - The model (gpt-oss-120b on Groq) reads the question and decides what to do
#   - The tools (Open-Meteo) fetch the real data
#
# Agent loop: question -> model picks a tool -> tool runs -> model reads the
# result -> picks another tool or writes the final answer.
#
# See weather_agent_with_mcp.py for the same agent with its tools on an MCP server.
# Run with: python weather_agent.py  (with the .venv activated)
# ==========================================================

# ─────────────────────────────────────────
# STEP 0: Load the API key
# ─────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()  # Loads variables like GROQ_API_KEY into environment variables
# Checks that the key has a value (it doesn't prove Groq will accept it)
print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))

# ─────────────────────────────────────────
# STEP 1: Initialize the model
# ─────────────────────────────────────────
# "groq:" tells LangChain which provider to use; it reads GROQ_API_KEY itself
from langchain.chat_models import init_chat_model
weather_model = init_chat_model("groq:openai/gpt-oss-120b")

# ─────────────────────────────────────────
# STEP 2: Define the tools
# ─────────────────────────────────────────
from langchain_core.tools import tool  # @tool turns a function into an agent tool
import openmeteo_requests               # Reads Open-Meteo's weather replies
import httpx                            # Sends plain web requests (used for geocoding)

# The Open-Meteo client for get_weather. Open-Meteo needs no API key.
client = openmeteo_requests.Client()

# The model only sees each tool's name, inputs (from the type hints) and
# docstring, and uses them to decide which tool to call, so keep docstrings clear.

@tool
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


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get the current temperature at a latitude and longitude."""
    # Ask for current conditions at these coordinates. The values come back
    # in the same order as the "current" list: 0 = temperature, 1 = wind speed.
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast",
        params={"latitude": latitude, "longitude": longitude, "current": ["temperature_2m", "wind_speed_10m"]})

    # responses[0] is the first (only) location; Variables(0) is temperature_2m
    temperature = responses[0].Current().Variables(0).Value()
    return f"{round(temperature, 1)} C degrees"


#Combine all of the tools in a list
tools = [get_coordinates, get_weather]

# ─────────────────────────────────────────
# STEP 3: Create the agent
# ─────────────────────────────────────────
# Connects the model to the tools. No system_prompt: the docstrings
# are enough for the model to pick the right tools in the right order.
from langchain.agents import create_agent
agent = create_agent(model = weather_model, tools = tools)


# ─────────────────────────────────────────
# STEP 4: Run the agent
# ─────────────────────────────────────────
def run_agent(question: str):
    """Run the agent and print a clean, and organized execution trace"""
    print(f"\nUser: {question}")
    print("-" * 20)
    # Send the question; the agent loops through the tools until it has an answer.
    # result["messages"] holds every message from the run, in order.
    result = agent.invoke({"messages": [("user", question)]})


    print("Clean Agent Execution Trace")
    print("-" * 20)

    step = 1

    # Print each message, labeled by what kind it is
    for msg in result["messages"]:
        #1 Human Message = Original User Question
        if msg.type == "human":
            print(f"{step}. User asked")
            print(f"    {msg.content}")
            step += 1
        #2. AI message with tool_calls = agent decided to use a tool
        # (a loop, because the model can ask for more than one tool at once)
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
              for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"{step}. Agent decision:")
                print(f"   I need to use the tool: {tool_name}")
                print(f"   Tool input: {tool_args}")
                step += 1

        #3. Tool message = what the tool returned
        elif msg.type == "tool":
            print(f"{step}. Tool returned")
            print(f"    {msg.content}")
            step += 1

        #4. AI message without tool_calls = final answer
        # (must come after #2, or every AI message would count as the answer)
        elif msg.type == "ai":
            print(f"{step}. Final Answer")
            print(f"    {msg.content}")
            step += 1

    print("---" * 20)


#TEST CASES
# These start at the far left, so they run when the file runs
#1. Agent should use both tools
run_agent("What is the weather in New Jersey right now?")
run_agent("What is the weather in New Delhi, India right now?")
# get_weather only gives the CURRENT temperature, so this tests how the
# agent handles a question its tools can't fully answer
run_agent("What is the weather in Manchester, England in 5 days?")
