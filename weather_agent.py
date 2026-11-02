import os
from dotenv import load_dotenv
load_dotenv()  # Loads variables like GROQ_API_KEY into environment variables
print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))

from langchain.chat_models import init_chat_model
weather_model = init_chat_model("groq:openai/gpt-oss-120b")

from langchain_core.tools import tool
import openmeteo_requests
import httpx

client = openmeteo_requests.Client()


@tool
def get_coordinates(city: str) -> str:
    """Look up the latitude and longitude of a city by its name. Use this before get_weather."""
    response = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=20)
    data = response.json()

    # When no city matches, the reply has no "results" key at all
    if "results" not in data:
        return f"Could not find a city called {city}"

    place = data["results"][0]
    return f"{place['name']}, {place['country']}: latitude {place['latitude']}, longitude {place['longitude']}"


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get the current temperature at a latitude and longitude."""
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast",
        params={"latitude": latitude, "longitude": longitude, "current": ["temperature_2m", "wind_speed_10m"]})

    temperature = responses[0].Current().Variables(0).Value()
    return f"{round(temperature, 1)} C degrees"


#Combine all of the tools in a list
tools = [get_coordinates, get_weather]

from langchain.agents import create_agent
agent = create_agent(model = weather_model, tools = tools)


def run_agent(question: str):
    """Run the agent and print a clean, and organized execution trace"""
    print(f"\nUser: {question}")
    print("-" * 20)
    result = agent.invoke({"messages": [("user", question)]})

    
    print("Clean Agent Execution Trace")
    print("-" * 20)

    step = 1

    for msg in result["messages"]: 
        #1 Human Message = Original User Question
        if msg.type == "human":
            print(f"{step}. User asked")
            print(f"    {msg.content}")
            step += 1
        #2. AI message with tool_calls = agent decided to use a tool
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
        elif msg.type == "ai":
            print(f"{step}. Final Answer")
            print(f"    {msg.content}")
            step += 1

    print("---" * 20)


#TEST CASES
#1. Agent should use both tools
run_agent("What is the weather in New Jersey right now?")
run_agent("What is the weather in New Delhi, India right now?")
run_agent("What is the weather in Manchester, England in 5 days?")




