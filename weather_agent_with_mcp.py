# ==========================================================
# Weather Agent with MCP (tools come from an MCP server)
# ==========================================================
# The same agent as weather_agent.py, but this file has no tools.
# It starts weather_mcp_server.py, asks it which tools it has,
# and gives them to the model.
#
#   weather_agent_with_mcp.py   ──starts and talks to──▶   weather_mcp_server.py
#   (model + Groq key)                                      (tools, no key)
#
# What changed from weather_agent.py:
#   - Tools come from await client.get_tools() instead of @tool functions
#   - The code runs inside async def main(), because talking to the
#     server means waiting for replies (await)
#   - agent.invoke(...) became await agent.ainvoke(...)
#
# Run with: python weather_agent_with_mcp.py  (with the .venv activated,
# because "command": "python" below uses whichever python the terminal finds)
# ==========================================================

# ─────────────────────────────────────────
# STEP 0: Load the API key
# ─────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()  # Loads variables like GROQ_API_KEY into environment variables
# Checks that the key has a value (it doesn't prove Groq will accept it)
print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Hides library warnings to keep the output clean

import asyncio                                            # Runs async functions: asyncio.run(main())
from langchain_mcp_adapters.client import MultiServerMCPClient  # The MCP client: connects to servers

# ─────────────────────────────────────────
# STEP 1: Initialize the model
# ─────────────────────────────────────────
# Same as before: MCP changes where the tools come from, not the model
from langchain.chat_models import init_chat_model
weather_model = init_chat_model("groq:openai/gpt-oss-120b")

from langchain.agents import create_agent

async def main():

    """
    Main async function — MCP connections are async because
    they involve I/O (spawning processes, network calls).
    """

    # ─────────────────────────────────────────
    # STEP 2: Connect to the MCP server
    # ─────────────────────────────────────────
    # Instead of defining tools here, we point to an MCP server and
    # let the adapter discover all of its tools automatically.

    # Build the full path to the server file, so it's found no matter
    # which folder you run this file from. The file name must match exactly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "weather_mcp_server.py")

    # These are only the connection settings: nothing contacts the server
    # until get_tools() below.
    client = MultiServerMCPClient(
        {
            "weather": {                    # Our label for this server (any name works)
                # The MCP server to connect to: run "python weather_mcp_server.py"
                "command": "python",
                "args": [server_path],
                "transport": "stdio",       # Local process (stdin/stdout)
            },

            # ── Want MORE tools? Just add more servers! ──
            # The MultiServerMCPClient can connect to several servers at once,
            # and the agent gets all of their tools. For example:
            #
            # "github": {
            #     "command": "npx",
            #     "args": ["-y", "@modelcontextprotocol/server-github"],
            #     "transport": "stdio",
            #     "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")},
            # },
        }
    )

    # ─────────────────────────────────────────
    # STEP 3: Discover the tools from the server
    # ─────────────────────────────────────────
    # Starts the server, asks for its tools (the server logs this as
    # "Processing request of type ListToolsRequest") and converts them
    # into LangChain tools. await pauses main() until the reply arrives.
    tools = await client.get_tools()
    print("=" * 55)
    print(" MCP Agent — Tools discovered from MCP Server")
    print("=" * 55)
    print(f"\n📋 Found {len(tools)} tools from MCP server:\n")
    for t in tools:
        print(f"{t.name}: {t.description[:60]}...")  # Name + first 60 characters of the docstring
    print()

    # ─────────────────────────────────────────
    # STEP 4: Create the agent
    # ─────────────────────────────────────────
    # Same as before: the agent doesn't know or care that the tools came
    # from MCP. It just sees LangChain tools and uses them normally.
    agent = create_agent(model = weather_model, tools = tools)

    # ─────────────────────────────────────────
    # STEP 5: Run the agent
    # ─────────────────────────────────────────
    # Defined inside main() because it uses agent, which only exists here.
    # async because ainvoke waits on Groq and on the server.
    async def run_agent(question: str):
            """Run the agent and print the execution trace."""
            print(f" User: {question}")
            print("-" * 50)

            # Send the question; the agent loops through the tools until it has an answer.
            # result["messages"] holds every message from the run, in order.
            result = await agent.ainvoke({
                "messages": [("user", question)]
            })

            for msg in result["messages"]:
                # The question itself (already printed above)
                if msg.type == "human":
                    continue
                elif msg.type == "ai":
                    # AI message with tool_calls = the agent decided to use a tool
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"Agent thinks → calling: {tool_call['name']}({tool_call['args']})")
                    # AI message without tool_calls = the final answer
                    elif msg.content:
                        print(f"Agent answer: {msg.content}")
                elif msg.type == "tool":
                    # Extract clean result from MCP tool response
                    content = msg.content
                    # MCP returns structured content as a list of dicts
                    # e.g. [{'type': 'text', 'text': '120.0', ...}]
                    # Extract just the text value for clean display
                    if isinstance(content, list):
                        texts = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
                        content = ", ".join(texts) if texts else str(content)
                    print(f"Tool result: {content}")

            print("=" * 55)
            print()

    # ─────────────────────────────────────────
    # TEST CASES
    # ─────────────────────────────────────────
    # Same questions as weather_agent.py, so the results can be compared.
    # Indented 4 spaces (inside main, not inside run_agent), and awaited
    # because run_agent is async.

    await run_agent("What is the weather in New Jersey right now?")
    await run_agent("What is the weather in New Delhi, India right now?")
    # get_weather only gives the CURRENT temperature, so this tests how the
    # agent handles a question its tools can't fully answer
    await run_agent("What is the weather in Manchester, England in 5 days?")

# Start main(). asyncio.run is what runs an async function; calling
# main() by itself would not run its code.
if __name__ == "__main__":
    asyncio.run(main())
