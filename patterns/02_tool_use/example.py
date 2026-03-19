import os
from typing import TypedDict, Annotated, List, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define some tools
@tool
def get_weather(city: str):
    """Get the current weather for a city."""
    if "london" in city.lower():
        return "15°C and Cloudy"
    elif "new york" in city.lower():
        return "22°C and Sunny"
    else:
        return "Unknown city weather."

@tool
def calculate_area(radius: float):
    """Calculates the area of a circle."""
    import math
    return math.pi * (radius ** 2)

tools = [get_weather, calculate_area]
tool_node = ToolNode(tools)

# Initialize the model and bind tools
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# Define nodes
def call_model(state: AgentState):
    """Calls the LLM to decide on a tool or a final answer."""
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Determines if the graph should go to the 'tools' node or end."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

def visualize_graph():
    """Prints ASCII and saves PNG image using LangGraph's built-in method."""
    print("\n--- Graph Visualization ---")
    app.get_graph().print_ascii()
    print("---------------------------\n")
    
    try:
        # LangGraph built-in PNG generation
        png_data = app.get_graph().draw_mermaid_png()
        with open("patterns/02_tool_use/graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as PNG to patterns/02_tool_use/graph.png")
    except Exception as e:
        print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    visualize_graph()
    print("--- Pattern 02: Tool Use (ReAct) ---")
    inputs = {"messages": [HumanMessage(content="What is the area of a circle with a radius of 5? Also, what is the weather in London?")]}
    
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\nNode: {key}")
            if "messages" in value:
                msg = value["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"Tool Call: {msg.tool_calls}")
                else:
                    print(f"Content: {msg.content[:200]}...")
