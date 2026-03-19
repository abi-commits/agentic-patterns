import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sender: str

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def writer_node(state: AgentState):
    """The Writer agent drafts the content."""
    prompt = [
        SystemMessage(content="You are a professional writer. Your task is to draft a short blog post based on the topic provided."),
        *state["messages"]
    ]
    response = llm.invoke(prompt)
    return {"messages": [response], "sender": "writer"}

def editor_node(state: AgentState):
    """The Editor agent reviews the writer's work."""
    prompt = [
        SystemMessage(content="You are a meticulous editor. Review the writer's post and provide constructive feedback or a revised version."),
        *state["messages"]
    ]
    response = llm.invoke(prompt)
    return {"messages": [response], "sender": "editor"}

def should_continue(state: AgentState):
    """Decides if the collaboration should continue or end."""
    if state["sender"] == "editor":
        return END
    return "editor"

from langgraph.graph import StateGraph, START, END

# ... (rest of imports)

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.add_edge(START, "writer")
workflow.add_conditional_edges(
    "writer",
    should_continue,
    {
        "editor": "editor",
        END: END
    }
)
workflow.add_edge("editor", END)

app = workflow.compile()

def visualize_graph():
    """Prints ASCII and saves PNG image using LangGraph's built-in method."""
    print("\n--- Graph Visualization ---")
    app.get_graph().print_ascii()
    print("---------------------------\n")
    
    try:
        # LangGraph built-in PNG generation
        png_data = app.get_graph().draw_mermaid_png()
        with open("patterns/04_multi_agent/graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as PNG to patterns/04_multi_agent/graph.png")
    except Exception as e:
        print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    visualize_graph()
    print("--- Pattern 04: Multi-Agent Collaboration ---")
    inputs = {"messages": [HumanMessage(content="Write a post about the future of AI agents.")], "sender": ""}
    
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\nNode: {key}")
            if "messages" in value:
                print(f"Content: {value['messages'][-1].content[:200]}...")
