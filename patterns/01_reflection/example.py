import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Load environment variables (API keys)
load_dotenv()

# Define the state of our graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    iterations: int

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def generator_node(state: AgentState):
    """Generates a response or an update based on feedback."""
    response = llm.invoke(state['messages'])
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }

def critic_node(state: AgentState):
    """Critiques the generator's output to find potential improvements."""
    # We take the last message (the generation) and ask for a critique
    last_message = state['messages'][-1].content
    critique_prompt = f"Critique the following code for efficiency, readability, and edge cases. Provide specific feedback for improvement:\n\n{last_message}"
    
    critique = llm.invoke([HumanMessage(content=critique_prompt)])
    # We mark this as a critique in the message history
    return {"messages": [AIMessage(content=f"CRITIQUE: {critique.content}")]}

def should_continue(state: AgentState):
    """Decides whether to reflect again or finish."""
    if state["iterations"] >= 2:
        return END
    return "critic"

# Define the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)

# Set entry point
workflow.set_entry_point("generator")

# Add edges
workflow.add_edge("critic", "generator")
workflow.add_conditional_edges(
    "generator", 
    should_continue,
    {
        "critic": "critic",
        END: END
    }
)

# Compile
app = workflow.compile()

def visualize_graph():
    """Prints ASCII and saves PNG image using LangGraph's built-in method."""
    print("\n--- Graph Visualization ---")
    app.get_graph().print_ascii()
    print("---------------------------\n")
    
    try:
        # LangGraph built-in PNG generation
        png_data = app.get_graph().draw_mermaid_png()
        with open("patterns/01_reflection/graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as PNG to patterns/01_reflection/graph.png")
    except Exception as e:
        print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    visualize_graph()
    print("--- Pattern 01: Reflection ---")
    initial_prompt = "Write a fast Python function to calculate the nth Fibonacci number."
    inputs = {"messages": [HumanMessage(content=initial_prompt)], "iterations": 0}
    
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\nNode: {key}")
            if "messages" in value:
                print(f"Content: {value['messages'][-1].content[:200]}...")
