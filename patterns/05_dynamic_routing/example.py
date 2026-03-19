import os
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define Router Schema
class Route(BaseModel):
    category: Literal["technical", "billing", "general"] = Field(description="The category of the user's inquiry.")

# Define Agent State
class AgentState(TypedDict):
    input: str
    category: str
    response: str

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", temperature=0)
router_llm = llm.with_structured_output(Route)

def router_node(state: AgentState):
    """Categorizes the user's inquiry."""
    prompt = f"Categorize the following customer inquiry into 'technical', 'billing', or 'general':\n\n{state['input']}"
    route = router_llm.invoke(prompt)
    return {"category": route.category}

def technical_node(state: AgentState):
    """Handles technical inquiries."""
    prompt = f"Provide technical support for: {state['input']}"
    response = llm.invoke(prompt)
    return {"response": response.content}

def billing_node(state: AgentState):
    """Handles billing inquiries."""
    prompt = f"Provide billing support for: {state['input']}"
    response = llm.invoke(prompt)
    return {"response": response.content}

def general_node(state: AgentState):
    """Handles general inquiries."""
    prompt = f"Provide general support for: {state['input']}"
    response = llm.invoke(prompt)
    return {"response": response.content}

def route_inquiry(state: AgentState):
    """Routes the inquiry based on category."""
    if state["category"] == "technical":
        return "technical"
    elif state["category"] == "billing":
        return "billing"
    else:
        return "general"

from langgraph.graph import StateGraph, START, END

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("technical", technical_node)
workflow.add_node("billing", billing_node)
workflow.add_node("general", general_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_inquiry,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general"
    }
)
workflow.add_edge("technical", END)
workflow.add_edge("billing", END)
workflow.add_edge("general", END)

app = workflow.compile()

def visualize_graph():
    """Prints ASCII and saves PNG image using LangGraph's built-in method."""
    print("\n--- Graph Visualization ---")
    app.get_graph().print_ascii()
    print("---------------------------\n")
    
    try:
        # LangGraph built-in PNG generation
        png_data = app.get_graph().draw_mermaid_png()
        with open("patterns/05_dynamic_routing/graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as PNG to patterns/05_dynamic_routing/graph.png")
    except Exception as e:
        print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    visualize_graph()
    print("--- Pattern 05: Dynamic Routing ---")
    queries = [
        "My laptop is not turning on. It's showing a blue screen.",
        "I was overcharged for my last subscription payment.",
        "What are your business hours?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        inputs = {"input": query}
        result = app.invoke(inputs)
        print(f"Category: {result['category']}")
        print(f"Response: {result['response'][:200]}...")
