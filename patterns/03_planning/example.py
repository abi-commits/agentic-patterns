import os
from typing import TypedDict, Annotated, List, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the data structure for the plan
class Plan(BaseModel):
    steps: List[str] = Field(description="A list of steps to execute the task.")

# Define Agent State
class AgentState(TypedDict):
    input: str
    plan: List[str]
    results: List[str]
    current_step: int

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", temperature=0)
planner_llm = llm.with_structured_output(Plan)

def planner_node(state: AgentState):
    """Generates a plan based on the input."""
    prompt = f"Create a step-by-step plan to achieve the following goal: {state['input']}. Limit to 3 steps."
    plan_output = planner_llm.invoke(prompt)
    return {"plan": plan_output.steps, "current_step": 0}

def executor_node(state: AgentState):
    """Executes the current step of the plan."""
    current_step_text = state["plan"][state["current_step"]]
    prompt = f"Perform the following task: {current_step_text}"
    result = llm.invoke(prompt)
    return {
        "results": state.get("results", []) + [result.content],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: AgentState):
    """Decides if there are more steps to execute."""
    if state["current_step"] >= len(state["plan"]):
        return END
    return "executor"

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "executor": "executor",
        END: END
    }
)

app = workflow.compile()

def visualize_graph():
    """Prints ASCII and saves PNG image using LangGraph's built-in method."""
    print("\n--- Graph Visualization ---")
    app.get_graph().print_ascii()
    print("---------------------------\n")
    
    try:
        # LangGraph built-in PNG generation
        png_data = app.get_graph().draw_mermaid_png()
        with open("patterns/03_planning/graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as PNG to patterns/03_planning/graph.png")
    except Exception as e:
        print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    visualize_graph()
    print("--- Pattern 03: Planning ---")
    inputs = {"input": "Outline a marketing strategy for a new coffee shop in Seattle."}
    
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\nNode: {key}")
            if key == "planner":
                print(f"Plan: {value['plan']}")
            elif key == "executor":
                print(f"Step Executed: Result length {len(value['results'][-1])}")
