from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

# os.environ["OPENAI_API_KEY"] = openai_api_key
# os.environ["OPENAI_BASE_URL"] = openai_base_url
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(
    model="gpt-5.2",
    api_key=openai_api_key,
    base_url=openai_base_url
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    "Make a tool-calling agent"

    @tool
    def add(a: float, b: float):
        """Add two numbers."""
        return a + b
    
    tool_node = ToolNode([add])
    model_with_tool = llm.bind_tools([add])

    def call_model(state: State):
        messages = state["messages"]
        response = model_with_tool.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END
        
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

    return graph_workflow.compile()

agent = make_alternative_graph()