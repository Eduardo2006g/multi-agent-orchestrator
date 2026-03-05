from langgraph.graph import StateGraph, START, END
from typing import Literal
from state import MultiAgentState
from router import orchestrator_node
from agents.conversational_agent import conversational_node
from apis.oracle_api import oracle_agent_node
from apis.edite_api import edite_agent_node

def route_from_orchestrator(state: MultiAgentState) -> Literal["oraculo_api", "edite_api", "conversational_agent"]:
    intent = state.get("next_agent")
    if intent == "oraculo":
        return "oraculo_api"
    if intent == "edite":
        return "edite_api"
    return "conversational_agent"

def format_final_output(state: MultiAgentState):
    messages = state.get("messages", [])
    if messages:
        return {"final_response": messages[-1].content}
    return {}

builder = StateGraph(MultiAgentState)

builder.add_node("orchestrator", orchestrator_node)

builder.add_node("conversational_agent", conversational_node)
builder.add_node("oraculo_api", oracle_agent_node)
builder.add_node("edite_api", edite_agent_node)

builder.add_node("format_final_output", format_final_output)

builder.add_edge(START, "orchestrator")

# o orquestrador decide para qual agente/api enviar com base no next_agent
builder.add_conditional_edges("orchestrator", route_from_orchestrator)

# depois de consultar a api, o fluxo deve voltar para o agente conversacional
builder.add_edge("oraculo_api", "conversational_agent")
builder.add_edge("edite_api", "conversational_agent")

# o agente conversacional é sempre o último da cadeia
builder.add_edge("conversational_agent", "format_final_output")
builder.add_edge("format_final_output", END)

# O grafo será compilado no main.py com o checkpointer
# app = builder.compile()
