import os
import asyncio
import warnings
from dotenv import load_dotenv
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.messages import HumanMessage
from graph import builder

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
TTL_CONFIG = {"default_ttl": 5, "refresh_on_read": False}

async def run_test(app, user_input: str, config: dict):
    print(f"\n{'-'*50}")
    print(f"entrada do usuário: {user_input}")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "next_agent": None,
        "delegation_instruction": None,
        "final_response": None
    }
    
    final_state = initial_state
    
    async for output in app.astream(initial_state, config):
        for node_name, state_update in output.items():
            if "next_agent" in state_update:
                print(f"-> roteando o controle para: [{state_update['next_agent']}]")
            final_state = state_update
            
            if "final_response" in state_update:
                return final_state.get('final_response')

async def main():
    async with AsyncRedisSaver.from_conn_string(REDIS_URL, ttl=TTL_CONFIG) as _checkpointer:
        await _checkpointer.asetup()
        
        _graph = builder.compile(checkpointer=_checkpointer)
        config = {"configurable": {"thread_id": "test_session_1"}}
        
        while True:
            try:
                msg = input("\nDigite sua mensagem (ou 'sair'): ")
            except EOFError:
                break

            if msg.lower() == "sair":
                break
        
            response = await run_test(_graph, msg, config)
            print(f"\n[resposta final]: {response}")

if __name__ == "__main__":
    asyncio.run(main())