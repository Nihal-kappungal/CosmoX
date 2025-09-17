# run_demo.py
import asyncio
from research_agents import orchestrator, Runner

async def chat_once(user_text: str):
    result = await Runner.run(orchestrator, input=user_text, max_turns=8)
    print("\n— Assistant —\n")
    print(result.final_output)

if __name__ == "__main__":
    print("Type your query (Ctrl+C to exit).")
    try:
        while True:
            msg = input("\nYou: ")
            asyncio.run(chat_once(msg))
    except KeyboardInterrupt:
        pass
