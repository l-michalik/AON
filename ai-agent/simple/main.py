from mcp import ClientSession, StdioServerParameters # type: ignore
from mcp.client.stdio import stdio_client # type: ignore
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME", "gpt-4.1"),
    temperature=float(os.getenv("TEMPERATURE", 0.0)),
    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
)

server_params = StdioServerParameters(
    command="npx",
    env={
        'FIRECRAWL_API_KEY': os.getenv("FIRECRAWL_API_KEY", ""),
    },
    args=['firecrawl-mcp']
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can use tools to answer questions."
                }
            ]
            
            print("Available tools:", *[tool.name for tool in tools], sep="\n")
            print("-" * 60)
            
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                
                messages.append({"role": "user", "content": user_input[:175000]})
                
                try:
                    agent_response = await agent.ainvoke({"messages": messages})
                    
                    ai_message = agent_response["messages"][-1].content
                    print(f"AI: {ai_message}")
                except Exception as e:
                    print(f"Error: {e}")
                    
if __name__ == "__main__":
    asyncio.run(main())
                    
                    