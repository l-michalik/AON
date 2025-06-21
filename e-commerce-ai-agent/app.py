import json
import os
import asyncio

from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from flask import Flask, render_template, redirect, url_for, flash, request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

server_params = StdioServerParameters(
    command="npx",
    args=['@brightdata/mcp'],
    env={
        'API_TOKEN': os.getenv("BRIGHT_DATA_API_KEY"),
        'BROWSER_AUTH': os.getenv("BROWSER_AUTH"),
        'WEB_UNLOCKER_ZONE': os.getenv("WEB_UNLOCKER_ZONE"),
    }
)

SYSTEM_PROMPT = (
    "To find products, first use the search_engine tool. When finding products, use the web_data tool for the platform. If none exists, scrape as markdown."
    "Example: Don't use web_data_bestbuy_products for search. Use it only for getting data on specific products you already found in search."
)