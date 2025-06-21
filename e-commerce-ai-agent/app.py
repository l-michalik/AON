import json
import os
import asyncio

from typing import List, Optional
from pydantic import BaseModel, Field
from flask import Flask, render_template, redirect, url_for, flash, request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent