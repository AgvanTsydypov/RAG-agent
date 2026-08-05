from langchain_core.tools import tool

from tools.calculator import calculator
from tools.financials import query_financials
from tools.retrieve import retrieve_documents
from tools.websearch import web_search

# tool(func) turns a plain function into a LangChain tool, deriving its JSON
# schema from the type hints and docstring. The docstrings we wrote in Stage 3
# ARE the descriptions the model reads to choose a tool.
AGENT_TOOLS = [
    tool(retrieve_documents),
    tool(query_financials),
    tool(web_search),
    tool(calculator),
]