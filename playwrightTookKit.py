from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser
)


llm = OllamaLLM(
    model="deepseek-r1:32b",
    temperature=0.0,
    base_url="http://localhost:11434"
)

async_browser = create_async_playwright_browser()


toolkit = PlayWrightBrowserToolkit.from_browser(async_browser)

tools = toolkit.get_tools()
print(tools)


tool_by_name = {tool.name: tool for tool in tools}
print(tool_by_name)
navigate_tool = tool_by_name["navigate_browser"]
navigate_tool
