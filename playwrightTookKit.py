import json

import playwright.async_api as playwright_async_api
import playwright.sync_api as playwright_sync_api
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_ollama import OllamaLLM


TARGET_URL = "http://eaapp.somee.com/Employee"


# Compatibility fix:
# Newer Playwright versions expose Browser, not AsyncBrowser/SyncBrowser symbols.
# langchain_community Playwright tools still validate against AsyncBrowser/SyncBrowser.
# We add aliases so toolkit validation passes without changing site-packages code.
if not hasattr(playwright_async_api, "AsyncBrowser") and hasattr(
    playwright_async_api, "Browser"
):
    playwright_async_api.AsyncBrowser = playwright_async_api.Browser

if not hasattr(playwright_sync_api, "SyncBrowser") and hasattr(
    playwright_sync_api, "Browser"
):
    playwright_sync_api.SyncBrowser = playwright_sync_api.Browser


def main() -> None:
    # Step 1: Configure local Ollama model.
    # Kept for learning flow; this script does browser tooling directly.
    llm = OllamaLLM(
        model="deepseek-r1:14b",
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    # Step 2: Start a sync Playwright browser (avoids async event-loop conflicts).
    browser = create_sync_playwright_browser()

    try:
        # Step 3: Build toolkit and fetch all browser tools.
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
        tools = toolkit.get_tools()

        print("\nAvailable Tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # Step 4: Map tool names to tool instances for easy lookup.
        tool_by_name = {tool.name: tool for tool in tools}

        # Step 5: Navigate to target page.
        navigate_result = tool_by_name["navigate_browser"].run({"url": TARGET_URL})
        print(f"\nNavigation result: {navigate_result}")

        # Step 6: Get table cell values.
        # Fix note: get_elements returns JSON string, so parse before iterating.
        raw_cells = tool_by_name["get_elements"].run(
            {"selector": "td", "attributes": ["innerText"]}
        )
        cells = json.loads(raw_cells)

        print("\nExtracted Table Data:")
        for cell in cells:
            print(cell)

        # Step 7: Extract full page text and print a short sample.
        page_text = tool_by_name["extract_text"].run({})
        print("\nPage Text Sample:")
        print(page_text[:500])
    finally:
        # Step 8: Always close browser to avoid leftover processes.
        browser.close()


if __name__ == "__main__":
    main()
