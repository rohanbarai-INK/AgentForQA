from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_ollama import OllamaLLM

# ===============================
# Custom Tools
# ===============================

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return result"""
    return int(a) + int(b)

@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers and return result"""
    return int(a) - int(b)

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers and return result"""
    return int(a) * int(b)

tools = [add_numbers, subtract_numbers, multiply_numbers]

# ===============================
# LLM Configuration
# ===============================

llm = OllamaLLM(
    model="deepseek-r1:32b",
    temperature=0.0,
    base_url="http://localhost:11434"
)

# ===============================
# Agent (Verbose ON)
# ===============================

agent_verbose = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print("\n--- Verbose ON ---")
print(agent_verbose.run("What's the sum of 20 and 30?"))

# ===============================
# Agent (Verbose OFF)
# ===============================

agent_silent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

print("\n--- Verbose OFF ---")
print(agent_silent.run("What's the sum of 20 and 30?"))