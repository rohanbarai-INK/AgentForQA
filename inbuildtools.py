from json import tool
from langchain.agents import initialize_agent, AgentType 
from langchain.agents import load_tools
from langchain_ollama import OllamaLLM

# ===============================
# 1️⃣ Configure LLM
# ===============================

llm = OllamaLLM(
    model="deepseek-r1:32b",
    temperature=0.5,
    base_url="http://localhost:11434"
)


# ===============================
# 2️⃣ Load Inbuilt Tools
# ===============================

tools = load_tools(
    ["wikipedia"],
    llm=llm
)

# ===============================
# 3️⃣ Initialize Agent
# ===============================

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ===============================
# 4️⃣ Ask Question (Streaming)
# ===============================

question = "what is tool in AI Agent"

for chunk in agent.stream(question):
    print(chunk)