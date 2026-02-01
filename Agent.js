import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { loadTools } from "langchain/agents";
import { Ollama } from "@langchain/ollama";

// ===============================
// 1️⃣ Configure LLM
// ===============================

const llm = new Ollama({
  model: "deepseek-r1:32b",
  temperature: 0.5,
  baseUrl: "http://localhost:11434",
});

// ===============================
// 2️⃣ Load Inbuilt Tools
// ===============================

const tools = await loadTools(
  ["wikipedia"],
  { llm }
);

// ===============================
// 3️⃣ Initialize Agent
// ===============================

const agentExecutor = await initializeAgentExecutorWithOptions(
  tools,
  llm,
  {
    agentType: "zero-shot-react-description",
    verbose: true,
  }
);

// ===============================
// 4️⃣ Ask Question (Streaming)
// ===============================

const question = "what is tool in AI Agent";

for await (const chunk of agentExecutor.stream({ input: question })) {
  console.log(chunk);
}
