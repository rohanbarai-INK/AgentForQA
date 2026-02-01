# ==========================================================
# 1️⃣ Imports
# ==========================================================

import os
# Failed to send telemetry event ...
# These are harmless and caused by:
# - Version mismatch inside Chroma telemetry
# - No impact on retrieval or answers
# You can silence them with:
# 🔴 Chroma telemetry (THIS is the one causing your error)
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"  # optional, legacy

# 🔴 LangChain telemetry (good practice)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# 🔴 Ollama telemetry (optional, not your error)
os.environ["OLLAMA_NO_TELEMETRY"] = "1"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ==========================================================
# 2️⃣ Load PDF Documents
# =========================================================

docs_folder = "./Docs"
documents = []

for filename in os.listdir(docs_folder):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(docs_folder, filename)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

# ✅ CHANGE 1: add filename explicitly to metadata
for doc in documents:
    doc.metadata["filename"] = os.path.basename(doc.metadata["source"])

print(os.listdir(docs_folder))
print(f"Loaded {len(documents)} document pages")


# ==========================================================
# 3️⃣ Split Documents into Chunks
# ==========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


# ==========================================================
# 4️⃣ Create Embedding Model
# ==========================================================

embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


# ==========================================================
# 5️⃣ Create Vector Store (db)
# ==========================================================

### -  RUN THIS ONE TIME TO CREATE DB FOLDER - ################
# db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding,
#     persist_directory="./qa_db",
#     collection_name="qa_documents"
# )
# db.persist()
db = Chroma(
    persist_directory="./qa_db",
    embedding_function=embedding,
    collection_name="qa_documents"
)
#### - RUN ABOVE EVERY TIME AFTER DB FOLDER GET CREATED IT WILL LOAD THE EXISTING DB 
print("Vector store created")


# ==========================================================
# 6️⃣ Create Retriever
# ==========================================================

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12,  "fetch_k": 60 }
    
)


# ==========================================================
# 7️⃣ Configure LLM
# ==========================================================

llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0.0,
    base_url="http://localhost:11434"
)


# ==========================================================
# 8️⃣ QA Prompt (Grounded)
# ==========================================================

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a QA assistant.\n"
        "- Answer ONLY using the text inside <context>.\n"
        "- Do NOT use external knowledge.\n"
        "- If the answer is not present, say: \"I don't know based on the provided documents.\".\n\n"
        "When answering technical questions:\n"
        "- Explain the architecture in a structured way.\n"
        "- Cover components, data flow, and scalability if mentioned.\n"
        "- Use bullet points or numbered sections when possible.\n\n"
        "<context>\n{context}\n</context>"
    ),
    ("human", "{input}")
])


# ==========================================================
# 9️⃣ Combine Documents + LLM
# ==========================================================

qa_document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)


# ==========================================================
# 🔟 Create Retrieval Chain (Modern Replacement)
# ==========================================================

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=qa_document_chain
)


# ==========================================================
# 1️⃣1️⃣ Ask Question
# ==========================================================

question = "Explain , Recommender System"

response = retrieval_chain.invoke({
    "input": question
})

print("\n✅ Answer (from documents):")
print(response["answer"])


print("\n📄 Retrieved Sources (ground truth):")
for doc in response["context"]:
    print(
        f"- {doc.metadata['filename']} "
        f"(page {doc.metadata.get('page')})"
    )