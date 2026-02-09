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

# Loop through all files in the docs folder
for filename in os.listdir(docs_folder):
    # Check if the file is a PDF
    if filename.lower().endswith(".pdf"):
        # Create the full path to the file
        file_path = os.path.join(docs_folder, filename)
        # Initialize the PDF loader
        loader = PyPDFLoader(file_path)
        # Load the PDF and add the pages to the documents list
        documents.extend(loader.load())

# ✅ CHANGE 1: add filename explicitly to metadata
# Add the filename to each document's metadata for reference
for doc in documents:
    doc.metadata["filename"] = os.path.basename(doc.metadata["source"])

print(os.listdir(docs_folder))
print(f"Loaded {len(documents)} document pages")


# ==========================================================
# 3️⃣ Split Documents into Chunks
# ==========================================================

# Create a text splitter to break documents into manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum characters per chunk
    chunk_overlap=200  # Characters to overlap between chunks
)

# Split the loaded documents into chunks
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


# ==========================================================
# 4️⃣ Create Embedding Model
# ==========================================================

# Initialize the embedding model to convert text to vectors
embedding = OllamaEmbeddings(
    model="nomic-embed-text",  # Embedding model name
    base_url="http://localhost:11434"  # Ollama server URL
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
# Load the vector store from the persisted directory
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

# Create a retriever to fetch relevant documents from the vector store
retriever = db.as_retriever(
    search_type="mmr",  # Use Maximal Marginal Relevance for diversity
    search_kwargs={"k": 12, "fetch_k": 60}  # Return 12 docs, fetch 60 candidates
)


# ==========================================================
# 7️⃣ Configure LLM
# ==========================================================

# Set up the language model for generating answers
llm = OllamaLLM(
    model="llama3.1:8b",  # Model name
    temperature=0.0,  # Low temperature for consistent answers
    base_url="http://localhost:11434"  # Ollama server URL
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

# Create a chain that combines retrieved documents with the LLM
qa_document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)


# ==========================================================
# 🔟 Create Retrieval Chain (Modern Replacement)
# ==========================================================

# Create the complete retrieval-augmented generation chain
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=qa_document_chain
)


# ==========================================================
# 1️⃣1️⃣ Ask Question
# ==========================================================

# Define the question to ask the system
question = "Explain , Recommender System"

# Run the retrieval chain with the question to get the answer
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