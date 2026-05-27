from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load Chroma Vector Store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# User Query
query = "How much did Microsoft pay to acquire GitHub?"


# Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Optional similarity threshold search
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }
# )

# Retrieve documents
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("\n--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Build Prompt
combined_input = f"""
Based on the following documents, please answer this question:

{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents.

If you can't find the answer in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

# -----------------------------------
# Gemini Model
# -----------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

# -----------------------------------
# Messages
# -----------------------------------
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# -----------------------------------
# Generate Response
# -----------------------------------
result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Content only:")
print(result.content)