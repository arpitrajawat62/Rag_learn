from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# SAME embedding model used during indexing
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# User query
query = "How much did Microsoft pay to acquire GitHub?"

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Retrieve relevant docs
relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")

print("\n--- Relevant Documents ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}")
    print("-" * 50)
    print(doc.page_content)

    
# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"