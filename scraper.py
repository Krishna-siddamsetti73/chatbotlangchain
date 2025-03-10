from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Path to store the ChromaDB
CHROMA_DB_PATH = "./chroma_db"

# Step 1: Load or Create ChromaDB
if not os.path.exists(CHROMA_DB_PATH):
    print("Loading data for the first time...")

    # Load Data from Webpage
    url = "https://brainlox.com/courses/category/technical"
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Load Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store Embeddings in ChromaDB
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_DB_PATH)
    vectorstore.persist()
    print("Data saved to ChromaDB.")

else:
    print("Loading existing data from ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

# Step 2: Use Chroma Retriever
retriever = vectorstore.as_retriever()

# Flask API to Handle Chat Requests
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve the most relevant documents
    relevant_docs = retriever.get_relevant_documents(user_query)
    results = [doc.page_content for doc in relevant_docs]

    return jsonify({"response": results})

if __name__ == "__main__":
    app.run(debug=True)
