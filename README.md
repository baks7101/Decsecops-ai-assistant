# Decsecops-ai-assistant
# devsecops-ai-assistant

# ================================
# 📄 requirements.txt
# ================================
# Add this to your requirements.txt file

langchain
faiss-cpu
tqdm
python-dotenv
openai  # Only if you want optional cloud fallback


# ================================
# 📄 main.py
# ================================
import os
from ollama_interface import get_response_from_ollama
from memory.embedder import search_logs


def main():
    print("\n🔐 DevSecOps AI Assistant\n------------------------")
    query = input("Describe your problem or task: ")

    # Search logs for similar content
    context_chunks = search_logs(query)
    print("\n📚 Retrieved relevant logs:\n")
    for chunk in context_chunks:
        print(f"- {chunk}\n")

    full_prompt = "Here is some context:\n" + "\n".join(context_chunks) + f"\n\nNow respond to this: {query}"
    response = get_response_from_ollama(full_prompt)

    print("\n💡 Assistant's Suggestion:\n")
    print(response)


if __name__ == "__main__":
    main()


# ================================
# 📄 ollama_interface.py
# ================================
import subprocess
import json

def get_response_from_ollama(prompt, model="mistral"):
    print("\n🧠 Generating response using Ollama...")
    command = ["ollama", "run", model]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate(input=prompt.encode())

    if process.returncode != 0:
        print("Error calling Ollama:", error.decode())
        return "[ERROR] LLM failed."
    
    return output.decode()


# ================================
# 📄 memory/embedder.py
# ================================
import os
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from utils.parser import load_logs

# NOTE: Replace OpenAIEmbeddings with a local embedder in future (like HuggingFaceSentenceTransformer)

embedding_model = OpenAIEmbeddings()

def search_logs(query, top_k=3):
    chunks = load_logs("logs/")
    texts = [c["text"] for c in chunks]

    vectors = embedding_model.embed_documents(texts)
    query_vec = embedding_model.embed_query(query)

    index = faiss.IndexFlatL2(len(query_vec))
    index.add(np.array(vectors).astype("float32"))
    _, indices = index.search(np.array([query_vec]).astype("float32"), top_k)

    return [texts[i] for i in indices[0]]


# ================================
# 📄 utils/parser.py
# ================================
import os

def load_logs(log_dir):
    chunks = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".md"):
            with open(os.path.join(log_dir, filename), "r") as f:
                content = f.read()
                parts = content.split("\n\n")
                for part in parts:
                    if part.strip():
                        chunks.append({"text": part.strip()})
    return chunks


# ================================
# 📄 logs/sample_log.md
# ================================
# CI/CD Pipeline Failure - Jenkins

Problem:
Pipeline failed at security scan stage.

Fix:
- Updated the `trivy` image to latest version.
- Increased memory allocation for container.
- Reran the job, passed successfully.

Notes:
This often happens when older scanners are cached in the agent.
