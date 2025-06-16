# 📚 LangChain RAG with Chroma + OpenAI

A simple Retrieval-Augmented Generation (RAG) system using:

* 🔎 `Chroma` as a vector database
* 🧠 OpenAI embeddings + `ChatOpenAI`
* 🧩 LangChain for document splitting and processing

---

## ⚙️ Requirements

```bash
pip install langchain langchain-openai langchain-chroma openai python-dotenv
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=sk-...
```

---

## 📁 Files

* `generate.py` – loads `.txt` files from `data/books/`, splits them, and saves embeddings to Chroma
* `query.py` – asks questions using the context retrieved from Chroma

---

## 🚀 Usage

1. Build the vector store:

   ```bash
   python generate.py
   ```

2. Ask a question:

   ```bash
   python query.py "What is The Plague about?"
   ```

---

## ✅ Example Output

```
Response: The main character is a doctor fighting an epidemic.
Sources: ['data/books/the_plague.txt']
```