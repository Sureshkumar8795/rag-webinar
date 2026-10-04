# RAG Chatbot with Groq, HuggingFace & Chroma

A simple **Retrieval-Augmented Generation (RAG)** chatbot built using:

- Python
- LangChain
- Groq
- HuggingFace Embeddings
- Chroma Vector Database

## What is RAG?

**RAG = Retrieval-Augmented Generation**

RAG is a technique where an AI model first searches for relevant information from documents or a database, and then uses that information to generate an answer.

### Simple Workflow

```text
User Question
      ↓
Search Documents / Database
      ↓
Find Relevant Information
      ↓
Give Information to AI
      ↓
AI Generates Answer
      ↓
Final Answer
```

### Easy to Remember

```text
RAG = Retrieve → Augment → Generate
```

- **Retrieve** → Find relevant information
- **Augment** → Add that information to the AI prompt
- **Generate** → AI generates the answer

---

# Project Workflow

```text
1. Install Required Packages
        ↓
2. Import Required Libraries
        ↓
3. Add Groq API Key
        ↓
4. Create Knowledge Base
        ↓
5. Create Text Embeddings
        ↓
6. Store Documents in Chroma Vector Database
        ↓
7. Create Retriever
        ↓
8. Create Groq LLM
        ↓
9. Build RAG Chat Function
        ↓
10. Retrieve Relevant Documents
        ↓
11. Create Context + Prompt
        ↓
12. Generate Answer Using Groq
        ↓
13. Test the RAG Chatbot
        ↓
14. Run Continuous Chatbot Loop
```

---

# Technologies Used

### Python

Used to build the complete RAG application.

### LangChain

Used to connect and manage different components of the RAG pipeline.

### HuggingFace Embeddings

Used to convert text into numerical vector representations.

### Chroma

Used as the vector database to store and search document embeddings.

### Groq

Used as the Large Language Model (LLM) to generate answers.

---

# Installation

Install the required packages:

```python
!pip install -q langchain langchain-community langchain-groq langchain-huggingface chromadb sentence-transformers
```

---

# Import Libraries

```python
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
```

---

# Add Groq API Key

```python
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Replace `YOUR_GROQ_API_KEY` with your actual Groq API key.

**Never share your API key publicly.**

---

# Create Knowledge Base

```python
documents = [
    Document(
        page_content="""
        Python is a popular programming language.
        It is easy to learn and widely used in AI, machine learning,
        web development, automation and data science.
        """
    ),

    Document(
        page_content="""
        Machine Learning is a field of AI where computers learn patterns
        from data and use those patterns to make predictions or decisions.
        """
    ),

    Document(
        page_content="""
        RAG stands for Retrieval-Augmented Generation.
        RAG allows an AI model to retrieve relevant information from
        external documents before generating an answer.
        """
    )
]

print("Documents created:", len(documents))
```

---

# Create Embeddings

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Embeddings convert text into numerical vectors so that similar information can be found through semantic search.

---

# Create Chroma Vector Database

```python
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

print("Vector database created!")
```

The documents are converted into embeddings and stored in Chroma.

---

# Create Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)
```

The retriever searches the vector database and returns the most relevant documents.

`k=2` means it retrieves the top 2 relevant documents.

---

# Create Groq LLM

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
```

The Groq model generates the final answer using the retrieved information.

---

# Build the RAG Chat Function

```python
def rag_chat(question):

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Combine retrieved information
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I don't know based on the provided information."
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content
```

---

# How the RAG Function Works

```text
User Question
      ↓
Retriever
      ↓
Search Chroma Database
      ↓
Relevant Documents
      ↓
Create Context
      ↓
Context + Question
      ↓
Groq LLM
      ↓
Generated Answer
```

---

# Test the RAG Chatbot

```python
question = "What is RAG?"

answer = rag_chat(question)

print(answer)
```

Example output:

```text
RAG stands for Retrieval-Augmented Generation.
It allows an AI model to retrieve relevant information
from external documents before generating an answer.
```

---

# Ask Another Question

```python
question = "What is Python used for?"

print(rag_chat(question))
```

Example output:

```text
Python is used in AI, machine learning,
web development, automation and data science.
```

---

# Test an Unknown Question

```python
question = "Who is the President of the United States?"

print(rag_chat(question))
```

If the information is not available in the knowledge base, the chatbot is instructed to respond:

```text
I don't know based on the provided information.
```

---

# Simple Chatbot Loop

```python
while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    answer = rag_chat(question)

    print("Bot:", answer)
```

Now you can continuously ask questions.

Example:

```text
You: What is Python?

Bot: Python is a popular programming language...

You: What is RAG?

Bot: RAG stands for Retrieval-Augmented Generation...

You: What is machine learning?

Bot: Machine Learning is a field of AI...

You: bye

Bot: Goodbye!
```

---

# Complete RAG Architecture

```text
                KNOWLEDGE BASE
                      │
                      ↓
              HuggingFace Embeddings
                      │
                      ↓
              Chroma Vector Database
                      │
                      │
User Question ────────┤
                      ↓
                  Retriever
                      ↓
             Relevant Documents
                      ↓
              Context + Question
                      ↓
                  Groq LLM
                      ↓
                Final Answer
```

---

# Key Concepts

| Concept | Purpose |
|---|---|
| Document | Stores the knowledge |
| Embedding | Converts text into vectors |
| Vector Database | Stores and searches vectors |
| Chroma | Our vector database |
| Retriever | Finds relevant documents |
| Context | Retrieved information given to the LLM |
| LLM | Generates the final answer |
| RAG | Connects retrieval with generation |

---

# Final Summary

RAG allows an AI model to use **external knowledge** when generating answers.

The basic process is:

```text
Question
   ↓
Retrieve
   ↓
Relevant Information
   ↓
Augment
   ↓
LLM
   ↓
Generate
   ↓
Answer
```

**RAG = Retrieve → Augment → Generate**
