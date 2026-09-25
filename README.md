# 🧠 PDFMind — AI-Powered PDF Question Answering

**PDFMind** is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions about its content.

Instead of manually searching through lengthy documents, PDFMind processes the uploaded PDF, retrieves the most relevant information, and uses a Large Language Model (LLM) to generate an answer based on the document.

The project demonstrates an end-to-end **RAG pipeline using LangChain, Document Loaders, Text Splitters, Embeddings, and a Vector Database**.

---

## 🚀 Features

* 📤 Upload any PDF document
* 📖 Extract text from the uploaded PDF
* ✂️ Split documents into smaller chunks
* 🔢 Generate embeddings for document chunks
* 🗄️ Store embeddings in a Vector Database
* 🔍 Retrieve relevant information using similarity search
* 🤖 Generate answers using an LLM
* 💬 Ask multiple questions about the uploaded document
* 🧠 Uses RAG to ground responses in the uploaded document

---

## 🏗️ How PDFMind Works

PDFMind follows a standard **Retrieval-Augmented Generation (RAG)** pipeline:

```text
                 PDF Upload
                     │
                     ▼
              Document Loader
                     │
                     ▼
              Text Extraction
                     │
                     ▼
               Text Splitter
                     │
                     ▼
                Text Chunks
                     │
                     ▼
                 Embeddings
                     │
                     ▼
              Vector Database
                     │
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         │
   User Question                 │
        │                         │
        ▼                         │
  Question Embedding             │
        │                         │
        ▼                         │
  Similarity Search ─────────────┘
        │
        ▼
 Relevant Document Chunks
        │
        ▼
        LLM
        │
        ▼
   Final Answer
```

---

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines information retrieval with Large Language Models.

A normal LLM generates an answer using the knowledge it learned during training.

RAG adds an additional retrieval step where relevant information is first retrieved from an external knowledge source — in this case, the user's uploaded PDF.

### Without RAG

```text
Question
   ↓
  LLM
   ↓
Answer
```

### With RAG

```text
Question
   ↓
Retrieve Relevant Information
   ↓
Relevant PDF Chunks
   ↓
LLM + Retrieved Context
   ↓
Answer
```

This allows PDFMind to generate responses based on the content of the uploaded document.

---

# 🔄 RAG Pipeline in PDFMind

## 1. 📄 Document Loading

When a user uploads a PDF, PDFMind uses a **Document Loader** to read and extract the document content.

```text
PDF
 ↓
Document Loader
 ↓
Extracted Document
```

The extracted content is then passed to the next stage of the pipeline.

---

## 2. ✂️ Text Splitting

Large documents are divided into smaller chunks using a **Text Splitter**.

```text
Large PDF
    ↓
 ┌─────────┐
 │ Chunk 1 │
 ├─────────┤
 │ Chunk 2 │
 ├─────────┤
 │ Chunk 3 │
 ├─────────┤
 │   ...   │
 └─────────┘
```

Chunking makes it easier for the retrieval system to find the specific parts of the document relevant to a user's question.

---

## 3. 🔢 Embeddings

Each text chunk is converted into a numerical representation called an **embedding**.

For example:

```text
"Transformers use self-attention"
                 ↓
          Embedding Model
                 ↓
     [0.21, -0.43, 0.87, ...]
```

Embeddings represent the semantic meaning of text as vectors.

Texts with similar meanings tend to have similar vector representations.

---

## 4. 🗄️ Vector Database

The generated embeddings are stored in a **Vector Database**.

When the user asks a question, the question is also converted into an embedding.

The system then performs a similarity search to find the document chunks that are most relevant to the question.

```text
User Question
      ↓
Question Embedding
      ↓
Vector Similarity Search
      ↓
Relevant PDF Chunks
```

---

## 5. 🔍 Retrieval

The most relevant document chunks are retrieved and used as context for the LLM.

```text
PDF Chunks
    ↓
Vector Search
    ↓
Top Relevant Chunks
    ↓
Context
```

This is the **Retrieval** part of Retrieval-Augmented Generation.

---

## 6. 🤖 Generation

The retrieved context is combined with the user's question and passed to the LLM.

```text
User Question
      +
Retrieved Context
      ↓
     LLM
      ↓
Generated Answer
```

The LLM uses the retrieved information to generate a natural-language response.

---

# 💡 Example

Suppose a user uploads a research paper about **Transformers**.

The user asks:

> **"What is self-attention?"**

PDFMind performs the following steps:

```text
Question
   ↓
Convert question to embedding
   ↓
Search Vector Database
   ↓
Retrieve relevant PDF chunks
   ↓
Send chunks + question to LLM
   ↓
Generate answer
```

The user receives an answer based on the content of the uploaded PDF.

---

# 🛠️ Technologies Used

| Technology           | Purpose                                     |
| -------------------- | ------------------------------------------- |
| **Python**           | Core programming language                   |
| **LangChain**        | Building and managing the RAG pipeline      |
| **Document Loaders** | Loading and extracting PDF content          |
| **Text Splitters**   | Breaking documents into smaller chunks      |
| **Embeddings**       | Converting text into vector representations |
| **Vector Database**  | Storing and retrieving document embeddings  |
| **LLM**              | Generating answers from retrieved context   |
| **RAG**              | Combining retrieval with generation         |

---

# 📂 Project Structure

```text
PDFMind/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   └── uploaded_documents/
│
└── ...
```

> The exact structure may vary depending on the implementation.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

```bash
cd PDFMind
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
```

Replace the value with your own API key.

⚠️ **Never commit your `.env` file or API keys to GitHub.**

Make sure `.env` is included in `.gitignore`:

```text
.env
venv/
__pycache__/
```

---

## 5. Run PDFMind

If the application uses Streamlit:

```bash
streamlit run app.py
```

Otherwise, run:

```bash
python app.py
```

---

# 🎯 Use Cases

PDFMind can be used for:

* 📚 Research paper analysis
* 🎓 Academic document Q&A
* 📖 Book and document analysis
* 💼 Business report analysis
* 📑 Technical documentation
* 🧪 Scientific papers
* 📝 Study material
* 📊 Company reports
* 📄 Large PDF document search

---

# 📚 Key Concepts Learned

Through this project, I explored and implemented:

* Document loading
* PDF text extraction
* Text chunking
* Chunk size and overlap
* Embeddings
* Vector databases
* Vector similarity search
* Retrieval-Augmented Generation
* LangChain pipelines
* Prompt construction
* LLM-based question answering
* Environment variable management
* End-to-end GenAI application development

---

# 🔮 Future Improvements

* [ ] Support multiple PDFs simultaneously
* [ ] Add conversational chat history
* [ ] Display source documents/chunks for every answer
* [ ] Add PDF page-number citations
* [ ] Implement hybrid search
* [ ] Add document re-ranking
* [ ] Support DOCX and TXT files
* [ ] Add persistent vector storage
* [ ] Improve retrieval accuracy
* [ ] Deploy the application online
* [ ] Add user authentication

---

# 📈 Complete Workflow

```text
              ┌─────────────────┐
              │    Upload PDF   │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Document Loader │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Text Splitting │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   Embeddings    │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Vector Database │
              └────────┬────────┘
                       │
                       │
              User Question
                       ↓
              ┌─────────────────┐
              │Question Embedding│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Similarity Search│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Relevant Context│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │       LLM       │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Final Answer   │
              └─────────────────┘
```

---

# 👨‍💻 Author

**Jatin Arya**

IIT Madras
Dual Degree — Biological Engineering / Biotechnology

Interested in:

**Generative AI • Machine Learning • AI/ML Engineering • Data Science • Computational Biology**

---

## ⭐ Support

If you found **PDFMind** useful or interesting, consider giving the repository a ⭐ on GitHub!
