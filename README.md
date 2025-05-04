# 🧠 RAG-LLM: Multi-Source Q&A System with Evaluation

## ✨ Project Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system designed to generate high-quality answers from multiple authoritative sources including Wikipedia, NASA, National Geographic, History.com, and Britannica. The solution leverages state-of-the-art tools and models to build a production-ready pipeline, supporting everything from data scraping to question-answer evaluation.

### 🔑 Key Features:

- 📚 **Comprehensive Data Collection**: Scraping and processing 100 URLs from multiple sources:
  - Wikipedia pages (Science, History, Mathematics)
  - NASA articles (Space, Astronomy, Technology)
  - National Geographic content (Environment, History, Science)
  - History.com articles (Historical Events, Civilizations)
  - Britannica entries (Science, History, Culture)
- 🔍 **Advanced Text Processing**: Efficient chunking and cleaning of content
- 🤖 **State-of-the-art Embedding**: Using sentence-transformers/all-MiniLM-L6-v2 model
- 💾 **Efficient Storage**: FAISS vector database for fast similarity search
- 🧠 **Smart Generation**: Ollama LLM for high-quality answer generation
- 🎨 **User-friendly Interface**: Streamlit-based UI with real-time interaction
- 📊 **Robust Evaluation**: Comprehensive testing and evaluation framework
- 📝 **Detailed Logging**: Complete interaction history and performance tracking
- ⚡ **Performance Optimization**: Fast retrieval and response times

## 📁 Folder Structure

```
Assignment/
├── app/                      # Main application code
│   ├── __init__.py
│   ├── all_Urls.py          # URL management
│   ├── data_processing.py   # Data processing utilities
│   ├── embeddings.pkl       # Stored embeddings
│   ├── faiss_index.bin      # FAISS vector index
│   ├── llm.py              # Language model related code
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   ├── scraper.py          # Web scraping functionality
│   ├── scraped_data.txt    # Scraped data storage
│   ├── texts.pkl           # Processed text data
│   └── vector_store.py     # Vector store implementation
│
├── testing/                # Testing related files
│   ├── test_rag.py        # QA testing
│   ├── test_unit.py       # Unit testing
│  
├── test_cases/           # Test cases
├── logs/                 # Log files
├── chroma_db_store/     # ChromaDB storage
├── rag_env/             # Virtual environment
├── main.py              # Main entry point
├── requirements.txt     # Project dependencies
└── .gitignore          # Git ignore file
```

## 🏗️ Architecture Overview

The system follows a standard RAG pipeline:

1. **Data Collection**: Scraping Wikipedia content
2. **Text Processing**: Cleaning and chunking the content
3. **Embedding**: Converting text to vectors using sentence-transformers
4. **Storage**: FAISS vector database for efficient similarity search
5. **Retrieval**: Finding relevant context for questions
6. **Generation**: Using Ollama LLM to generate answers 
7. **Evaluation**: Comprehensive testing and evaluation framework

## 🔄 Workflow

```
┌───────────────┐
│ 1️⃣ Data       │  📚 Scrape multi-source content (Wikipedia, NASA, NatGeo, History, Britannica)
│ Collection    │
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ 2️⃣ Processing │  🧹 Clean & chunk data, extract metadata, validate
│    Phase      │
└──────┬────────┘
       │
       ▼
┌──────────────┐
│ 3️⃣ Embedding │  🤖 Embed text with SOTA model, store in FAISS
│    Phase     │
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ 4️⃣ Query           │  ❓ User input ➡️ Embed query ➡️ Retrieve context from FAISS
│   Processing       │
└──────┬─────────────┘
       │
       ▼
┌──────────────────────┐
│ 5️⃣ Answer            │  🧠 LLM answer generation, context prep, formatting
│   Generation         │
└──────┬───────────────┘
       │
       ▼
┌────────────────────────────┐
│ 6️⃣ QA Test Case            │  ✍️ Manual 1000+ Q&A test cases, categorize, validate, document
│    Generation              │
└──────┬─────────────────────┘
       │
       ▼
┌───────────────┐
│ 7️⃣ Evaluation │  📊 Evaluate & log metrics
│    Phase      │
└───────────────┘
```

## 🌐 Scraping Pipeline

### 📦 Core Module: scraper.py

- Uses custom Wikipedia scraper to extract clean text
- Saves content in structured format
- Handles error cases and logging

### 🔄 Workflow:

1. Initialize scraper with target URLs
2. Extract and clean content from each page
3. Save processed content for further use
4. Log all operations for debugging

## ⚙️ Chunking, Embedding & FAISS Storage

### 🔍 Embedding:

- Model Used: sentence-transformers/all-MiniLM-L6-v2
- Optimized for semantic search
- Efficient vector operations

### 💾 Storage:

- FAISS for fast similarity search
- Persistent storage of embeddings and texts
- Efficient retrieval for real-time Q&A

## 🔎 Retrieval & RAG Pipeline

- Embedding of user query
- Similarity search in FAISS vector store
- FAISS vector store retrieves top-k relevant chunks
- Retrieved context and question sent to LLM for answer generation
- LLM generates the final response

## 🖥️ Streamlit UI Chatbot

### ✨ Features:

- Question input form
- Answer display
- Response time metrics
- Sidebar with history
- Error handling and user feedback

## 📈 Evaluation Strategy

### 📊 Evaluation Components:

1. **QA Evaluation**
   - Automated testing framework
   - Multiple(1000+) QA test cases
   - Performance metrics:
     - **BERT F1 Score (0.911)**: Measures semantic similarity between generated and reference answers
     - **Cosine Similarity (0.747)**: Evaluates vector space similarity between answers
     - **BLEU Score (0.222)**: Measures n-gram overlap between generated and reference answers
     - **ROUGE Score (0.488)**: Evaluates recall of n-grams between answers
     - **METEOR Score (0.566)**: Considers word alignment and synonym matching
     - **Entailment Analysis**:
       - Contradiction Rate: 46.08% (411 cases)
       - Entailment Rate: 43.83% (391 cases)
       - Neutral Rate: 10.09% (90 cases)

2. **Unit Testing**
   - Comprehensive test suite for core components:
     - Scraper functionality testing
     - Vector store operations (FAISS)
     - LLM integration and response generation
     - RAG pipeline processing
     - Data processing and storage
   - Edge case handling:
     - Large batch processing
     - Special character handling
     - Empty input scenarios
   - Mock testing for external dependencies

3. **Logging and Monitoring**
   - Detailed interaction logs
   - Error monitoring

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main application:
```bash
python main.py
```

3. Access the Streamlit UI in your browser

## 🔮 Future Work

- Enhanced evaluation metrics
- Support for more document types
- Improved chunking strategies
- Using more Quick Vector Storage and Retrieval
- High performance Text Embedding LLM
- Asynchronous scraping and test case evaluation

## 📄 Documentation

For a detailed explanation of each project file and its role in the pipeline, see:
[project-file-explain-doc](https://shorthillstech.sharepoint.com/:fl:/g/contentstorage/CSP_50cb9b3f-f5a4-4ab6-92e4-91bc8701321a/EUyCBVreyJhBmGBgl9twR1EBJXGJkKs2fvKp3GPIZwRJdg?e=FPQKgr&nav=cz0lMkZjb250ZW50c3RvcmFnZSUyRkNTUF81MGNiOWIzZi1mNWE0LTRhYjYtOTJlNC05MWJjODcwMTMyMWEmZD1iJTIxUDV2TFVLVDF0a3FTNUpHOGh3RXlHcVpQcmRIbnJhVkZwNEx0S2JCM2dITWk0V3JHVnBmN1FZUTJRUkdRSzJSUyZmPTAxV1NDUU9GS01RSUNWVlhXSVRCQVpRWURBUzdOWEFSMlImYz0lMkYmYT1Mb29wQXBwJnA9JTQwZmx1aWR4JTJGbG9vcC1wYWdlLWNvbnRhaW5lciZ4PSU3QiUyMnclMjIlM0ElMjJUMFJUVUh4emFHOXlkR2hwYkd4emRHVmphQzV6YUdGeVpYQnZhVzUwTG1OdmJYeGlJVkExZGt4VlMxUXhkR3R4VXpWS1J6aG9kMFY1UjNGYVVISmtTRzV5WVZaR2NEUk1kRXRpUWpOblNFMXBORmR5UjFad1pqZFJXVkV5VVZKSFVVc3lVbE44TURGWFUwTlJUMFpQTmxsTVNVRkhUak5LVVVwRFdVZExXRWRWVDBsRVRFY3pSQSUzRCUzRCUyMiUyQyUyMmklMjIlM0ElMjIyYzEyZjIwNS02YjAyLTQ4MWItODZjNC1lYmQ0Y2ZmMDJhYTMlMjIlN0Q%3D)
