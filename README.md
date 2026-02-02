# RAG Chatbot Project Documentation
## Project Structure
The project has a simple structure with all source code in the src/ directory containing 11 Python modules:
- `main.py` - Entry point for the application
- `config.py` - Configuration and environment variable management
- `rag.py` - Core RAG logic coordinating retrieval and generation
- `ingestion.py` - Document ingestion and vector store initialization
- `loader.py` - Document loading and text chunking
- `embeddings.py` - Text embedding generation
- `vectorstore.py` - Vector database operations using ChromaDB
- `bm25.py` - BM25 index build, persistence, and search
- `hybrid_search.py` - Reciprocal Rank Fusion and hybrid retrieval
- `llm.py` - LLM prompt building and API calls
- `memory.py` - Conversation history management
## Dependencies
The project includes a `requirements.txt` file. Core dependencies include:
- `python-dotenv` - For loading environment variables 
- `google-generativeai` - For Google's Gemini API and embeddings 
- `chromadb` - For vector database storage 
- `pypdf` - For PDF document reading 
- `langchain-text-splitters` - For text chunking 
- `rank-bm25` - For BM25 keyword search
## Configuration Requirements (Environment Variables)
The application requires the following environment variables to be set in a `.env` file:
- `GOOGLE_API_KEY` (Required) - Google API key for Gemini and embeddings
- `GEMINI_MODEL` (Optional) - Gemini model to use, defaults to "models/gemini-flash-latest"
- `MEMORY_MAX_MESSAGES` (Optional) - Maximum number of conversation messages to retain, defaults to 10
- `HYBRID_VECTOR_K` (Optional) - Number of vector candidates to retrieve before fusion, defaults to 8
- `HYBRID_BM25_K` (Optional) - Number of BM25 candidates to retrieve before fusion, defaults to 8
- `RRF_K` (Optional) - RRF rank constant, defaults to 60

## Setup Instructions
Clone the repository
Install the required dependencies
Create a `.env` file in the project root with your `GOOGLE_API_KEY`
The vector database will be automatically created at `data/vectordb/`

## Usage Examples
The application can be run in two modes:
1. Interactive Mode
Run without arguments to start an interactive chat session:
```python
python src/main.py
```
> Type your questions and receive answers. Exit by typing `exit` or `quit`, or pressing Enter on an empty line.

2. Single Query Mode
Pass a question as command-line arguments:
```python
python src/main.py "What is the main topic of the document?"
```

## Current Features
1. **PDF Document Ingestion** - Loads and processes PDF files into a vector database 
2. **Text Chunking** - Splits documents into manageable chunks with configurable size and overlap (default: 500 characters with 50 character overlap)
3. **Semantic Search** - Uses Google's text-embedding-004 model for generating embeddings 
4. **Hybrid Retrieval** - Combines vector search with BM25 keyword search using Reciprocal Rank Fusion
5. **Conversation Memory** - Maintains chat history with configurable message limit
6. **Context-Aware Responses** - Retrieves top k relevant chunks (default k=3) and includes conversation history in prompts 
7. **Gemini-Powered Generation** - Uses Google's Gemini API for generating responses

## Next Steps
- [ ] local LLM
- [ ] multiple doc usage for context
- [ ] concurrency embedding
