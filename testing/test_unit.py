import sys
import os
import json
import numpy as np
import pytest
import torch
import faiss
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.scraper import WikipediaScraper, save_scraped_data
from app.rag_pipeline import RAGPipeline
from app.llm import OllamaLLM
from app.data_processing import scrape_and_store
from app.vector_store import FAISSVectorStore
from main import log_interaction, display_history, run_ui, main
from sentence_transformers import SentenceTransformer


# --- FIXTURES ---
@pytest.fixture
def setup_components():
    scraper = WikipediaScraper()
    embedding_model = MagicMock(spec=SentenceTransformer)
    vector_store = MagicMock()
    llm = MagicMock(spec=OllamaLLM)
    rag_pipeline = RAGPipeline(vector_store, llm, embedding_model)
    return scraper, embedding_model, vector_store, llm, rag_pipeline


@pytest.fixture
def faiss_store():
    embedding_model = MagicMock()
    embedding_model.encode.return_value = np.random.rand(2, 384)
    return FAISSVectorStore(embedding_model, dimension=384)


# --- SCRAPER TESTS ---
def test_scrape_success(setup_components, mocker):
    scraper, *_ = setup_components
    mock_get = mocker.patch('app.scraper.requests.get')
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "<p>Mock content</p>"
    result = scraper.scrape("https://example.com")
    assert len(result) > 0


# --- VECTOR STORE TESTS ---
def test_faiss_store_and_query(faiss_store):
    texts = ["doc1", "doc2"]
    embeddings = np.random.rand(2, 384).tolist()
    faiss_store.store(texts, embeddings)
    query_embedding = np.random.rand(384).tolist()
    results = faiss_store.query(query_embedding, top_k=1)
    assert isinstance(results, list)
    assert results[0] in texts


def test_faiss_store_empty(faiss_store, caplog):
    faiss_store.store([], [])
    assert "No texts provided for storing." in caplog.text


def test_faiss_mmr_query(faiss_store):
    texts = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 384).tolist()
    faiss_store.store(texts, embeddings)
    results = faiss_store.query(np.random.rand(384).tolist(), top_k=2, use_mmr=True)
    assert len(results) == 2


def test_faiss_save_load(faiss_store, tmp_path):
    texts = ["doc1", "doc2"]
    embeddings = np.random.rand(2, 384).tolist()
    faiss_store.store(texts, embeddings)

    index_file = tmp_path / "test_index.bin"
    texts_file = tmp_path / "test_texts.pkl"
    emb_file = tmp_path / "test_emb.pkl"

    faiss_store.save(index_file, texts_file, emb_file)
    loaded = FAISSVectorStore.load(faiss_store.embedding_model, index_file, texts_file, emb_file)
    assert loaded.texts == texts


# --- LLM TESTS ---
def test_ollama_generate_success(mocker):
    llm = OllamaLLM()
    mocker.patch('app.llm.requests.get', return_value=MagicMock(status_code=200))
    mock_post = mocker.patch('app.llm.requests.post')

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b'{"response": "Hello", "done": true}']
    mock_post.return_value.__enter__.return_value = mock_response

    result = llm.generate("Test prompt")
    assert "Hello" in result


def test_ollama_json_error(mocker):
    llm = OllamaLLM()
    mocker.patch('app.llm.requests.get', return_value=MagicMock(status_code=200))
    mock_post = mocker.patch('app.llm.requests.post')

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b'invalid json']
    mock_post.return_value.__enter__.return_value = mock_response

    result = llm.generate("Test prompt")
    assert "Error" in result


# --- RAG PIPELINE TESTS ---
def test_rag_prompt_structure(setup_components):
    _, embedding_model, vector_store, llm, rag_pipeline = setup_components
    embedding_model.encode.return_value = np.array([[0.1]*384])
    vector_store.query.return_value = ["context1"]
    llm.generate.side_effect = lambda prompt: prompt

    prompt = rag_pipeline.process("Explain RAG.")
    assert "Context:" in prompt
    assert "Question:" in prompt


def test_rag_process_no_contexts(setup_components):
    _, embedding_model, vector_store, _, rag_pipeline = setup_components
    embedding_model.encode.return_value = np.array([[0.1]*384])
    vector_store.query.return_value = []

    result = rag_pipeline.process("Test question")
    assert "No relevant information found" in result


# --- MAIN MODULE TESTS ---
def test_log_interaction_content(mocker):
    mock_open_func = mock_open()
    mocker.patch("builtins.open", mock_open_func)
    log_interaction("New question?", "Detailed answer!")

    handle = mock_open_func()
    written_data = handle.write.call_args[0][0]
    assert "New question?" in written_data
    assert "Detailed answer!" in written_data


def test_display_history_parsing(mocker):
    test_data = [
        '{"question": "Q1", "answer": "A1"}\n',
        'invalid json\n',
        '{"question": "Q2", "answer": "A2"}\n'
    ]

    mocker.patch('main.os.path.exists', return_value=True)
    mocker.patch('main.open', mock_open(read_data=''.join(test_data)))
    mock_st = mocker.patch("main.st")

    display_history()
    assert mock_st.sidebar.expander.call_count == 2


# --- DATA PROCESSING TESTS ---
def test_scrape_and_store_new_data(mocker):
    mocker.patch('app.data_processing.os.path.exists', return_value=False)
    scraper = MagicMock()
    vector_store = MagicMock()
    scraper.scrape.side_effect = [["text1"], ["text2"]]

    scrape_and_store(scraper, vector_store, ["url1", "url2"])
    vector_store.store.assert_called_once()
    vector_store.save.assert_called_once()


def test_scrape_and_store_empty_urls(mocker):
    mocker.patch('app.data_processing.os.path.exists', return_value=False)
    scraper = MagicMock()
    vector_store = MagicMock()

    scrape_and_store(scraper, vector_store, [])
    vector_store.store.assert_called_once()


# --- EDGE CASE TESTS ---
def test_large_batch_processing(faiss_store):
    large_texts = [f"doc{i}" for i in range(1000)]
    embeddings = np.random.rand(1000, 384).tolist()

    faiss_store.store(large_texts, embeddings)
    results = faiss_store.query(np.random.rand(384).tolist(), top_k=100)
    assert len(results) == 100


def test_special_characters_handling(setup_components):
    _, embedding_model, vector_store, llm, rag_pipeline = setup_components
    vector_store.query.return_value = ["Context with special chars: äöü©"]
    llm.generate.return_value = "Properly handled"

    result = rag_pipeline.process("Test special chars?")
    assert "handled" in result


if __name__ == "__main__":
    pytest.main(["-v", __file__])