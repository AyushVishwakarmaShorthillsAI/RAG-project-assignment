import sys
import os
import json
import numpy as np
import pytest
import torch
import faiss
from unittest.mock import MagicMock, mock_open, patch
import requests

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
    mock_html = (
        '<html><body><div id="mw-content-text">'
        '<p>This is the first paragraph of mock content which is definitely longer than fifty characters to pass the filter.</p>'
        '<p>This is the second paragraph, also sufficiently long for testing purposes.</p>'
        '</div></body></html>'
    )
    mock_get.return_value.text = mock_html
    result = scraper.scrape("https://en.wikipedia.org/wiki/Dark_(TV_series)")
    assert len(result) > 0
    assert "first paragraph" in result[0]


def test_scrape_request_error(setup_components, mocker, caplog):
    scraper, *_ = setup_components
    mock_get = mocker.patch('app.scraper.requests.get', side_effect=requests.exceptions.RequestException("Network Error"))
    result = scraper.scrape("http://example.com/fail")
    assert result == []
    assert "Scraping failed for http://example.com/fail: Network Error" in caplog.text


def test_scrape_unexpected_html(setup_components, mocker, caplog):
    scraper, *_ = setup_components
    mock_get = mocker.patch('app.scraper.requests.get')
    mock_get.return_value.status_code = 200
    # HTML without the expected #mw-content-text or .mw-parser-output
    mock_html = '<html><body><div><p>Just some random paragraph.</p></div></body></html>'
    mock_get.return_value.text = mock_html
    result = scraper.scrape("http://example.com/no_content")
    assert len(result) == 0
    assert "Extracted 0 paragraphs" in caplog.text


def test_scrape_invalid_url_404(setup_components, mocker, caplog):
    scraper, *_ = setup_components
    mock_get = mocker.patch('app.scraper.requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
    mock_get.return_value = mock_response

    result = scraper.scrape("http://example.com/not_found")
    assert result == []
    assert "Scraping failed for http://example.com/not_found: 404 Client Error" in caplog.text


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


def test_faiss_query_empty_store(faiss_store, caplog):
    query_embedding = np.random.rand(384).tolist()
    results = faiss_store.query(query_embedding, top_k=1)
    assert results == []
    assert "FAISS index is empty" in caplog.text


def test_faiss_load_file_not_found(faiss_store, tmp_path):
    index_file = tmp_path / "non_existent_index.bin"
    texts_file = tmp_path / "non_existent_texts.pkl"
    emb_file = tmp_path / "non_existent_emb.pkl"

    with pytest.raises(FileNotFoundError):
        FAISSVectorStore.load(faiss_store.embedding_model, str(index_file), str(texts_file), str(emb_file))


def test_faiss_query_k_greater_than_n(faiss_store):
    texts = ["doc1", "doc2"]
    embeddings = np.random.rand(2, 384).tolist()
    faiss_store.store(texts, embeddings)
    query_embedding = np.random.rand(384).tolist()
    # Ask for 5 results when only 2 are stored
    results = faiss_store.query(query_embedding, top_k=5)
    assert isinstance(results, list)
    # Should return all available documents (2)
    assert len(results) == 2
    assert "doc1" in results
    assert "doc2" in results


def test_faiss_save_load(faiss_store, tmp_path):
    texts = ["doc1", "doc2"]
    embeddings = np.random.rand(2, 384).tolist()
    faiss_store.store(texts, embeddings)

    index_file = tmp_path / "test_index.bin"
    texts_file = tmp_path / "test_texts.pkl"
    emb_file = tmp_path / "test_emb.pkl"

    faiss_store.save(str(index_file), str(texts_file), str(emb_file))
    loaded = FAISSVectorStore.load(faiss_store.embedding_model, str(index_file), str(texts_file), str(emb_file))
    assert loaded.texts == texts


# --- LLM TESTS ---
def test_ollama_generate_connection_error(mocker):
    llm = OllamaLLM()
    # Mock requests.get and requests.post to raise ConnectionError
    mocker.patch('app.llm.requests.get', side_effect=requests.exceptions.ConnectionError("Ollama not available"))
    mocker.patch('app.llm.requests.post', side_effect=requests.exceptions.ConnectionError("Ollama not available"))

    result = llm.generate("Test prompt")
    assert "Error: Ollama server not running or unreachable." in result


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


def test_rag_process_long_question(setup_components):
    _, embedding_model, vector_store, llm, rag_pipeline = setup_components
    long_question = "Explain " + " ".join(["everything"] * 500) + " about the universe?"
    embedding_model.encode.return_value = np.array([[0.2]*384])
    vector_store.query.return_value = ["long context"]
    # Just return the prompt to check if it was formed correctly
    llm.generate.side_effect = lambda prompt: f"LLM processed: {prompt}"

    result = rag_pipeline.process(long_question)
    assert "LLM processed" in result
    assert "Context:\nlong context" in result
    assert f"Question: {long_question}\nBased on the provided context, answer the question in brief." in result


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


def test_display_history_no_file(mocker):
    mocker.patch('main.os.path.exists', return_value=False)
    mock_st = mocker.patch("main.st")
    display_history()
    # Check if the info message for no history was called
    mock_st.sidebar.info.assert_called_once_with("No previous interactions found.")
    # Ensure no expanders were created
    mock_st.sidebar.expander.assert_not_called()


def test_run_ui_empty_question_submission(mocker):
    # Mock the RAG pipeline
    mock_rag_pipeline = MagicMock(spec=RAGPipeline)

    # Mock streamlit components and functions used by run_ui
    mock_st = mocker.patch("main.st")
    mocker.patch("main.display_history") # Prevent history display during test
    mock_log_interaction = mocker.patch("main.log_interaction")

    # Simulate form submission with an empty question
    mock_st.text_input.return_value = "   " # Whitespace question
    mock_st.form_submit_button.return_value = True # Form submitted

    # Mock session state using MagicMock to allow attribute access
    mock_st.session_state = MagicMock()

    # Call the function under test
    run_ui(mock_rag_pipeline)

    # Assertions
    mock_st.text_input.assert_called_once_with("Enter your question:")
    mock_st.form_submit_button.assert_called_once_with("Get Answer")
    # Check that the warning is displayed
    mock_st.warning.assert_called_once_with("Please enter a valid question.")
    # Check that the RAG pipeline was NOT called
    mock_rag_pipeline.process.assert_not_called()
    # Check that interaction was NOT logged
    mock_log_interaction.assert_not_called()


# --- DATA PROCESSING TESTS ---
def test_scrape_and_store_new_data(mocker, setup_components):
    mocker.patch('app.data_processing.os.path.exists', return_value=False)
    scraper = MagicMock()
    vector_store = MagicMock()
    _, embedding_model, _, _, _ = setup_components
    scraper.scrape.side_effect = [["text1"*20], ["text2"*20]]

    scrape_and_store(scraper, vector_store, embedding_model, ["url1", "url2"])
    vector_store.store.assert_called_once()
    if hasattr(vector_store, 'save'):
        vector_store.save.assert_called_once()


def test_scrape_and_store_empty_urls(mocker, setup_components):
    mocker.patch('app.data_processing.os.path.exists', return_value=False)
    scraper = MagicMock()
    vector_store = MagicMock()
    _, embedding_model, _, _, _ = setup_components

    scrape_and_store(scraper, vector_store, embedding_model, [])
    vector_store.store.assert_called_once_with([], [])
    if hasattr(vector_store, 'save'):
        vector_store.save.assert_called_once()


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