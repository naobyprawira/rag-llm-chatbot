# RAG+LLM Powered Chatbot API

This is a Flask API application that serves a chatbot using the `Nusantara-7b-Indo-Chat` model. The application leverages transformers, sentence embeddings, and ChromaDB for managing a retrieval-based question-answering (QA) system.

## About Nusantara-7b-Indo-Chat

**Nusantara-7b-Indo-Chat** is part of the Nusantara series, which consists of Open Weight Language Models specifically designed for the Indonesian language (Bahasa Indonesia). Nusantara is based on the **Qwen1.5** Language Model and has been fine-tuned using **Indonesian** datasets.

As a chat-implemented language model, Nusantara is highly capable of handling question-answering and responding to instructions in **Bahasa Indonesia**. Therefore, this application works best when interacting in the Indonesian language.

## Features

- Uses `Nusantara-7b-Indo-Chat` for text generation.
- Optimized for Bahasa Indonesia.
- Utilizes `LazarusNLP/all-indo-e5-small-v4` for text embeddings.
- Incorporates ChromaDB for efficient document retrieval.
- Provides an API endpoint to interact with the chatbot.

## Requirements

- Python 3.8 or higher
- `torch` with CUDA support
- Flask
- transformers
- pandas
- sentence-transformers
- chromadb
- langchain
- flask-cors

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/naobyprawira/rag-llm-chatbot-api.git
    cd rag-llm-chatbot-api
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Place your data file (`dataset.xlsx`) in the project directory.

## Running the Application

1. Ensure that your system has a CUDA-compatible GPU with the necessary drivers installed.

2. Run the Flask application:
    ```bash
    python app.py
    ```

3. The API will be available at `http://127.0.0.1:5000/api/chatbot`.

## API Usage

- **Endpoint:** `/api/chatbot`
- **Method:** `POST`
- **Request Body:** JSON containing the query
    ```json
    {
      "query": "Your question here"
    }
    ```
- **Response:** JSON with the chatbot's response
    ```json
    {
      "response": "Chatbot's response here"
    }
    ```

## Example

You can use `curl` to test the API:

```bash
curl -X POST http://127.0.0.1:5000/api/chatbot \
-H "Content-Type: application/json" \
-d '{"query": "Bagaimana evolusi teknologi memengaruhi cara manusia hidup dan bekerja selama ribuan tahun?"}'
```

## Contributors

[Andhikardg](https://github.com/Andhikardg), [hanifaditia](https://github.com/hanifaditia), ⁠[hauraadzkiaa](https://github.com/hauraadzkiaa) , [marcelbinggi03](https://github.com/marcelbinggi03), ⁠[naobyprawira](https://github.com/naobyprawira)
