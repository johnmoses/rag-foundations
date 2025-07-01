# RAG Starter

## App 1

### 1.1. Install Requirements

```bash
pip install flask
```

### 1.2. Start the Flask Server

```bash
python app1.py
```

### 1.3. Send request to server

Send a POST request to `http://127.0.0.1:5001/query` with JSON body like:

```json
{
  "query": "Flask is"
}
```

Response:

```json
{
    "response": "Answer based on query: 'Flask is' and context: 'Flask is a lightweight web framework for Python.'"
}
```

## App 2

### 2.1. Install Requirements

```bash
pip install flask
```

### 2.2. Start the Flask Server

```bash
python app2.py
```

### 2.3. Send request to server

Send a POST request to `http://127.0.0.1:5001/query` with JSON body like:

```json
{
  "query": "What is Milvus"
}
```

Response:

```json
{
    "retrieved_documents": [
        "Connects to Milvus.",
        "After starting up Milvus,",
        "After starting up Milvus,"
    ]
}
```

## App 3

### 3.1. Install Requirements

```bash
pip install flask
```

### 3.2. Start the Flask Server

```bash
python app3.py
```

### 3.3. Send request to server

Send a POST request to `http://127.0.0.1:5001/query` with JSON body like:

```json
{
  "query": "How is data stored in Milvus"
}
```

Response:

```json
{
    "retrieved_documents": [
        [
            "## What is Milvus vector database?",
            0.6877738237380981
        ],
        [
            "As a database specifically designed to handle queries over input vectors, it is capable of indexing vectors on a trillion scale. Unlike existing relational databases which mainly deal with structured data following a pre-defined pattern, Milvus is designed from the bottom-up to handle embedding vectors converted from [unstructured data](#Unstructured-data).",
            0.6117408275604248
        ],
        [
            "As the Internet grew and evolved, unstructured data became more and more common, including emails, papers, IoT sensor data, Facebook photos, protein structures, and much more. In order for computers to understand and process unstructured data, these are converted into vectors using embedding techniques. Milvus stores and indexes these vectors. Milvus is able to analyze the correlation between two vectors by calculating their similarity distance. If the two embedding vectors are very similar, it means that the original data sources are similar as well.",
            0.5873212814331055
        ]
    ]
}
```
