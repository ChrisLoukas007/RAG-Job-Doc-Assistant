# API Endpoints Documentation

## Core Endpoints

### POST /query

- **Purpose**: Submit questions to the RAG system
- **Request Body**:
  ```json
  {
    "question": "string",
    "conversation_id": "string" (optional)
  }
  ```
- **Response**:
  - Answer grounded in retrieved documents
  - Source documents with relevance scores
  - Confidence metrics
- **Rate Limit**: 100 requests per minute
- **Authentication**: Required (Bearer token)

### POST /upload

- **Purpose**: Ingest new documents into the knowledge base
- **Supported Formats**: PDF, TXT, MD, DOCX
- **Maximum File Size**: 10MB
- **Bulk Upload**: Up to 5 files simultaneously
- **Processing**: Automatic text extraction and embedding generation
- **Response**: Document IDs and processing status

### GET /health

- **Purpose**: System health check
- **Returns**: System status, model availability, index stats
- **No Authentication Required**

## Administrative Endpoints

### POST /feedback

- **Purpose**: Submit user feedback on answers
- **Tracks**: Answer quality, relevance, helpfulness

### GET /stats

- **Purpose**: Retrieve system usage statistics
- **Admin Access Required**
