from flask import Flask, request, jsonify
import os
import json
from azure.storage.blob import BlobServiceClient
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
import threading
import re
import uuid
from datetime import datetime

app = Flask(__name__)

# Azure Blob Storage setup (keeping this for file access if needed)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Elasticsearch setup
ES_ENDPOINT = os.getenv('ES_ENDPOINT')
ES_API_KEY = os.getenv('ES_API_KEY')
ES_USERNAME = "elastic"
ES_PASSWORD =os.getenv('ES_PASSWORD')
ES_INDEX_NAME = "weez-documents"

# Initialize Elasticsearch client
# Using API key authentication
es_client = Elasticsearch(
    ES_ENDPOINT,
    api_key=ES_API_KEY,
    verify_certs=True
)

# Alternatively, use username/password authentication if API key doesn't work
if not es_client.ping():
    es_client = Elasticsearch(
        ES_ENDPOINT,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=True
    )

# Thread-local storage
thread_local = threading.local()


def create_es_index_if_not_exists(force_recreate=False):
    """
    Create the Elasticsearch index if it doesn't exist with 3072-dimensional vectors

    Args:
        force_recreate (bool): If True, will delete and recreate the index even if it exists
    """
    try:
        # Define the required dimension for vectors
        REQUIRED_VECTOR_DIMS = 3072

        # Check if index exists
        index_exists = es_client.indices.exists(index=ES_INDEX_NAME)

        if index_exists and not force_recreate:
            print(f"Index {ES_INDEX_NAME} already exists")

            # Check if dimensions match what we need
            try:
                mapping = es_client.indices.get_mapping(index=ES_INDEX_NAME)
                current_dims = mapping[ES_INDEX_NAME]['mappings']['properties']['content_vector'].get('dims', 0)

                if current_dims != REQUIRED_VECTOR_DIMS:
                    print(f"WARNING: Index exists but has {current_dims} dimensions instead of {REQUIRED_VECTOR_DIMS}")
                    print("You should recreate the index with the correct dimensions by calling /recreate-index")
                    return False
                else:
                    print(f"Index has the correct dimension: {current_dims}")
                    return True
            except Exception as e:
                print(f"Error checking vector dimensions: {str(e)}")
                return False

        # Delete the index if it exists and we're forcing recreation
        if index_exists and force_recreate:
            es_client.indices.delete(index=ES_INDEX_NAME)
            print(f"Deleted existing index: {ES_INDEX_NAME}")

        # Define index mapping with vector search capabilities
        index_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "file_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_path": {"type": "text"},
                    "content": {"type": "text"},
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": REQUIRED_VECTOR_DIMS,  # Set to 3072 dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {"type": "date"}
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0  # Reduce to 0 for development, increase for production
                }
            }
        }

        # Create the index
        es_client.indices.create(index=ES_INDEX_NAME, body=index_mapping)
        print(f"Created index: {ES_INDEX_NAME} with {REQUIRED_VECTOR_DIMS} dimensions")
        return True

    except Exception as e:
        print(f"Error creating/checking Elasticsearch index: {str(e)}")
        raise

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get request data
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')

        # Optional parameters for search tuning
        top_k = data.get('top_k', 10)  # Number of results to return
        min_score = data.get('min_score', 0.1)  # Minimum similarity threshold

        # Validate input
        if not query or not user_id:
            return jsonify({"error": "Query and user_id are required."}), 400

        print(f"[DEBUG] Searching for query: '{query}' for user: '{user_id}'")

        # Generate embedding for the query text
        embedding_client = AzureOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            api_version="2024-02-01",
            azure_endpoint="https://weez-openai-resource.openai.azure.com/"
        )

        try:
            # Generate query embedding - make sure your model generates 3072-dimensional embeddings
            embedding_response = embedding_client.embeddings.create(
                model="text-embedding-3-large",  # Ensure this model outputs 3072-dim vectors
                input=query
            )
            query_embedding = embedding_response.data[0].embedding

            # Verify embedding dimension
            if len(query_embedding) != 3072:
                print(f"[WARNING] Query embedding dimension is {len(query_embedding)}, expected 3072")

            print(f"[DEBUG] Generated embedding for query with dimension: {len(query_embedding)}")
        finally:
            embedding_client.close()

        # Optimized vector search query
        vector_search_query = {
            "size": top_k,
            "_source": ["file_name", "file_path", "id"],
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"user_id": user_id}}
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        # Execute search with profile flag for performance metrics
        print(f"[DEBUG] Executing vector search query")
        search_results = es_client.search(
            index=ES_INDEX_NAME,
            body=vector_search_query,
            profile=True  # Enable profiling for performance analysis
        )

        # Extract performance metrics
        search_time_ms = search_results.get('took', 0)
        print(f"[DEBUG] Search completed in {search_time_ms}ms, found {len(search_results['hits']['hits'])} results")

        # Process results
        matches = []
        for hit in search_results["hits"]["hits"]:
            # Elasticsearch returns scores typically between 0-2 for cosine similarity + 1.0
            # Normalize to 0-1 range
            raw_score = hit["_score"]
            normalized_score = (raw_score - 1.0) / 1.0  # Now between 0-1

            # Filter by similarity threshold
            if normalized_score > min_score:
                match_info = {
                    "id": hit["_source"].get("id", ""),
                    "file_name": hit["_source"].get("file_name", ""),
                    "file_path": hit["_source"].get("file_path", ""),
                    "similarity_score": float(normalized_score)
                }
                print(f"[DEBUG] Match found: {match_info['file_name']} with score {normalized_score}")
                matches.append(match_info)

        # Sort and return top matches
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return results
        if matches:
            print(f"[DEBUG] Returning {len(matches)} matches")
            return jsonify({
                "matches": matches,
                "total_matches": len(matches),
                "search_time_ms": search_time_ms
            }), 200
        else:
            print("[DEBUG] No matches found")
            return jsonify({
                "message": "No matches found",
                "search_time_ms": search_time_ms
            }), 200

    except Exception as e:
        print(f"[CRITICAL] Search error: {str(e)}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route('/index', methods=['POST'])
def index_document():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        file_path = data.get('file_path')
        content = data.get('content')
        file_name = data.get('file_name', file_path.split('/')[-1] if file_path else 'unknown')

        # You can also accept pre-computed embeddings directly
        pre_computed_embedding = data.get('embedding')

        # Validate input
        if not user_id or not (file_path or content):
            return jsonify({"error": "User ID and either file path or content are required."}), 400

        # Generate embeddings for content if not provided
        if not pre_computed_embedding:
            embedding_client = AzureOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_version="2024-02-01",
                azure_endpoint="https://weez-openai-resource.openai.azure.com/"
            )

            try:
                embedding_response = embedding_client.embeddings.create(
                    model="text-embedding-3-large",  # Make sure this outputs 3072-dim vectors
                    input=content
                )
                content_embedding = embedding_response.data[0].embedding

                # Verify embedding dimension
                if len(content_embedding) != 3072:
                    print(f"[WARNING] Generated embedding dimension is {len(content_embedding)}, expected 3072")
            finally:
                embedding_client.close()
        else:
            content_embedding = pre_computed_embedding
            # Verify embedding dimension
            if len(content_embedding) != 3072:
                return jsonify(
                    {"error": f"Provided embedding has {len(content_embedding)} dimensions, expected 3072"}), 400

        # Create document for indexing
        document = {
            "id": f"{user_id}_{file_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "file_name": file_name,
            "file_path": file_path,
            "content": content,
            "content_vector": content_embedding,
            "created_at": datetime.now().isoformat()
        }

        # Upload to index
        result = es_client.index(index=ES_INDEX_NAME, body=document)

        if result['result'] == 'created' or result['result'] == 'updated':
            return jsonify({
                "message": "Document indexed successfully",
                "document_id": document["id"]
            }), 200
        else:
            return jsonify({"error": "Failed to index document"}), 500

    except Exception as e:
        print(f"[CRITICAL] Indexing error: {str(e)}")
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500


# Add a utility function to bulk load documents from blob storage
@app.route('/bulk-index', methods=['POST'])
def bulk_index_from_blob():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        batch_size = data.get('batch_size', 50)
        force_recreate_index = data.get('force_recreate_index', False)
        skip_dimension_check = data.get('skip_dimension_check', False)

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        print(f"[DEBUG] Starting bulk indexing for user: {user_id}")

        # Ensure index exists with correct dimensions
        index_ok = create_es_index_if_not_exists(force_recreate=force_recreate_index)
        if not index_ok:
            return jsonify({
                "error": "Index has incorrect dimensions. Set force_recreate_index=true to recreate it."
            }), 400

        # Access blob storage to get documents
        container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
        blobs = list(container_client.list_blobs(name_starts_with=f"{user_id}/"))

        print(f"[DEBUG] Found {len(blobs)} blobs for user {user_id}")

        documents_indexed = 0
        failed_documents = 0
        dimension_mismatches = 0
        batch_actions = []

        for blob in blobs:
            try:
                # Get blob content
                blob_client = container_client.get_blob_client(blob.name)
                blob_content = blob_client.download_blob().readall().decode('utf-8')
                blob_data = json.loads(blob_content)

                # Extract filename without path and extension
                file_name = blob.name.split('/')[-1].replace('.json', '')

                # Ensure embeddings exist
                if "embeddings" not in blob_data or not blob_data["embeddings"]:
                    print(f"[WARN] No embeddings found for {blob.name}, skipping")
                    failed_documents += 1
                    continue

                # Check embedding dimension if not skipping check
                if not skip_dimension_check:
                    embedding_dim = len(blob_data["embeddings"])
                    if embedding_dim != 3072:
                        print(
                            f"[WARN] Embedding dimension mismatch for {blob.name}: expected 3072, got {embedding_dim}")
                        dimension_mismatches += 1
                        failed_documents += 1
                        continue

                # Generate a unique ID that includes user_id
                doc_id = f"{user_id}_{file_name}_{uuid.uuid4().hex[:8]}"

                # Create document
                document = {
                    "id": doc_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "file_path": blob_data.get("file_path", ""),
                    "content": blob_data.get("content", f"File {file_name}"),
                    "content_vector": blob_data["embeddings"],
                    "created_at": datetime.now().isoformat()
                }

                # Add to bulk actions
                batch_actions.append({"index": {"_index": ES_INDEX_NAME, "_id": doc_id}})
                batch_actions.append(document)

                # Process in smaller batches to avoid memory issues
                if len(batch_actions) >= batch_size * 2:  # *2 because each doc is 2 actions
                    bulk_response = es_client.bulk(body=batch_actions)

                    # Check for errors
                    if bulk_response.get('errors', False):
                        for idx, item in enumerate(bulk_response['items']):
                            if 'error' in item.get('index', {}):
                                failed_documents += 1
                                error_reason = item['index'].get('error', {}).get('reason', 'Unknown error')
                                print(f"[ERROR] Failed to index document: {error_reason}")

                                # Check if the error is related to dimension mismatch
                                if 'dimension' in error_reason.lower() and 'mapping' in error_reason.lower():
                                    dimension_mismatches += 1
                            else:
                                documents_indexed += 1
                    else:
                        documents_indexed += len(batch_actions) // 2

                    batch_actions = []

            except Exception as e:
                print(f"[ERROR] Error processing blob {blob.name}: {str(e)}")
                failed_documents += 1
                continue

        # Process any remaining documents
        if batch_actions:
            bulk_response = es_client.bulk(body=batch_actions)

            # Check for errors
            if bulk_response.get('errors', False):
                for item in bulk_response['items']:
                    if 'error' in item.get('index', {}):
                        failed_documents += 1
                        error_reason = item['index'].get('error', {}).get('reason', 'Unknown error')
                        if 'dimension' in error_reason.lower() and 'mapping' in error_reason.lower():
                            dimension_mismatches += 1
                    else:
                        documents_indexed += 1
            else:
                documents_indexed += len(batch_actions) // 2

        # Refresh the index to make documents immediately searchable
        es_client.indices.refresh(index=ES_INDEX_NAME)

        response_message = f"Bulk indexing complete. {documents_indexed} documents indexed. {failed_documents} documents failed."
        if dimension_mismatches > 0:
            response_message += f" {dimension_mismatches} failed due to dimension mismatch."
            response_message += " Use force_recreate_index=true to recreate the index with proper dimensions."

        return jsonify({
            "message": response_message,
            "indexed": documents_indexed,
            "failed": failed_documents,
            "dimension_mismatches": dimension_mismatches
        }), 200

    except Exception as e:
        print(f"[CRITICAL] Bulk indexing error: {str(e)}")
        return jsonify({"error": f"Bulk indexing failed: {str(e)}"}), 500

@app.route('/document-count', methods=['GET'])
def get_document_count():
    user_id = request.args.get('user_id')
    count_query = {
        "query": {
            "term": {
                "user_id": user_id
            }
        }
    }
    result = es_client.count(index=ES_INDEX_NAME, body=count_query)
    return jsonify({"count": result["count"]}), 200


# Add a utility to delete all documents for a user
@app.route('/delete-user-documents', methods=['POST'])
def delete_user_documents():
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        # Delete query
        delete_query = {
            "query": {
                "term": {
                    "user_id": user_id
                }
            }
        }

        result = es_client.delete_by_query(index=ES_INDEX_NAME, body=delete_query)

        return jsonify({
            "message": f"Successfully deleted {result['deleted']} documents for user {user_id}"
        }), 200

    except Exception as e:
        print(f"[CRITICAL] Delete error: {str(e)}")
        return jsonify({"error": f"Document deletion failed: {str(e)}"}), 500


def extract_entities_from_ner(ner_results):
    """Extract meaningful entities from NER tagged results"""
    entities = []
    current_entity = []
    current_tag = None

    for line in ner_results:
        # Skip empty lines
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) >= 2:
            word = parts[0]
            tag = parts[-1]  # Last part is the tag

            # Check if this is an entity
            if tag != 'O':
                # New entity type
                if tag.startswith('B-') or (current_tag and current_tag != tag):
                    # Save previous entity if exists
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []

                    # Start new entity
                    current_entity.append(word)
                    current_tag = tag
                # Continuation of current entity
                elif tag.startswith('I-'):
                    current_entity.append(word)
                    current_tag = tag
            else:
                # End of an entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                    current_tag = None

    # Add the last entity if exists
    if current_entity:
        entities.append(' '.join(current_entity))

    return entities


def perform_ner(query):
    # Azure OpenAI settings
    endpoint = "https://weez-openai-resource.openai.azure.com/"
    api_key = os.getenv('OPENAI_API_KEY')
    api_version = "2024-12-01-preview"
    deployment = "gpt-4o"  # Using GPT-4 as in the original code

    # Create client with API key
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

    messages = [
        {
            "role": "system",
            "content": """You are a Named Entity Recognition (NER) expert. Your task is to analyze a given text and identify specific \
                entities based on the following tags:
                - **B-DOC / I-DOC**: Represents document types like PDF, DOCX, XLSX, PPTX, and their variants. It also includes generic document mentions such as "report", "presentation", or "excel sheet".
                - **B-PER / I-PER**: Represents names of people such as "John Watson", "Elon Musk", or "Mary".
                - **B-TOP / I-TOP**: Represents topics or subjects such as "natural disaster", "machine learning", or "climate change".
                - **B-DATE / I-DATE**: Represents relative or absolute dates, such as "two months ago", "on April 24th", "one year ago", "24/10/24", or "yesterday".
                Each word in the input should be tagged as:
                - `B-TAG`: Beginning of an entity.
                - `I-TAG`: Continuation of an entity.
                - `O`: Not part of any entity.
                ### Additional Guidelines:
                1. The query can be formal or informal.
                   - Example (Formal): "Please provide me with the PDF about natural disasters."
                   - Example (Informal): "Give me a pdf on natural disasters sent by John."
                   - Example (Vague): "Show me the document which contains data on natural disasters."
                   - Example (Direct Data): "Different kinds of natural disasters."
                2. Even if the query directly mentions file content or metadata (like topics, dates, or names), identify and tag all relevant entities.
                ### Example:
                **Input**:
                  "Give me a pdf sent by John on the topic of natural disaster two months ago."
                **Output**:
                  ```plaintext
                Give  O
                me    O
                a     O
                pdf   B-DOC
                sent  O
                by    O
                John  B-PER
                on    O
                the   O
                topic O
                of    O
                natural B-TOP
                disaster I-TOP
                two   B-DATE
                months I-DATE
                ago    I-DATE
                ```"""
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # Call Azure OpenAI API for NER using new client syntax
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        max_tokens=500  # Increased max tokens to handle longer queries
    )

    # Extract and return the response content
    result = response.choices[0].message.content.strip()

    # Check if result contains actual NER output (looking for at least some tagged words)
    if not re.search(r'\b[BI]-[A-Z]+\b', result):
        print(f"NER result might not be in expected format: {result}")
        # Try to parse it anyway or return empty

    # Return the response content split by newlines
    return result.split("\n")


@app.route('/test-vector-search', methods=['GET'])
def test_vector_search():
    try:
        # Create a test vector with 3072 dimensions (all 0.5)
        test_vector = [0.5] * 3072

        user_id = request.args.get('user_id')

        # Build query
        vector_query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"user_id": user_id}} if user_id else {"match_all": {}}
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {"query_vector": test_vector}
                    }
                }
            }
        }

        # Execute search with profiling
        start_time = datetime.now()
        result = es_client.search(index=ES_INDEX_NAME, body=vector_query, profile=True)
        search_time = (datetime.now() - start_time).total_seconds() * 1000  # ms

        # Get profiling data
        profile_data = result.get('profile', {})

        # Process and return top matches
        matches = []
        for hit in result["hits"]["hits"]:
            # Extract vector dimension if available
            vector_dim = len(hit["_source"].get("content_vector", [])) if "content_vector" in hit[
                "_source"] else "unknown"

            matches.append({
                "id": hit["_source"].get("id", ""),
                "user_id": hit["_source"].get("user_id", ""),
                "file_name": hit["_source"].get("file_name", ""),
                "score": hit["_score"],
                "vector_dimension": vector_dim
            })

        return jsonify({
            "message": "Vector search test completed",
            "total_hits": result["hits"]["total"]["value"] if "total" in result["hits"] else len(matches),
            "matches": matches,
            "performance": {
                "total_time_ms": search_time,
                "es_took_ms": result.get('took', 0)
            },
            "profile": profile_data
        }), 200

    except Exception as e:
        print(f"[ERROR] Vector search test error: {str(e)}")
        return jsonify({"error": f"Vector search test failed: {str(e)}"}), 500


# Add this utility endpoint to delete and recreate the index with correct dimensions
@app.route('/recreate-index', methods=['POST'])
def recreate_index():
    try:
        data = request.get_json()
        confirm = data.get('confirm', False)

        if not confirm:
            return jsonify({
                "warning": "This will delete all indexed documents. Set 'confirm':true to proceed."
            }), 400

        # Delete the existing index if it exists
        if es_client.indices.exists(index=ES_INDEX_NAME):
            es_client.indices.delete(index=ES_INDEX_NAME)
            print(f"[INFO] Deleted existing index: {ES_INDEX_NAME}")

        # Define new index mapping with correct dimensions
        index_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "file_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_path": {"type": "text"},
                    "content": {"type": "text"},
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": 3072,  # Set to 3072 dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {"type": "date"}
                }
            }
        }

        # Create the new index
        es_client.indices.create(index=ES_INDEX_NAME, body=index_mapping)
        print(f"[INFO] Created new index with 3072 dimensions: {ES_INDEX_NAME}")

        return jsonify({
            "message": f"Successfully recreated index {ES_INDEX_NAME} with 3072-dimensional vectors",
            "next_steps": "You'll need to reindex your documents"
        }), 200

    except Exception as e:
        print(f"[CRITICAL] Error recreating index: {str(e)}")
        return jsonify({"error": f"Failed to recreate index: {str(e)}"}), 500


# Alternative approach: Create a new index with a different name
@app.route('/create-new-index', methods=['POST'])
def create_new_index():
    try:
        data = request.get_json()
        new_index_name = data.get('new_index_name', f"{ES_INDEX_NAME}-3072")

        # Define new index mapping with correct dimensions
        index_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "file_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_path": {"type": "text"},
                    "content": {"type": "text"},
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": 3072,  # Set to 3072 dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {"type": "date"}
                }
            }
        }

        # Create the new index
        if not es_client.indices.exists(index=new_index_name):
            es_client.indices.create(index=new_index_name, body=index_mapping)
            print(f"[INFO] Created new index {new_index_name} with 3072 dimensions")

            return jsonify({
                "message": f"Successfully created new index {new_index_name} with 3072-dimensional vectors",
                "next_steps": "Update your application to use the new index name"
            }), 200
        else:
            return jsonify({
                "message": f"Index {new_index_name} already exists",
                "status": "no_action"
            }), 200

    except Exception as e:
        print(f"[CRITICAL] Error creating new index: {str(e)}")
        return jsonify({"error": f"Failed to create new index: {str(e)}"}), 500



if __name__ == '__main__':
    # Import datetime here to avoid circular imports
    from datetime import datetime

    # Create ES index when starting the server
    create_es_index_if_not_exists()
    app.run(debug=True)
