from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from elasticsearch import Elasticsearch, helpers
from PyPDF2 import PdfReader
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging
from pprint import pformat
from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import math

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Elasticsearch with security settings
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "rdCqw2vPytV83FNkrUHO"),
        verify_certs=False  # For development only
    )
    if not es.ping():
        raise ValueError("Could not connect to Elasticsearch")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    es = None

# Initialize OpenAI client (new way)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def chunk_document(text: str, chunk_size: int = 1024, overlap: int = 256) -> List[dict]:
    """
    Split document into overlapping chunks optimized for RAG.
    chunk_size=1024 for better context
    overlap=256 ensures context continuity
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n+', ' ', text)
    
    # Get sentences and their lengths
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        
        # Don't split in the middle of a sentence if it's reasonably sized
        if sentence_length > chunk_size * 1.5:
            # Split long sentences at punctuation or spaces
            sub_sentences = re.split(r'[,;:]', sentence)
            for sub_sent in sub_sentences:
                if len(sub_sent.strip()) > 0:
                    current_chunk.append(sub_sent.strip())
                    current_length += len(sub_sent)
                
                if current_length >= chunk_size:
                    _create_and_append_chunk(current_chunk, chunks, text)
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[-1])
        else:
            if current_length + sentence_length > chunk_size:
                _create_and_append_chunk(current_chunk, chunks, text)
                # Keep last 2-3 sentences for overlap
                overlap_sentences = current_chunk[-2:]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk if it exists
    if current_chunk:
        _create_and_append_chunk(current_chunk, chunks, text)
    
    return chunks

def _create_and_append_chunk(sentences: List[str], chunks: List[dict], full_text: str) -> None:
    """Helper function to create chunk with metadata"""
    chunk_text = ' '.join(sentences)
    start_idx = full_text.index(sentences[0])
    end_idx = full_text.index(sentences[-1]) + len(sentences[-1])
    
    chunks.append({
        'content': chunk_text,
        'length': len(chunk_text),
        'start_idx': start_idx,
        'end_idx': end_idx,
        'sentence_count': len(sentences)
    })

# Add root route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Add file upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process and index the file
            if process_file(file_path, filename):
                return jsonify({
                    "success": True,
                    "message": "File uploaded and processed successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Error processing file"
                }), 500
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error: {str(e)}"
            }), 500
    
    return jsonify({
        "success": False,
        "message": "File type not allowed"
    }), 400

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Get embeddings for a batch of texts"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts,
            encoding_format="float"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings batch: {e}")
        raise

def process_chunks_with_embeddings(chunks: List[dict], batch_size: int = 100) -> List[dict]:
    """Process chunks in batches to get embeddings efficiently"""
    # Prepare batches
    chunk_texts = [chunk['content'] for chunk in chunks]
    total_chunks = len(chunk_texts)
    num_batches = math.ceil(total_chunks / batch_size)
    
    processed_chunks = []
    
    logger.info(f"Processing {total_chunks} chunks in {num_batches} batches")
    
    for i in range(0, total_chunks, batch_size):
        batch_texts = chunk_texts[i:i + batch_size]
        logger.info(f"Processing batch {(i//batch_size)+1}/{num_batches}")
        
        # Get embeddings for the batch
        batch_embeddings = get_embeddings_batch(batch_texts)
        
        # Combine chunks with their embeddings
        for j, embedding in enumerate(batch_embeddings):
            chunk_idx = i + j
            if chunk_idx < len(chunks):
                processed_chunk = chunks[chunk_idx].copy()
                processed_chunk['embedding'] = embedding
                processed_chunks.append(processed_chunk)
    
    return processed_chunks

def process_file(file_path: str, filename: str) -> bool:
    """Process file with embeddings for semantic search"""
    try:
        if not es:
            raise Exception("Elasticsearch is not connected")

        # Extract text content
        if filename.endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        if not content:
            return False

        # Create document chunks
        chunks = chunk_document(content)
        
        # Create or update index with dense_vector mapping
        if not es.indices.exists(index="documents"):
            mapping = {
                "mappings": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 3072,  # text-embedding-3-large dimension
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "file_type": {"type": "keyword"},
                                "chunk_size": {"type": "integer"},
                                "total_document_size": {"type": "integer"}
                            }
                        }
                    }
                }
            }
            es.indices.create(index="documents", body=mapping)

        # Process chunks with embeddings in batches
        processed_chunks = process_chunks_with_embeddings(chunks)

        # Prepare bulk indexing actions
        actions = []
        for chunk in processed_chunks:
            actions.append({
                "_index": "documents",
                "_source": {
                    "filename": filename,
                    "content": chunk['content'],
                    "embedding": chunk['embedding'],
                    "metadata": {
                        "file_type": filename.split('.')[-1].lower(),
                        "chunk_size": chunk['length'],
                        "total_document_size": len(content)
                    }
                }
            })

        # Bulk index with progress logging
        logger.info(f"Starting bulk indexing of {len(chunks)} chunks for {filename}")
        success, failed = helpers.bulk(es, actions, stats_only=True)
        logger.info(f"Bulk indexing completed: {success} successful, {failed} failed")

        return True

    except Exception as e:
        logger.exception(f"Error processing file: {e}")
        return False

@app.route('/query', methods=['POST'])
def query():
    try:
        if not es:
            logger.error("Elasticsearch is not connected")
            return jsonify({
                "success": False,
                "response": "Search service is currently unavailable"
            }), 503

        data = request.get_json()
        user_query = data.get('query', '')

        logger.info(f"\n{'='*50}\nNew Query Request\n{'='*50}")
        logger.info(f"User Query: {user_query}")

        if not user_query:
            logger.warning("Empty query received")
            return jsonify({"success": False, "response": "No query provided"}), 400

        # Get embedding for query
        query_embedding = get_embedding(user_query)

        # Semantic search using kNN
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 5,
                "num_candidates": 50
            },
            "_source": ["filename", "content", "metadata"],
            "highlight": {
                "fields": {
                    "content": {
                        "type": "unified",
                        "number_of_fragments": 3,
                        "fragment_size": 300
                    }
                }
            }
        }

        logger.info(f"\nElasticsearch Query:\n{pformat(search_body)}")

        # Get search results
        es_response = es.search(index="documents", body=search_body)
        es_response_dict = es_response.body if hasattr(es_response, 'body') else dict(es_response)

        # Process search results
        context = []
        if es_response_dict["hits"]["hits"]:
            for hit in es_response_dict["hits"]["hits"]:
                file_ref = f"[From {hit['_source']['filename']}]"
                
                # Use highlighted content if available, otherwise use full content
                content = hit["highlight"]["content"][0] if "highlight" in hit else hit["_source"]["content"]
                context.append(f"{file_ref}\n{content}")
                
                logger.info(f"\nMatch Score: {hit['_score']}")
                logger.info(f"Content: {content[:200]}...")

        # Combine context with document references
        combined_context = "\n\n".join(context)
        
        logger.info(f"\nCombined Context:\n{combined_context}")

        if not combined_context.strip():
            logger.warning("Empty context after processing")
            return jsonify({
                "success": True,
                "response": "I found some documents but couldn't extract relevant context. Please try rephrasing your question.",
                "debug": {
                    "elasticsearch_query": search_body,
                    "elasticsearch_response": es_response_dict,
                    "context": "",
                    "openai_messages": []
                }
            }), 200

        # Enhanced prompt for better context utilization
        messages = [
            {"role": "system", "content": """You are an insurance claims assistant. When answering, Cite specific sections from the documents when possible, If information is not in the context, clearly state that, If you find conflicting information, point it out"""},
            {"role": "user", "content": f"Question: {user_query}\n\nContext:\n{combined_context}"}
        ]

        logger.info(f"\nOpenAI Messages:\n{pformat(messages)}")

        # Update OpenAI call to use new client
        openai_response = client.chat.completions.create(
            model="gpt-4o",  # or your preferred model
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        # Convert OpenAI response to dict (new format)
        openai_response_dict = {
            "choices": [{
                "message": {
                    "content": openai_response.choices[0].message.content,
                    "role": openai_response.choices[0].message.role
                },
                "finish_reason": openai_response.choices[0].finish_reason
            }],
            "created": openai_response.created,
            "model": openai_response.model,
            "usage": {
                "prompt_tokens": openai_response.usage.prompt_tokens,
                "completion_tokens": openai_response.usage.completion_tokens,
                "total_tokens": openai_response.usage.total_tokens
            }
        }

        return jsonify({
            "success": True, 
            "response": openai_response.choices[0].message.content,
            "debug": {
                "elasticsearch_query": search_body,
                "elasticsearch_response": es_response_dict,
                "context": combined_context,
                "openai_messages": messages,
                "openai_response": openai_response_dict
            }
        }), 200

    except Exception as e:
        logger.exception("Error processing query")
        return jsonify({
            "success": False, 
            "response": "An error occurred while processing your query",
            "error": str(e)
        }), 500

def assemble_context(hits: List[dict], full_text: bool = False) -> str:
    """Assemble context from search results with smart ordering"""
    # Group chunks by document
    doc_chunks = {}
    for hit in hits:
        filename = hit['_source']['filename']
        if filename not in doc_chunks:
            doc_chunks[filename] = []
        doc_chunks[filename].append(hit)
    
    context_parts = []
    
    # Process each document's chunks
    for filename, chunks in doc_chunks.items():
        # Add document header
        context_parts.append(f"\n=== Document: {filename} ===\n")
        
        # Add relevant chunks with their context
        for chunk in chunks:
            score = chunk['_score']
            
            # Add relevance score
            context_parts.append(f"[Relevance: {score:.2f}]")
            
            # Add highlighted content
            if "highlight" in chunk:
                for highlight in chunk["highlight"]["content"]:
                    context_parts.append(highlight)
            
            context_parts.append("")  # Empty line between chunks
    
    return "\n".join(context_parts)

# Add this route after your other routes
@app.route('/reset', methods=['POST'])
def reset_documents():
    try:
        # Delete all files in the upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Delete all documents from Elasticsearch
        if es:
            es.indices.delete(index='documents', ignore=[404])
            # Recreate the index with proper mappings
            es.indices.create(
                index='documents',
                body={
                    'mappings': {
                        'properties': {
                            'content': {'type': 'text'},
                            'filename': {'type': 'keyword'},
                            'embedding': {
                                'type': 'dense_vector',
                                'dims': 3072,  # for text-embedding-3-large
                                'index': True,
                                'similarity': 'cosine'
                            }
                        }
                    }
                },
                ignore=[400]  # Ignore error if index already exists
            )
            
        return jsonify({
            'success': True,
            'message': 'All documents have been cleared'
        })
        
    except Exception as e:
        logger.exception("Error in reset_documents")
        return jsonify({
            'success': False,
            'message': f'Error clearing documents: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
