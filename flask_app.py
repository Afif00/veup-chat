from flask import Flask, request, jsonify
import threading
import json
import numpy as np
import threading
import logging
from utils import generate_embedding
# Initialize models globally to load them once


app = Flask(__name__)

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def readiness_check_2():
    return jsonify({'status': 'Ready'}), 200

@app.route('/rerank', methods=['POST'])
def rerank():
    try:
        data = None
        try:
            # Attempt to parse as JSON first
            data = request.json
            if data is None:
                # If request.json was empty, try decoding raw data as JSON string
                text_str = request.data.decode("utf-8")
                data = json.loads(text_str)
            # The entire request body is the JSON object Weaviate sends
            text = data
        except Exception as e:
            # Fallback for unexpected data formats
            try:
                text_str = request.data.decode("utf-8")
                text = json.loads(text_str)
            except Exception as e_inner:
                print(f"Error parsing request data: {e_inner}")
                return jsonify({'error': f"Could not parse request body: {e_inner}"}), 400

        # Validate expected input format from Weaviate
        if not isinstance(text, dict) or 'query' not in text or 'documents' not in text:
            print(f"Invalid input format. Expected dict with 'query' and 'documents'. Got: {text}")
            return jsonify({'error': "Invalid input format. Expected a dictionary with 'query' and 'documents'."}), 400

        query = text['query']
        documents = text['documents']

        if not documents:
            # Return an empty list of scores if no documents are provided for reranking
            # This handles cases where Weaviate might send an empty list, preventing errors
            return jsonify({'scores': []})

        # Prepare pairs for the reranker model
        compares = [(query, doc) for doc in documents]
        
        # Compute scores using the FlagReranker model
        scores = reranker.compute_score(compares)

        # Convert scores (typically a NumPy array or tensor) to a Python list
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores

        # Construct the response in the format Weaviate's reranker-transformers module expects
        # This includes the original document text and its computed score
        reranked_results = []
        for i, doc_text in enumerate(documents):
            score = scores_list[i]
            reranked_results.append({
                "document": doc_text,  # Include the original document text
                "score": float(score)  # Use "score" as the key, ensuring float type
            })

        return jsonify({'scores': reranked_results}) # Top-level key is "scores" (plural)

    except Exception as e:
        print(f"Unhandled error in /rerank: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vectors', methods=['POST'])
def vectorize():
    try:
        # Parse JSON strictly; fail if body is not JSON
        payload = request.get_json(force=True, silent=False)

        # Expect {"text": ["...", "..."]} or {"text": "..."}
        texts = payload.get("text")
        if texts is None:
            return jsonify({"error": 'Body must include key "text".'}), 400

        # Normalize to a list of strings
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": 'Key "text" must be a string or list of strings.'}), 400

        # Generate embeddings (one vector per input string)
        vectors = generate_embedding(texts)

        # --- Normalize return type to list-of-lists of floats ---
        import numpy as np

        # If a single vector is returned for single input, wrap it
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], (np.floating, float, int)):
            vectors = [vectors]  # single vector -> list-of-lists

        # Convert any numpy elements inside
        def to_float_list(v):
            if isinstance(v, np.ndarray):
                v = v.astype(float).tolist()
            else:
                v = [float(x) for x in v]
            return v

        if not isinstance(vectors, list):
            return jsonify({"error": "generate_embedding must return a list/list-of-lists"}), 500
        if len(vectors) == len(texts) and vectors and isinstance(vectors[0], (list, np.ndarray)):
            vectors = [to_float_list(v) for v in vectors]
        elif len(texts) == 1:
            vectors = [to_float_list(vectors)]
        else:
            return jsonify({"error": "Embedding count does not match input texts."}), 500

        return jsonify({"vectors": vectors}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
app.logger.disabled = True
# Get the Flask app's logger
log = logging.getLogger('werkzeug')
# Set logging level (ERROR or CRITICAL suppresses routing logs)
log.setLevel(logging.ERROR)
def run_app():
    app.run(host='0.0.0.0', port=5009, debug = False)

flask_thread = threading.Thread(target=run_app)
flask_thread.start()
