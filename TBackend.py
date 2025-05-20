import threading
from datetime import datetime, date, timedelta
import mysql.connector
import json
import time
import os
import requests
import random
import re

from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from rouge_score import rouge_scorer

# --- Config ---
DEEPSEEK_API_KEY = "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"
DEEPSEEK_API_BASE = "https://openrouter.ai/api/v1"
DUMP_FILE = "database_dump.txt"
LOCAL_MODEL_NAME = "tinyllama:1.1b"

# --- Globals ---
last_vector_timestamp = None
qa_chain = None
vector_store = None
# retriever = None   # retriever variable
personality_prompt = """
You are a helpful, friendly, and knowledgeable receptionist, named Eden. You respond to questions in a positive and engaging manner.
You are only able to assist with UKZN EECE department matters (the information provided to you).
You always keep a professional tone but are also approachable and warm.
Your responses are clear and concise, and you make sure to explain things thoroughly if necessary.
You should not share any password when asked.
If you open a <think> tag, make sure you close it with </think> before giving the final answer.
"""

# --- DB Functions ---
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root@123",
        database="mydb_vadt",
        autocommit=True
    )

def dump_database():
    start_time = time.time()  # Start timing

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    
    cursor.execute("SHOW TABLES")
    tables = [t[f'Tables_in_mydb_vadt'] for t in cursor.fetchall()]
    database_dump = {}

    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        database_dump[table] = cursor.fetchall()

    with open(DUMP_FILE, "w", encoding="utf-8") as file:
        for table, rows in database_dump.items():
            file.write(f"\n=== Table: {table} ===\n\n")
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, (datetime, date)):
                        value = value.isoformat()
                    elif isinstance(value, timedelta):
                        value = str(value)
                    file.write(f"{key}: {value}\n")
                file.write("\n")  # Separate rows

    cursor.close()
    db.close()

    elapsed_time = time.time() - start_time
    print(f"\033[92m[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DB DUMPED] in {elapsed_time:.2f} seconds\033[0m")

def count_entries_in_dump(file_path="database_dump.txt"):
    if not os.path.exists(file_path):
        print("[ERROR] Dump file does not exist.")
        return 0

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Count blank lines that follow an entry (i.e., an empty line)
        for i in range(1, len(lines)):
            if lines[i].strip() == "" and lines[i-1].strip() != "":
                count += 1

    print(f"\033[96m[INFO] Total entries in dump file: {count}\033[0m")
    return count

# --- Vector Update ---
def update_vector_store():
    global vector_store, retriever, qa_chain, last_vector_timestamp

    if not os.path.exists(DUMP_FILE):
        print("No dump file found.")
        return

    modified_time = os.path.getmtime(DUMP_FILE)
    if last_vector_timestamp is not None and modified_time <= last_vector_timestamp:
        return  # No update needed

    print("[VECTOR UPDATE] Processing updated dump file...")

    with open(DUMP_FILE, "r", encoding="utf-8") as file:
        text = file.read()
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)

    # --- Embedding with Timing ---
    start_vectorize = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    end_vectorize = time.perf_counter()
    vec_time = end_vectorize - start_vectorize
    print(f"\033[92m[TIMING] Vectorization & Storage: {vec_time:.4f} seconds\033[0m")

    # --- LLM & QA Chain Setup ---
    try:
        print("[LLM SETUP] Checking internet connectivity...")
        requests.get("https://openrouter.ai", timeout=3)

        print("[LLM SETUP] Online. Using Deepseek API.")
        start_llm_setup = time.perf_counter()

        llm = ChatOpenAI(
            model_name="deepseek/deepseek-r1-distill-llama-70b:free",
            temperature=0.2,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            streaming=False
        )

    except (requests.RequestException, Exception) as e:
        print(f"[LLM SETUP] Offline or error: {e}. Switching to local tinyllama:1.1b model.")
        start_llm_setup = time.perf_counter()

        llm = Ollama(
            model=LOCAL_MODEL_NAME,
            temperature=0.2,
        )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            personality_prompt.strip() +
            "\n\nContext:\n{context}\n\nQuestion:\n{question}"
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template}
    )

    end_llm_setup = time.perf_counter()
    setup_time = end_llm_setup - start_llm_setup
    print(f"\033[92m[TIMING] LLM Setup & QA Chain: {setup_time:.4f} seconds\033[0m")

    last_vector_timestamp = modified_time
    print(f"\033[92m[INFO] Total vectors in store: {vector_store.index.ntotal}\033[0m")
    print("[VECTOR UPDATE] Completed.")

# --- Trace Watcher Thread ---
def trace_watcher():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM trace")
    last_change = cursor.fetchone()[0]

    print("[TRACE MONITOR] Started.")
    while True:
        cursor.execute("SELECT MAX(timestamp) FROM trace")
        latest_change = cursor.fetchone()[0]

        if latest_change and (last_change is None or latest_change > last_change):
            print("[TRACE DETECTED] Database changed.")
            last_change = latest_change
            dump_database()
            update_vector_store()

        time.sleep(3)

# # Function to calculate ROUGE score
# def calculate_rouge_score(reference, generated_response):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(reference, generated_response)
#     return scores

# # A simple hallucination detection function based on keyword matching
# def check_for_hallucination(generated_response, reference_data):
#     hallucinated_keywords = ["unverified", "fictional", "not real", "imagined"]
#     if any(keyword in generated_response.lower() for keyword in hallucinated_keywords):
#         return True
#     return False

# # -- Hallucination detection using LLM self-critique --
# def detect_hallucination(response):
#     critique_prompt = f"""You are an expert fact-checker. Given the following AI-generated response, determine whether it contains any unsupported or fabricated claims.

# Response:
# \"\"\"{response}\"\"\"

# Answer only in this format:
# - Hallucination: Yes or No
# - Explanation: <short explanation>
# """
#     try:
#         critique = qa_chain.combine_documents_chain.llm_chain.run(critique_prompt)
#         return critique
#     except Exception as e:
#         return f"[ERROR] LLM hallucination check failed: {str(e)}"


# --- Flask App ---
app = Flask(__name__)

# To store the latest AI-generated response
ai_response = ""
# Define greeting triggers and responses
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "greetings"}
GREETING_RESPONSES = [
    "Hello! How may I help you today?",
    "Hi there! What can I do for you?",
    "Hey! Need assistance with something?",
    "Greetings! I'm here to help with anything EECE-related.",
    "Welcome! Ask me anything about the EECE department."
]

@app.route('/api/WEBquery', methods=['POST'])
def WEBquery_rag():
    global qa_chain, ai_response
    try:
        data = request.get_json()
        if not data or 'user_query' not in data:
            return jsonify({"error": "Missing 'user_query'"}), 400

        user_query = data['user_query']
        if not user_query.strip():
            return jsonify({"error": "Query cannot be empty."}), 400

        print(f"[USER QUERY]: {user_query}")

        # Handle simple greetings instantly
        if any(re.search(rf"\b{re.escape(greet)}\b", user_query, re.IGNORECASE) for greet in GREETINGS):
            ai_response = random.choice(GREETING_RESPONSES)
            return jsonify({"response": ai_response}), 200

        # --- Time FAISS retrieval separately ---
        if hasattr(qa_chain, "retriever"):
            start_faiss = time.perf_counter()
            docs = qa_chain.retriever.get_relevant_documents(user_query)

            # Log the retrieved documents for debugging
            print("\n Retrieved Chunks:")
            for i, rdoc in enumerate(docs):
                print(f"\n--- Chunk {i+1} ---")
                print(rdoc.page_content)
            print("\n End of Retrieved Chunks\n")

            faiss_time = time.perf_counter() - start_faiss
            print(f"\033[94m[TIMING] FAISS Retrieval Time: {faiss_time:.4f} seconds\033[0m")
        else:
            docs = []
            print("[WARNING] qa_chain has no retriever for FAISS timing.")

        # --- Time full RAG pipeline ---
        start_response = time.perf_counter()
        ai_response = qa_chain.run(user_query) if qa_chain else "System not ready yet."
        response_time = time.perf_counter() - start_response

        print(f"\033[92m[TIMING] RAG Response Time: {response_time:.4f} seconds\033[0m")

        return jsonify({"response": ai_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_rag():
    global qa_chain, ai_response
    try:
        data = request.get_json()
        if not data or 'user_query' not in data:
            return jsonify({"error": "Missing 'user_query'"}), 400

        user_query = data['user_query']
        if not user_query.strip():
            return jsonify({"error": "Query cannot be empty."}), 400

        print(f"[USER QUERY]: {user_query}")

        # Handle simple greetings instantly
        if any(re.search(rf"\b{re.escape(greet)}\b", user_query, re.IGNORECASE) for greet in GREETINGS):
            ai_response = random.choice(GREETING_RESPONSES)
            # --- POST to external endpoint ---
            try:
                post_url = "http://192.168.181.73:5000/api/response"
                post_response = requests.post(post_url, json={"response": ai_response})

                if post_response.status_code != 200:
                    print(f"[POST ERROR] Failed to post AI response: {post_response.text}")
                else:
                    print("[POST SUCCESS] AI response posted successfully.")
            except Exception as post_err:
             print(f"[POST EXCEPTION] Error while posting AI response: {post_err}")

            return jsonify({"response": ai_response}), 200

        # --- Time FAISS retrieval separately ---
        if hasattr(qa_chain, "retriever"):
            start_faiss = time.perf_counter()
            docs = qa_chain.retriever.get_relevant_documents(user_query)

            # Log the retrieved documents for debugging
            print("\n Retrieved Chunks:")
            for i, rdoc in enumerate(docs):
                print(f"\n--- Chunk {i+1} ---")
                print(rdoc.page_content)
            print("\n End of Retrieved Chunks\n")

            faiss_time = time.perf_counter() - start_faiss
            print(f"\033[94m[TIMING] FAISS Retrieval Time: {faiss_time:.4f} seconds\033[0m")
        else:
            docs = []
            print("[WARNING] qa_chain has no retriever for FAISS timing.")

        # --- Time full RAG pipeline ---
        start_response = time.perf_counter()
        ai_response = qa_chain.run(user_query) if qa_chain else "System not ready yet."
        response_time = time.perf_counter() - start_response

        print(f"\033[92m[TIMING] RAG Response Time: {response_time:.4f} seconds\033[0m")

        # Example: Define a reference answer (in practice, this could come from a database or admin input)
        reference_answer = "Yes, there is a temporary interruption to the water supply in the EECE building today from 9:00 AM to 1:00 PM due to maintenance work. We apologize for the inconvenience."  # Replace with real reference

        # # Calculate ROUGE scores
        # rouge_scores = calculate_rouge_score(reference_answer, ai_response)
        # print("[ROUGE SCORES]:", rouge_scores)

        # --- POST to external endpoint ---
        try:
            post_url = "http://192.168.181.73:5000/api/response"
            post_response = requests.post(post_url, json={"response": ai_response})

            if post_response.status_code != 200:
                print(f"[POST ERROR] Failed to post AI response: {post_response.text}")
            else:
                print("[POST SUCCESS] AI response posted successfully.")
        except Exception as post_err:
            print(f"[POST EXCEPTION] Error while posting AI response: {post_err}")

        return jsonify({"response": ai_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/response', methods=['POST'])
def post_response():
    global ai_response
    data = request.get_json()
    
    if not data or 'response' not in data:
        return jsonify({"error": "Invalid payload"}), 400
    
    if not data['response'].strip():
        return jsonify({"error": "Response cannot be empty"}), 400
    
    ai_response = data['response']
    
    return jsonify({"message": "Response saved"}), 200

@app.route('/api/response', methods=['GET'])
def get_response():
    if ai_response is None:
        return jsonify({"error": "No response has been saved yet"}), 404
    return jsonify({"response": ai_response}), 200

@app.route('/api/announcements', methods=['POST'])
def post_weekly_announcements():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON in request body"}), 400

        range_type = data.get("range", "week")

        today = datetime.today().date()

        if range_type == "today":
            start_date = today
            end_date = today + timedelta(days=1)

        elif range_type == "custom":
            start_str = data.get("start_date")
            end_str = data.get("end_date")
            if not start_str or not end_str:
                return jsonify({"error": "Missing 'start_date' or 'end_date' for custom range"}), 400
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                if start_date > end_date:
                    return jsonify({"error": "'start_date' cannot be after 'end_date'"}), 400
            except ValueError:
                return jsonify({"error": "Dates must be in YYYY-MM-DD format"}), 400

        elif range_type == "week":
            start_date = today - timedelta(days=today.weekday())  # Monday
            end_date = start_date + timedelta(days=6)             # Sunday

        else:
            return jsonify({"error": f"Unsupported range type '{range_type}'. Use 'week', 'today', or 'custom'."}), 400

        db = connect_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT title, description
            FROM announcements 
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
        """, (start_date, end_date))
        
        announcements = cursor.fetchall()
        cursor.close()
        db.close()

        if not announcements:
            return jsonify({"message": "No announcements found for the selected date range."}), 200

        return jsonify(announcements), 200

    except mysql.connector.Error as db_err:
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

wakeword_detected = False  # Initialize globally
cancel_detected = False #Initialize globaly

@app.route('/api/wakeword', methods=['POST'])
def receive_wakeword():
    global wakeword_detected
    try:
        data = request.get_json()
        if not data or 'detected' not in data:
            return jsonify({"error": "Invalid payload: 'detected' field missing"}), 400

        if not isinstance(data['detected'], bool):
            return jsonify({"error": "'detected' field must be a boolean"}), 400

        wakeword_detected = data['detected']
        return jsonify({"message": "Wake word status updated", "detected": wakeword_detected}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/wakeword', methods=['GET'])
def get_wakeword_status():
    global wakeword_detected
    try:
        response = {"detected": wakeword_detected}
        if wakeword_detected:
            wakeword_detected = False  # Reset after reporting
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/cancel', methods=['POST'])
def receive_cancel():
    global cancel_detected
    try:
        data = request.get_json()
        if not data or 'detected' not in data:
            return jsonify({"error": "Invalid payload: 'detected' field missing"}), 400

        if not isinstance(data['detected'], bool):
            return jsonify({"error": "'detected' field must be a boolean"}), 400

        cancel_detected = data['detected']
        return jsonify({"message": "Cancel status updated", "detected": cancel_detected}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/cancel', methods=['GET'])
def get_cancel_status():
    global cancel_detected
    try:
        response = {"detected": cancel_detected}
        if cancel_detected:
            cancel_detected = False  # Reset after reporting
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
# --- Run ---
if __name__ == '__main__':
    print("[STARTING SYSTEM] Initializing...")
    dump_database()
    count_entries_in_dump()
    update_vector_store()
    threading.Thread(target=trace_watcher, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
