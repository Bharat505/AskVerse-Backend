# qa_service.py
import os
import json
from json.decoder import JSONDecodeError
from typing import List, Dict, Any

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import google.auth
from retrieval import hybrid_search
from dotenv import load_dotenv 
###############################################################################
# 1) Gemini-based LLM
###############################################################################
# If you rely on GCP ADC or a service account, you may not need to explicitly call genai.configure()
# Otherwise: genai.configure(api_key="YOUR_API_KEY")
 # Load .env file

# Load environment variables from .env
# 1) Authenticate Using Service Account (Inside Docker)
###############################################################################
credentials, project_id = google.auth.default()  # Loads credentials from GOOGLE_APPLICATION_CREDENTIALS

# Configure Generative AI API using credentials
genai.configure(credentials=credentials)


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=8192,
    timeout=None,
    max_retries=5
)

###############################################################################
# 2) JSON Helper
###############################################################################
def get_json_output(llm_client, messages: List[Dict], num_retries: int = 2) -> Any:
    """
    Invokes the LLM with a JSON-oriented request (messages) and attempts to parse.
    """
    for attempt in range(num_retries):
        try:
            gen_cfg = {"response_mime_type": "application/json"}
            response = llm_client.invoke(messages, generation_config=gen_cfg)
            return json.loads(response.content)
        except JSONDecodeError as e:
            if attempt < num_retries - 1:
                print(f"JSON decoding failed (attempt {attempt+1}), retrying...\n{e}")
            else:
                raise e

###############################################################################
# 3) Step A: Main Q&A
###############################################################################
def answer_question_with_followups(query: str, top_k: int = 6, alpha: float = 0.6) -> Dict[str, Any]:
    """
    1) Hybrid search -> relevant chunks
    2) LLM: "Please respond in strict JSON with answer, follow_up_questions, citations."
    3) Return parsed JSON or an error.
    """

    # A) Retrieve relevant chunks
    search_results = hybrid_search(query, top_k=top_k, alpha=alpha)

    # B) Build a context from chunk text
    chunk_blocks = []
    for (src, text, score) in search_results:
        chunk_blocks.append(f"--- [Source: {src}] ---\n{text}\n")
    context_str = "\n".join(chunk_blocks)

    # C) Prompts
    system_message = (
        "You are an advanced AI specialized in Apache Spark, Apache Kafka, and React. "
        "Use ONLY the provided internal documents (the 'Bible') for your response. "
        "If the docs do not cover the query, disclaim it.\n\n"
        "Return strict JSON with exactly these keys:\n"
        '  "answer": a structured factual answer,\n'
        '  "follow_up_questions": array of exactly 5 relevant next questions,\n'
        '  "citations": array of chunk sources used.\n'
        "Do not include any extra keys. If you cannot find info in the docs, say so in 'answer'.\n"
    )

    user_message = (
        f"User Question: {query}\n\n"
        "Relevant Context:\n"
        f"{context_str}\n\n"
        "Respond in JSON. Example:\n"
        "{\n"
        '  "answer": "...",\n'
        '  "follow_up_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],\n'
        '  "citations": ["source1", "source2"]\n'
        "}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        parsed = get_json_output(model, messages, num_retries=2)
        if not isinstance(parsed, dict):
            return {
                "error": "LLM returned invalid structure",
                "answer": None,
                "follow_up_questions": [],
                "citations": []
            }
        return parsed
    except Exception as e:
        return {
            "error": str(e),
            "answer": None,
            "follow_up_questions": [],
            "citations": []
        }

###############################################################################
# 4) Step B: Confidence Scoring
###############################################################################
def evaluate_confidence(query: str, final_answer: str, context_str: str) -> float:
    """
    We do a second pass: show the question, final answer, context, 
    and ask the model to return "confidence_score" in [0.0, 1.0].
    """

    system_message = (
        "You are a meticulous AI evaluator. You see the user's question, the final answer, "
        "and the same context (the 'Bible'). Rate how confident you are in the correctness "
        "and completeness of that final answer, from 0.0 to 1.0. If partially correct or missing data, "
        "lower the score. If the context fully supports it, raise it.\n\n"
        "Return valid JSON exactly like:\n"
        "{\n"
        '  "confidence_score": 0.82\n'
        "}"
    )

    user_message = (
        f"User Question: {query}\n\n"
        f"Final Answer: {final_answer}\n\n"
        "Relevant Context:\n"
        f"{context_str}\n\n"
        "Return your confidence_score in JSON, e.g. { \"confidence_score\": 0.85 }"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        result = get_json_output(model, messages, num_retries=2)
        if (
            isinstance(result, dict)
            and "confidence_score" in result
            and isinstance(result["confidence_score"], (int, float))
        ):
            score = float(result["confidence_score"])
            return max(0.0, min(score, 1.0))
        return 1.0
    except Exception:
        return 1.0

###############################################################################
# 5) Orchestrator
###############################################################################
def answer_question_with_confidence(query: str, top_k: int = 6, alpha: float = 0.6) -> Dict[str, Any]:
    """
    1) Get main QA result from answer_question_with_followups()
    2) Then re-run or store context for confidence
    3) Return merged result with confidence_score
    """

    qa_result = answer_question_with_followups(query, top_k=top_k, alpha=alpha)
    if qa_result.get("error"):
        return qa_result

    final_answer = qa_result.get("answer", "")

    # re-run the search to feed the context
    sr = hybrid_search(query, top_k=top_k, alpha=alpha)
    context_blocks = []
    for (src, text, score) in sr:
        context_blocks.append(f"--- [Source: {src}] ---\n{text}\n")
    combined_context = "\n".join(context_blocks)

    # Evaluate confidence
    conf_score = evaluate_confidence(query, final_answer, combined_context)

    # Merge
    qa_result["confidence_score"] = conf_score
    return qa_result
