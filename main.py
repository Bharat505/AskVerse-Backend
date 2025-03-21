import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your final Q&A method
from qa_service import answer_question_with_confidence

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(payload: QueryModel):
    result = answer_question_with_confidence(payload.question, top_k=8)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
