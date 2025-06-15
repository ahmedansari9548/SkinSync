from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from model import chat_without_nlp

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chatbot")
async def chatbot_response(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    response = chat_without_nlp(user_message)
    return {"response": response}
