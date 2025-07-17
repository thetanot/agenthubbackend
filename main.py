from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.summerizationAgent import get_summarization_agent, summarize_content
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SummarizeRequest(BaseModel):
    text: list = None

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest = None):
    try:
        # Use the async function
        if request and request.url:
            result = await summarize_content(request.url)
        else:
            result = await summarize_content()
        
        return {"summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-default")
async def summarize_default(SummarizeRequest: SummarizeRequest = None):
    """Summarize the default URL"""
    try:
        result = await get_summarization_agent(SummarizeRequest)
        return {"summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Summarization API is running"}