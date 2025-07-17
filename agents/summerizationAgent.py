from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os
from pydantic import BaseModel


load_dotenv()
class SummarizeRequest(BaseModel):
    text: list = None
async def get_summarization_agent(summarizeRequest:SummarizeRequest):
    """Load documents and create summarization chain"""
    # Load documents (this can be slow, so good to make async)
    def load_and_summarize():
        # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        # docs= loader.load()
        docs=summarizeRequest.text
        print("Summarization request:", docs)
        # Debug: Check if docs are loaded
        print(f"Loaded {len(docs)} documents")
        if not docs:
            raise ValueError("No documents loaded")
        
        # Debug: Check document content
        print(f"First doc content length: {len(docs[0].get("page_content"))}")
        combined_content = "\n\n".join([doc.get("page_content") for doc in docs])
        print(f"Combined content : {combined_content}")
        # Truncate if too long (Gemini has token limits)
        if len(combined_content) > 30000:  # Rough character limit
            combined_content = combined_content[:30000] + "..."
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following:\\n\\n{context}")
        ])
        prompt = f"Write a concise summary of the following:\n\n{combined_content}"
        # chain = create_stuff_documents_chain(llm, prompt)
        
        # Make sure we're passing the docs correctly
        # result = chain.invoke({"context": docs})
        result = llm.invoke(prompt)
        summary_text = result.content
        
        # Extract additional info if needed
        metadata = result.response_metadata if hasattr(result, 'response_metadata') else {}
        
        return {
            "summary": summary_text,
            "metadata": metadata,
            "message_type": result.type
        }
    
    # Run the synchronous loader in a thread pool
    
    
    result = await asyncio.get_event_loop().run_in_executor(None, load_and_summarize)
    return {"summary": result}
    print("Summarization result:", result)
    # return result
    # Return both docs and chain for flexibility
    # return {"docs": docs, "chain": chain}

async def summarize_content(url: str = None):
    """Summarize content from URL or default URL"""
    if url:
        loader = WebBaseLoader(url)
    else:
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    
    docs = loader.load()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )

    chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoke chain and return result
    result = await chain.ainvoke({"context": docs})
    return result
