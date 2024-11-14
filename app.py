from pydantic import BaseModel
from response_generator import ResponseGenerator 
from fastapi.responses import StreamingResponse 
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 

# Hardcoded configuration values
PORT = 8000  # or any port you prefer

# Chat class initialization with API key
OPENAI_API_KEY = 'sk-proj-LsiMW7naIQGLpzsfvkcmT3BlbkFJMpyDM3ZHKapKke9sBrez'  # Replace with your OpenAI API key
chat = ResponseGenerator(OPENAI_API_KEY)

class UserInfo(BaseModel):
    user_query: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/bot/conversation")
async def conversation(user: UserInfo):
    print(f"Received request: {user.user_query}")
    try:
        generator = chat.data_response(user.user_query)
        # response = ""
        # async for token in generator:
        #     response += token
        #     print("Response..", response)
        return StreamingResponse(generator, media_type="text/event-stream")
        # return {"response": response}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error_message": f"Due to some technical issue, I can't help you right now. Sorry for the inconvenience. You can reach us at https://www.abelsontaylor.com/contact-us/"}

if __name__ == '__main__':
    print(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
