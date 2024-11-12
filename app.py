from pydantic import BaseModel
from fastapi.responses import StreamingResponse 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from response_generator import ResponseGenerator 
from sqlAgent import SQLAgent
import os 
import importlib.metadata
import importlib.util
from typing import List

directory_path = os.path.join(os.path.dirname(__file__), 'vectorEmbedding')
module_path = os.path.join(directory_path, 'index.py')

# Load the module
spec = importlib.util.spec_from_file_location('index', module_path)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)

# Import the vectorEmbedding class from the loaded module
vectorEmbedding = index.vectorEmbedding

PORT = 8001  # or any port you prefer

# Configuration for embedding processor
API_KEY = "sk-proj-LsiMW7naIQGLpzsfvkcmT3BlbkFJMpyDM3ZHKapKke9sBrez"
CONNECTION_STRING="postgres://tsdbadmin:dg7ehmx8wl7ahrwt@ufsai5bzli.ca1khab64u.tsdb.cloud.timescale.com:37119/tsdb?sslmode=require"


reponseGenerate = ResponseGenerator(API_KEY)
sql_agent = SQLAgent(API_KEY, CONNECTION_STRING)
pdfs_path="files/pdfs"
ve_instance = vectorEmbedding(API_KEY, CONNECTION_STRING, pdfs_path)

class Message(BaseModel):
    user: str
    bot: str

# Main UserInfo class
class UserInfo(BaseModel):
    user_query: str  # String type variable
    lastThree: List[Message]  # Array of Message objects

class QueryModel(BaseModel):
    user_query: str
    top_k:int=5

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatlistings-adamvenords-projects.vercel.app", "http://localhost:3000", "*"],
    # allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    try:
        print("Request received")
        # res = sql_agent.create_db_instance()
        # return res
         
        # Handle the streamed response
        # print(response)
        return {
            "message":"Server is Up"
        }
    except Exception as e:
        print("Error", e)
        return {"error_message": f"Due to some technical issue, I can't help you right now. Sorry for the inconvenience. You can reach us at help@realestate.yodata.me"}

# @app.post("")

@app.post("/bot/conversation/v4")
async def queryInput(user:UserInfo):
    try:

        # sqlResponse = sql_agent.create_db_instance(user.user_query)
        response = ve_instance.process_input_with_retrieval(user.user_query)
        return StreamingResponse(response, media_type="text/event-stream")
        # return response
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error_message": f"Due to some technical issue, I can't help you right now. Sorry for the inconvenience. You can reach us at help@realestate.yodata.me"}
    
