from langchain.schema import HumanMessage, SystemMessage, AIMessage
import asyncio
import json
from typing import AsyncIterable
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Dict
from langchain_openai import  ChatOpenAI

class ResponseGenerator:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        self.system_message_prompt = self.read_prompt_from_file('prompt.txt')

    def read_prompt_from_file(self, file_path: str) -> str:
        with open(file_path, 'r') as file:
            prompt = file.read()
            return prompt
    def read_data_from_file(self, file_path: str) -> Dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    async def realestate_response(self, text: str) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        # json_data = self.read_data_from_file('promptdataset.json')
        analysis_prompt = f"User query: {text}"
        messages = [
            SystemMessage(content=self.system_message_prompt),
            AIMessage(content="Hello! I'm your healthcare assistant. How can I help you today?"),
            HumanMessage(content=analysis_prompt)
        ]
        
        chat_openai = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=1.0,
            openai_api_key=self.openai_key,
            streaming=True,
            verbose=True,
            callbacks=[callback]
        )
        try:
            task = asyncio.create_task(chat_openai.agenerate(messages=[messages]))
            response = ''
            async for token in callback.aiter():
                response += ' ' + token
                yield token
            await task
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            print("Finally called")
            callback.done.set()
            
        
