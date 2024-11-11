import os
import re
import numpy as np
import psycopg2
import math
import csv
from psycopg2 import OperationalError
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
import asyncio
from typing import AsyncIterable, Dict

class vectorEmbedding:
    def __init__(self, api_key, connection_string,directory_path):
        self.client = OpenAI(api_key=api_key)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.connection_string = connection_string
        self.conn = None
        self.cur = None
        self.directory_path=directory_path
        self.api_key=api_key

    def get_embedding(self, text_chunks):
        embeddings = self.embeddings.embed_documents(text_chunks)
        return embeddings
   
    def generate_embedding_for_single_chunk(self, text_chunk):
        # Generate embedding for the single document
        embeddings = self.embeddings.embed_documents(text_chunk)
        # Since we know there is only one chunk, we can return the first element of the embeddings list
        return embeddings[0] if embeddings else None



    def save_embeddings_to_csv(self, chunked_docs):
    # Hard-code the complete file path
        file_path = "files/pdfs/embeddigns.csv"  # Update this path with your desired location

        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir:  # Check if there's an actual directory in the path (not just a filename)
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open the CSV file to write data with UTF-8 encoding
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['content', 'tokens', 'embedding'])

                # Write each chunk with its embeddings
                for doc in chunked_docs:
                    content = doc['content']
                    tokens = doc['tokens']
                    embedding = doc['embedding']  # Assuming the embedding is a list or vector

                    # Convert embedding to a string (if it's a list)
                    embedding_str = ','.join(map(str, embedding))

                    # Write the row in the CSV file
                    writer.writerow([content, tokens, embedding_str])

            print(f"Embeddings successfully saved to {file_path}")

        except Exception as e:
            print(f"Error occurred: {e}")




    def load_pdfs_from_directory(self, ):
        all_docs = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.pdf'):
                path = os.path.join(self.directory_path, filename)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for i in range(0, len(docs), 2):
                    chunk_docs = docs[i:i+2]
                    combined_content = " ".join([doc.page_content for doc in chunk_docs])
                    cleaned_content = self.clean_text(combined_content)
                    new_doc = {'page_content': cleaned_content}  # Creating a new doc with cleaned content
                    all_docs.append(new_doc)
                

        # Create chunks and get embeddings
        chunked_docs = self.create_chunks(all_docs)
        content_list = [chunk['content'] for chunk in chunked_docs]
        # Generate embeddings
        embeddings = self.get_embedding(content_list)
        for i, chunk in enumerate(chunked_docs):
            chunk['embedding'] = embeddings[i]


          # Define your file path here
        # uncomment to load data into csv file / later could be used to upload in other db's
        # self.save_embeddings_to_csv(chunked_docs)   
        return chunked_docs

    def clean_text(self, text):
        # Remove images, new lines, and unwanted characters
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
        text = re.sub(r'[{}|<>]', '', text)  # Remove {}|<>
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
        return text

    def create_chunks(self, docs, max_token_size=1000):
        chunks = []
        for doc in docs:
            # page_content = getattr(doc, 'page_content', "")
            page_content  = doc['page_content']
            title = self.extract_page_title(page_content)
            tokens = self.num_tokens_from_string(page_content)

            if tokens <= max_token_size:
                chunks.append({'title': title, 'content': page_content, 'tokens': tokens})
            else:
                words = page_content.split()
                start = 0
                while start < len(words):
                    end = min(start + max_token_size, len(words))
                    chunk = ' '.join(words[start:end])
                    chunks.append({'title': title, 'content': chunk, 'tokens': end - start})
                    start = end
        return chunks

    def extract_page_title(self, page_content):
        lines = page_content.split('\n')
        return lines[1].strip() if len(lines) > 1 else "Untitled Page"

    def num_tokens_from_string(self, text):
        return len(text.split())
    
    def connectdb(self):
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(self.connection_string)
                self.cur = self.conn.cursor()
                register_vector(self.conn)
                self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.conn.commit()
                print("Database connection established and extension registered.")
        except OperationalError as e:
            print("Database connection failed:", e)
            return None
        
    def get_cursor(self):
        self.connect()
        return self.cur

    def close(self):
        if self.conn is not None and not self.conn.closed:
            self.cur.close()
            self.conn.close()
            print("Database connection closed.")

    def createtable(self):
        if self.conn is None or self.cur is None:
            print("Connection is not established. Calling connectdb first.")
            self.connectdb()

        if self.conn is not None and self.cur is not None:
            try:
                table_create_command = """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id bigserial primary key,
                    content text,
                    tokens integer,
                    embedding vector(1536)
                );
                """
                self.cur.execute(table_create_command)
                self.conn.commit()
                print("Table 'embeddings' created successfully.")
                return "Table Created"
            except OperationalError as e:
                print("Failed to create table:", e)
            # finally:
                # self.cur.close()  # Optionally close cursor if no further queries
                # self.conn.close()  # Close connection safely
    
    def connect_and_create(self):
        self.connectdb()  # Ensure the database is connected and extension is set
        self.createtable()
        data=self.load_pdfs_from_directory()
        res=self.insert_data(data)
        return res

    def insert_data(self, data):
        #Batch insert embeddings and metadata from dataframe into PostgreSQL database
        try:
            register_vector(self.conn)
            cur = self.conn.cursor()
            # Prepare the list of tuples to insert
            # data_list = [( row['content'], int(row['tokens']), np.array(row['embeddings'])) for index, row in data.iterrows()]
            data_list = []
            for row in data:
            # Validate the length of the embeddings to ensure they match the expected dimension
                data_list.append((row['content'], int(row['tokens']), row['embedding']))
            # Use execute_values to perform batch insertion
            execute_values(cur, "INSERT INTO embeddings (content, tokens, embedding) VALUES %s", data_list)
            print("Data successfully saved to db")
            # Commit after we insert all embeddings
            self.conn.commit()
            return "Data successfully saved to database"
        except Exception as e:
            print("An error occurred:", e)
            self.conn.rollback()  # This will undo any changes if the commit fails

    def total_records(self):
        self.connectdb()
        self.cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
        num_records = self.cur.fetchone()[0]
        num_lists = num_records / 1000
        if num_lists < 10:
            num_lists = 10  
        if num_records > 1000000:
            num_lists = math.sqrt(num_records)

        #use the cosine distance measure, which is what we'll later use for querying
        self.cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});')
        self.conn.commit()
        return num_records
    
    async def pre_userquery_and_response(self, user_query, oldResponse):
        
        system_message_prompt = """
        Handle queries based on past interactions, including generating new queries that reflect requests for additional information or records related to previous queries. Ensure that responses are contextually relevant to the user's previous questions and the current query. Manage cases where users ask for more details or follow-up information, including but not limited to additional records or details about Berkshire Hathaway Services. Ensure responses are clear and directly address the user's needs based on their past interactions.
        """

        analysis_prompt = f"""
        Analyze the following data:

        Old User queries and responses from chatbot:
        Data: {oldResponse}

        Based on the user query: '{user_query}', determine if the query is related to previous responses or if it's a new request.

        1. **If the user query refers to past interactions:**
        - Generate a response indicating the user is seeking additional information or records related to their previous queries.
        - Ensure the response includes relevant information from past interactions and new details if applicable.
        - Exclude redundant records.

        2. **If the user query does not refer to past interactions:**
        - Just return the query as it is.

        3. **Handle different use cases if applicable.**

        Provide the output in natural language:
        - Exclude unnecessary details and focus on providing the rephrased query.
        - Just return the reponse , dont include the things like query is not related to old query, just process if query needs
        to be updated based on the previous reocrds then update the query and return dont include in response that 
        this is the change in the old and current query, just resonse should be query nothing else
        .
        """
        messages = [
            SystemMessage(content=system_message_prompt),
            AIMessage(content="Hello! I'm your assistant for Berkshire Hathaway HomeServices and BHHS listings. I can help with any questions you have, including those related to past interactions. Please let me know your current query or if you need additional information or records based on previous responses. I'll make sure to provide the most relevant and accurate information."),
            HumanMessage(content=analysis_prompt)
        ]
        print("messages")
        response = await self.get_completion_from_userQuery(messages)
        return response

    async def process_input_with_retrieval(self,user_input, sqlResponse) -> AsyncIterable[str]:
        arr=[user_input]
        embeddings=self.generate_embedding_for_single_chunk(arr)
        #Step 1: Get documents related to the user input from database
        related_docs = self.get_top3_similar_docs(embeddings)
        # Step 2: Get completion from OpenAI API
        # Set system message to help set appropriate tone and context for model
        system_message_prompt = """
        You are a friendly and knowledgeable chatbot specialized in Berkshire Hathaway Services and bhhs listings. \
        You can answer questions about Berkshire Hathaway, its features, and its use cases. \
        You respond in a concise, technically credible tone, providing the most relevant information based on the user's query.
        These characters should not be included in the response * ** " and useless spaces, unless they are specifically required
        Ensure the response is well-formatted with new lines after each point.
        If the new line is required the pass /n that will indicate here new line is required
        The response of hi or hello should be greetings and how may i assit you?
        Do not add properties that are not included in the data or in listings. And if users ask for properties that are not included then just say this property is not included for this data
        Do not query for questions that are not properly defined like single character, single letter, single digit and unrelated keywords or sentence if the query dont make sentence and response no specific information can be provided based on your query
        """

        analysis_prompt = f"""
        Analyze the following data:

        Berkshire Data:
        {related_docs}

        Listings:
        Data: {sqlResponse['data']}
        Column Names: {', '.join(sqlResponse['column_names'])}

        Based on the user query: '{user_input}', provide a concise and relevant response. 
        Do not include the characters ** and "" in the response.
        Instead of - in the response show numbers for main point and roman numbers for nested things
        """

        # Prepare messages to pass to the model
        # We use a delimiter to help the model understand where the user_input starts and ends
        messages = [
            SystemMessage(content=system_message_prompt),
            AIMessage(content="Hello! I'm Chatbot. I'm the HBerkshire Hathaway HomeServices and BHHS listings assistant who will help you today."),
            HumanMessage(content=analysis_prompt)
        ]
        print("Logging messages")
        async for token in self.get_completion_from_messages(messages):
            yield token

    async def get_completion_from_userQuery(self, messages) -> str:
        callback = AsyncIteratorCallbackHandler()
        chat_openai = ChatOpenAI(
            model="gpt-4o",
            temperature=1.0,
            openai_api_key=self.api_key,
            streaming=True,
            verbose=True,
            callbacks=[callback]
        )

        print("chat_openai initialized")
        response = ''

        try:
            # Create a task to generate the chat response
            task = asyncio.create_task(chat_openai.agenerate(messages=[messages]))
            
            # Collect tokens as they are generated
            async for token in callback.aiter():
                response += ' ' + token
                # Optionally, you can print tokens for debugging
                # print("Token...", token)

            # Await the completion of the chat generation task
            await task

        except Exception as e:
            print(f"Caught exception: {e}")
            response = f"Error: {str(e)}"  # Optionally return an error message

        finally:
            print("Finally called")
            callback.done.set()

        return response

    async def get_completion_from_messages(self,messages) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        chat_openai = ChatOpenAI(
            model="gpt-4o",
            temperature=1.0,
            openai_api_key= self.api_key,
            streaming=True,
            verbose=True,
            callbacks=[callback]
        )
        
        print("chat_openai initialized")
        try:
            task = asyncio.create_task(chat_openai.agenerate(messages=[messages]))
            response = ''
            async for token in callback.aiter():
                response += ' ' + token
                # print("Token...", token)
                yield token
            await task
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            print("Finally called")
            callback.done.set()

        # return response

    def get_top3_similar_docs(self,query_embedding):
        # self.connectdb()
        embedding_array = np.array(query_embedding)
        # Register pgvector extension
        # register_vector(self.conn)
        # print("Vector Registered")
        # cur = self.conn.cursor()
        # print("cur..", cur)
        # Get the top 3 most similar documents using the KNN <=> operator
        self.cur.execute("SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT 5", (embedding_array,))
        top3_docs = self.cur.fetchall()
        return top3_docs








