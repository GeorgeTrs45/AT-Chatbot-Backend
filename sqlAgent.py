import psycopg2
import json
from psycopg2 import OperationalError
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
import os
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import re
import ast
# Database connection parameters



class SQLAgent:
    def __init__(self, api_key, connection_string):
        self.api_key = api_key
        self.conn = None
        self.cursor = None
        self.connection_string=connection_string
        self.llm =  ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
        self.db = SQLDatabase.from_uri("postgresql://tsdbadmin:dg7ehmx8wl7ahrwt@ufsai5bzli.ca1khab64u.tsdb.cloud.timescale.com:37119/tsdb?sslmode=require")
        # os.environ["OPENAI_API_KEY"] = self.api_key
      
    def connectdb(self):
        try:
            # Connect to PostgreSQL
            self.conn = psycopg2.connect("postgresql://tsdbadmin:dg7ehmx8wl7ahrwt@ufsai5bzli.ca1khab64u.tsdb.cloud.timescale.com:37119/tsdb?sslmode=require")
            self.cursor = self.conn.cursor()
            return self.conn, self.cursor
        except OperationalError as e:
            self.conn = None
            self.cursor = None
            return None, None

    def create_listings_table(self):
         try:
             self.connectdb()
             table_create_command = """
                CREATE TABLE IF NOT EXISTS listings (
                topic TEXT, 
                originatingSystemName  TEXT, 
                originatingSystemKey  TEXT, 
                streetAddress  TEXT, 
                addressLocality  TEXT, 
                addressRegion  TEXT, 
                postalCode  TEXT, 
                propertyType  TEXT, 
                propertySubType  TEXT, 
                listingContractDate  TIMESTAMP, 
                addressCountry  TEXT,
                modificationTimestamp TIMESTAMP, 
                listingPrice_type  TEXT, 
                listingPrice_priceCurrency  TEXT, 
                listingPrice_price  INTEGER, 
                listingStatus  TEXT, 
                buyerAgent_type  TEXT, 
                buyerAgent_name  TEXT,  
                latitude  TEXT, 
                longitude  TEXT, 
                listingAgent_type  TEXT, 
                listingAgent_name  TEXT, 
                listingAgent_email  TEXT, 
                listingAgent_identifier_mlsAgentId  TEXT, 
                listingOffice_type  TEXT, 
                listingOffice_name  TEXT, 
                listingOffice_email  TEXT, 
                coListingAgent_name  TEXT, 
                coListingAgent_email  TEXT,
                listingId  TEXT, 
                listingOriginatingSystem_name  TEXT, 
                listingOriginatingSystem_identifier_orgId  TEXT, 
                livingArea_value DECIMAL(15, 2), 
                livingArea_unitCode TEXT, 
                livingArea_unitText TEXT, 
                lotSize_value DECIMAL(15, 2),  
                lotSize_unitText TEXT, 
                numberOfBedrooms DECIMAL(5, 2), 
                numberOfBathrooms DECIMAL(5, 2), 
                description TEXT, 
                closeDate TIMESTAMP,
                soldPrice_type TEXT, 
                soldPrice_price INTEGER,
                numberOfFullBathrooms TEXT,
                buyerOffice_name  TEXT, 
                buyerOffice_email  TEXT, 
                buyerAgent_email  TEXT,
                numberOfRooms TEXT,
                purchaseContractDate TIMESTAMP,
                yearBuilt TEXT,
                listingAgent_additionalProperty_agentId TEXT,
                listingOffice_additionalProperty_officeId TEXT,
                createdAt TIMESTAMP DEFAULT current_timestamp 
                );
                """
             self.cursor.execute(table_create_command)
             self.conn.commit()
             print("Table 'listings' created successfully.")
             return "Table Created"
         except Exception as e:
             return {
                 "error":e
             }


    def load_data(self):
        if not self.conn or not self.cursor:
            print("No database connection. Please connect to the database first.")
            return

        try:
            print("Request received to load data")
            path = "files/listings_data/bhhs-listings_1k.json"

            # Read JSON data
            with open(path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    return

            # Mapping of JSON keys to the expected database column names
            key_mapping = {
                "topic": "topic",
                "originatingsystemname": "originatingSystemName",
                "originatingsystemkey": "originatingSystemKey",
                "streetaddress": "streetAddress",
                "addresslocality": "addressLocality",
                "addressregion": "addressRegion",
                "postalcode": "postalCode",
                "propertytype": "propertyType",
                "propertysubtype": "propertySubType",
                "listingcontractdate": "listingContractDate",
                "addresscountry": "addressCountry",
                "modificationtimestamp": "modificationTimestamp",
                "listingprice_type": "listingPrice_type",
                "listingprice_pricecurrency": "listingPrice_priceCurrency",
                "listingprice_price": "listingPrice_price",
                "listingstatus": "listingStatus",
                "buyeragent_type": "buyerAgent_type",
                "buyeragent_name": "buyerAgent_name",
                "latitude": "latitude",
                "longitude": "longitude",
                "listingagent_type": "listingAgent_type",
                "listingagent_name": "listingAgent_name",
                "listingagent_email": "listingAgent_email",
                "listingagent_identifier_mlsagentid": "listingAgent_identifier_mlsAgentId",
                "listingoffice_type": "listingOffice_type",
                "listingoffice_name": "listingOffice_name",
                "listingoffice_email": "listingOffice_email",
                "colistingagent_name": "coListingAgent_name",
                "colistingagent_email": "coListingAgent_email",
                "listingid": "listingId",
                "listingoriginatingsystem_name": "listingOriginatingSystem_name",
                "listingoriginatingsystem_identifier_orgid": "listingOriginatingSystem_identifier_orgId",
                "livingarea_value": "livingArea_value",
                "livingarea_unitcode": "livingArea_unitCode",
                "livingarea_unittext": "livingArea_unitText",
                "lotsize_value": "lotSize_value",
                "lotsize_unittext": "lotSize_unitText",
                "numberofbedrooms": "numberOfBedrooms",
                "numberofbathrooms": "numberOfBathrooms",
                "closedate": "closeDate",
                "soldprice_type": "soldPrice_type",
                "soldprice_price": "soldPrice_price",
                "numberoffullbathrooms": "numberOfFullBathrooms",
                "buyeroffice_name": "buyerOffice_name",
                "buyeroffice_email": "buyerOffice_email",
                "buyeragent_email": "buyerAgent_email",
                "numberofrooms": "numberOfRooms",
                "purchasecontractdate": "purchaseContractDate",
                "yearbuilt": "yearBuilt",
                "listingagent_additionalproperty_agentid": "listingAgent_additionalProperty_agentId",
                "listingoffice_additionalproperty_officeid": "listingOffice_additionalProperty_officeId",
                "createdat": "createdAt"
            }

            # Normalize keys for each record in the data
            normalized_data = []
            # print(len(data))
            for record in data[:100]:
                normalized_record = {}
                for key, value in record.items():
                    normalized_key = key_mapping.get(key.lower())
                    if normalized_key:
                        normalized_record[normalized_key] = value
                normalized_data.append(normalized_record)

            # Insert JSON data into the listings table
            insert_query = """
            INSERT INTO listings (
                topic, originatingSystemName, originatingSystemKey, streetAddress, 
                addressLocality, addressRegion, postalCode, propertyType, 
                propertySubType, listingContractDate, addressCountry, modificationTimestamp, 
                listingPrice_type, listingPrice_priceCurrency, listingPrice_price, listingStatus, 
                buyerAgent_type, buyerAgent_name, latitude, longitude, listingAgent_type, 
                listingAgent_name, listingAgent_email, listingAgent_identifier_mlsAgentId, 
                listingOffice_type, listingOffice_name, listingOffice_email, coListingAgent_name, 
                coListingAgent_email, listingId, listingOriginatingSystem_name, 
                listingOriginatingSystem_identifier_orgId, livingArea_value, livingArea_unitCode, 
                livingArea_unitText, lotSize_value, lotSize_unitText, numberOfBedrooms, 
                numberOfBathrooms, closeDate, soldPrice_type, soldPrice_price, 
                numberOfFullBathrooms, buyerOffice_name, buyerOffice_email, buyerAgent_email, 
                numberOfRooms, purchaseContractDate, yearBuilt, listingAgent_additionalProperty_agentId, 
                listingOffice_additionalProperty_officeId, createdAt
            ) VALUES (
                %(topic)s, %(originatingSystemName)s, %(originatingSystemKey)s, %(streetAddress)s, 
                %(addressLocality)s, %(addressRegion)s, %(postalCode)s, %(propertyType)s, 
                %(propertySubType)s, %(listingContractDate)s, %(addressCountry)s, %(modificationTimestamp)s, 
                %(listingPrice_type)s, %(listingPrice_priceCurrency)s, %(listingPrice_price)s, %(listingStatus)s, 
                %(buyerAgent_type)s, %(buyerAgent_name)s, %(latitude)s, %(longitude)s, %(listingAgent_type)s, 
                %(listingAgent_name)s, %(listingAgent_email)s, %(listingAgent_identifier_mlsAgentId)s, 
                %(listingOffice_type)s, %(listingOffice_name)s, %(listingOffice_email)s, %(coListingAgent_name)s, 
                %(coListingAgent_email)s, %(listingId)s, %(listingOriginatingSystem_name)s, 
                %(listingOriginatingSystem_identifier_orgId)s, %(livingArea_value)s, %(livingArea_unitCode)s, 
                %(livingArea_unitText)s, %(lotSize_value)s, %(lotSize_unitText)s, %(numberOfBedrooms)s, 
                %(numberOfBathrooms)s, %(closeDate)s, %(soldPrice_type)s, %(soldPrice_price)s, 
                %(numberOfFullBathrooms)s, %(buyerOffice_name)s, %(buyerOffice_email)s, %(buyerAgent_email)s, 
                %(numberOfRooms)s, %(purchaseContractDate)s, %(yearBuilt)s, %(listingAgent_additionalProperty_agentId)s, 
                %(listingOffice_additionalProperty_officeId)s, %(createdAt)s
            )
            """
            print("Here....")
            # Execute the insert query for each normalized record
            for record in normalized_data:
                print("Here 2")
                self.cursor.execute(insert_query, record)

            # Commit the transaction
            self.conn.commit()
            print("Data has been successfully loaded into the listings table.")

        except Exception as e:
            print(f"Error while loading data: {e}")
            self.conn.rollback()

    def closedb(self):
        if self.cursor:
            self.cursor.close()
            print("Cursor closed.")
        if self.conn:
            self.conn.close()
            print("Connection closed.")

    def extract_column_names(self, sql_query):
        # Use regex to find the columns between SELECT and FROM
        match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
        if match:
            columns = match.group(1).split(',')
            # Strip quotes and whitespace
            columns = [col.strip().strip('"') for col in columns]
            return columns
        return []

    def extract_column_names(self, sql_query):
        # Use regex to find the columns between SELECT and FROM
        match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
        if match:
            columns = match.group(1).split(',')
            # Strip quotes and whitespace
            columns = [col.strip().strip('"') for col in columns]
            return columns
        return []

    def extract_table_name(self, sql_query):
        # Use regex to find the table name after FROM
        match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def get_column_names(self, table_name):
        if not table_name:
            return []
        query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}';"
        columns = ast.literal_eval(self.db.run(query))
        # Flatten the list of tuples
        column_names = [col[0] for col in columns]
        return column_names

    def correct_sql_query(self, query):
        """
        Corrects common issues in an SQL query.
        Specifically, it ensures that string literals are correctly formatted.
        """
        # Correct the LIKE patterns
        def fix_like_pattern(match):
            column = match.group(1)
            pattern = match.group(2).strip()  # Remove existing wildcards and whitespace
            return f'{column} LIKE \'%{pattern}%\''

        # Correct the ILIKE patterns
        def fix_ilike_pattern(match):
            column = match.group(1)
            pattern = match.group(2).strip()  # Remove existing wildcards and whitespace
            return f'{column} ILIKE \'%{pattern}%\''

        # Correct equality patterns for string literals
        def fix_equality_pattern(match):
            column = match.group(1)
            value = match.group(2).strip()  # Remove existing quotes and whitespace
            return f'{column} = \'{value}\''

        # Find and correct LIKE and ILIKE clauses
        query = re.sub(r'(\w+\sLIKE\s)([^;\s]+)', fix_like_pattern, query, flags=re.IGNORECASE)
        query = re.sub(r'(\w+\sILIKE\s)([^;\s]+)', fix_ilike_pattern, query, flags=re.IGNORECASE)
        # Find and correct equality clauses for string literals
        query = re.sub(r'(\w+\s*=\s*)([^;\s]+)', fix_equality_pattern, query)

        return query


    def create_db_instance(self, userQuery):
        data = []
        column_names = []
        try:
            chain = create_sql_query_chain(self.llm, self.db)
            userQuestion = userQuery
            botString = """
            and for records and listings query in listings table, always include listingid. 
            Remember, 'Berkshire Hathaway HomeServices' is not related to the listings data unless the user explicitly asks for a comparison. 
            If the user asks for general listings data, avoid checking for 'Berkshire Hathaway HomeServices' in any fields of the listings table. 
            For example:
            1. If the user asks 'please provide some listings data and Berkshire Hathaway HomeServices info', only provide listings data and avoid including 'Berkshire Hathaway HomeServices' in the SQL query.
            2. If the user explicitly asks 'compare listings data with Berkshire Hathaway HomeServices', then include the comparison in the SQL query.
            """
            response = chain.invoke({"question":  f"{userQuestion}{botString}"})
            response_query = response.strip().replace('`', '').replace("''", "").replace("'''", "")
            corrected_query = self.correct_sql_query(response_query)
            # Check if the query is SELECT *
            if 'SELECT *' in corrected_query:
                # table_name = self.extract_table_name(corrected_query)
                column_names = self.get_column_names("listings")
            else:
                # Extract column names from the SQL query
                column_names = self.extract_column_names(corrected_query)
            # print("Response Query", corrected_query)
            data =self.db.run(corrected_query)
            # print("data", data)
            return {"data":data, "column_names":column_names}
        except Exception as e:
            print("Error...", e)
            return {"error": str(e), "data": data, "column_names": column_names}
        