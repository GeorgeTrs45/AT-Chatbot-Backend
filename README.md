# AT-chatbot-Backend
##Python version required##
python3 --version(use 3.9)
##command:##
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload

##Issue faced:##
Error: Could not find a version that satisfie psycopg2-client==2.9.9
Fix: remove psycopg2 from requirements.txt and install other packages. install command seprately 'pip install psycopg2'. run (source command if not running) and run pip install -r requirements.txt

##Setup DB##
create a account on timescale db. Change connection_string and API key from app.py and sqlAgent.py