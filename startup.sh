#!/bin/bash
echo "Setting up virtual environment"
python3 -m venv env
source env/bin/activate

echo "Installing dependencies and upgrading pip"
pip install -r requirements.txt

echo "Starting unicorn server"
uvicorn app:app --reload
echo "Server started"

