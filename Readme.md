# Image Classification Flask API

This project is a Flask-based image classification API. It uses pre-trained CNN models to predict the class of an uploaded image.

## Quick Start Guide

### 1. Clone the Repository

Open a terminal and run:
git clone https://github.com/hikartik/Classification_app.git
cd your_repo

2. Set Up Your Environment (Local or VM)

Create a Virtual Environment (optional but recommended):

On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate

Install Dependencies:

pip install --upgrade pip
pip install -r requirements.txt


3. Running the Application

After installing the dependencies, run:
python app.py
Then open your browser and navigate to:
http://localhost:5000
(If running on a VM, replace localhost with your VM’s public IP address.)



Project Structure Overview

project/
├── app.py                   # Flask application code
├── Dockerfile               # Docker build instructions
├── requirements.txt         # Python dependencies list
├── model/                   # Folder containing saved model weights (.pth files)
├── templates/               # HTML templates for the web interface
│   └── index.html
└── static/                  # Static files (CSS, images, etc.)
    └── style.css
