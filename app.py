import os
import logging
from flask import Flask , render_template , request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from salesgpt.agents import SalesGPT
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import sys 
import io


load_dotenv()


logging.getLogger().setLevel(logging.ERROR)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")


llm = HuggingFaceHub(
     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
     model_kwargs = {'temperature': 0.8, 'max_length':1000}

)

from data import kb  # Ensure data.py is structured correctly to export 'kb'

instruction = """
Role: You are an AI bot for a lawyer's website, tasked with providing accurate and professional answers to user questions.
Response Style: Answer the question directly based on the information provided without making up any details.
Proactivity: Provide complete and helpful responses, using the Q&A data for reference.
Accuracy: If the information is not in the provided Q&A data, indicate that the information is not available.
Clarity: Respond to user questions clearly and concisely, using the provided Q&A data as a reference.
Brevity: Keep your responses as concise as possible while still providing necessary information.
"""

prompt = PromptTemplate(
    input_variables=["user_input", "kb", "instruction"],
    template="{instruction}\n\nQ&A Data:\n{kb}\n\nUser Question:\n{user_input}\nAI Response:"
)


# Initialize LLMChain with Mixtral model
hub_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)


app = Flask(__name__)
CORS(app)


# Generate response
def generate_response(user_input):
    response = hub_chain.run({
        "user_input": user_input,
        "kb": kb,
        "instruction": instruction
    })
    # Process the response to remove the instruction and Q&A data
    processed_response = response.split("AI Response:")[-1].strip()
    return processed_response

@app.route("/")
def home():
    return render_template("banner.html")

@app.route('/get-response', methods=['POST'])
def get_response():
    user_input = request.json['message'].lower().strip()
    user_id = request.json.get('user_id')  # Get user/session identifier from the request
    if not user_id:
        return jsonify({'response': "Error: User ID is missing or invalid."})

    response = generate_response(user_input)  # Generate response from the AI
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

    