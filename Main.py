# Import Libraries
import json
import os
import requests
from dotenv import load_dotenv
from langchain.evaluation.scoring.prompt import SYSTEM_MESSAGE
from openai import OpenAI
import anthropic
import gradio as gr
import ollama
from IPython.display import Markdown, display
from pathlib import Path


# Global constant

MODEL_GPT = 'gpt-4o-mini' #'gpt-4.1-nano', 'gpt-4.1', 'o3-mini'
MODEL_CLAUDE = "claude-3-7-sonnet-latest" #'claude-3-haiku-20240307'
MODEL_LLAMA = 'llama3.2'
HISTORY_DIR = Path('history')
HISTORY_DIR.mkdir(exist_ok=True)
SYSTEM_MESSAGE = ("You are an AI and Python developer. You are good at programming and writing code. You are "
                  "also good at explaining complex concepts and providing detailed answers to questions.")
# Load API key

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
llama_api_key = 'ollama'

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

# Connect to OpenAI, Anthropic

openai = OpenAI()
claude = anthropic.Anthropic()
# use the OpenAI client python library to call Ollama:
# ollama_via_openai = OpenAI(base_url = 'http://localhost:11434/v1', api_key = llama_api_key)

# Function to save chat history
def save_chat_history(history, model_name):
    # Create timestamp for the filename
    timestamp = history[-1]['timestamp']
    # Create the filename
    filename = f"{model_name}_{timestamp}.json"
    filepath = HISTORY_DIR / filename
    # Format the chat history for saving
    formatted_history = []
    for entry in history:
        formatted_history.append({
            "user": entry[0],
            "assistant": entry[1]
        })
    # Save the chat history to a JSON file
    with open(filepath, 'w') as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "conversation": formatted_history
        }, f, indent=2)
    return filepath.name

# Function to list available chat histories
def list_chat_histories():
    files = list(HISTORY_DIR.glob('*.json'))
    return [f.name for f in files]

# Function to load chat history
def load_chat_history(filename):
    if not filename:
        return []

    filepath = HISTORY_DIR / filename
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            history = [(exchange['user'], exchange['assistant']) for exchange in data['conversation']]
            return history
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

# Function to generate response from each model
def chat_stream_gpt(message, history):
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages,
        stream=True
    )
    result = ""

    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def chat_stream_claude(message, history):
    cleaned_history = []
    for msg in history:
        cleaned_msg = {"role": msg["role"], "content": msg["content"]}
        cleaned_history.append(cleaned_msg)

    messages = cleaned_history + [{"role": "user", "content": message}]

    result = claude.messages.stream(
        model=MODEL_CLAUDE,
        max_tokens=4000,
        temperature=0.7,
        system=SYSTEM_MESSAGE,
        messages=messages,
    )
    response = ""

    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


def chat_stream_llama(message, history):
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history + [{"role": "user", "content": message}]
    try:
        stream = ollama.chat(
            model=MODEL_LLAMA,
            messages=messages,
            stream=True
        )
        result = ""

        for chunk in stream:
            if chunk['message']['content'] and len(chunk['message']['content']) > 0:
                content = chunk['message']['content']
                if content:
                    result += content
                    yield result

    except Exception as e:
        print(f"Error in stream_llama: {e}")
        yield f"Error: {str(e)}"

# Chat Interface with option to select LLMs
def chat_model(message, history, model):
    if model == "GPT":
        result = chat_stream_gpt(message, history)
    elif model == "Claude":
        result = chat_stream_claude(message, history)
    elif model == "Llama":
        result = chat_stream_llama(message, history)
    else:
        raise ValueError("Unknown model")

    yield from result

# Launch chat interface with history with option of different LLMs
system_message = "You are an AI and Python developer. You are good in explaining complex things in a simpler way"

view = gr.ChatInterface(
    fn = chat_model,
    additional_inputs = [gr.Dropdown(["GPT", "Claude", "Llama"], label= "Select model", value= "GPT")],
    type="messages"
)
view.launch()