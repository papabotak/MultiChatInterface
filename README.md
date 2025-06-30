# Multi-LLM Chat Interface

A Python-based chat interface that allows users to interact with multiple Language Models (LLMs) including GPT, Claude, and Llama through a unified Gradio interface.

## 🌟 Features

- Support for multiple LLMs:
  - OpenAI GPT
  - Anthropic Claude
  - Llama
- Real-time streaming responses
- Chat history management (save, load, list)
- User-friendly Gradio interface
- Environment variable configuration for API keys

## 🔧 Prerequisites

- Python 3.x
- Required API keys:
  - OpenAI API key
  - Anthropic API key
  - Ollama (for Llama model)

## 📦 Installation

1. Clone the repository: bash git clone <your-repository-url>

2. Install required packages: bash pip install -r requirements.txt

3. Create a `.env` file in the project root and add your API keys: OPENAI_API_KEY=your_openai_api_key ANTHROPIC_API_KEY=your_anthropic_api_key

## 🚀 Usage

1. Run the application: bash python main.py
2. Open your web browser and navigate to the local Gradio interface (typically http://localhost:7860)

3. Select your preferred model from the dropdown menu (GPT, Claude, or Llama)

4. Start chatting!

## 📊 Project Structure

- `main.py`: Main application file containing the chat interface implementation
- `history/`: Directory containing saved chat histories
- `.env`: Configuration file for API keys

## 🛠️ Features in Detail

- **Model Selection**: Choose between different LLM providers
- **Chat History**: Automatically saves conversations with timestamps
- **Streaming Responses**: Real-time response generation
- **System Message**: Customizable system prompt for AI behavior


## 🙏 Acknowledgments

- LangChain team
- Gradio team
- Ollama project
- HuggingFace community

## 👤 Author

- **Shahril Mohd**
- Email: mohd.shahrils@yahoo.com
- Copyright © 2025