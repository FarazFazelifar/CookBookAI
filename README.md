# CookBookAI: Advanced AI-Powered Culinary Assistant

CookBookAI is a sophisticated, AI-driven cooking assistant that harnesses the power of natural language processing, machine learning, and advanced reasoning techniques to revolutionize your culinary experience. By combining Retrieval Augmented Generation (RAG), Chain of Thought (CoT) reasoning, and ReAct methodologies, CookBookAI offers unparalleled recipe generation, modification, and culinary advice.

## Features

- **Intelligent PDF Processing**: Automatically extract, chunk, and index recipes from PDF cookbooks using PyPDFLoader and RecursiveCharacterTextSplitter.
- **Advanced Vector Storage**: Utilize FAISS for efficient similarity search and retrieval of recipe information.
- **Dual LLM Approach**: Leverage two Ollama models for specialized tasks:
 - llama3.1:8b-instruct-q8_0 for instruction following and structured output
 - llama3.1:latest with higher temperature for creative recipe generation and modification
- **Retrieval Augmented Thoughts (RAT)**: Enhance recipe generation through iterative retrieval and refinement.
- **Chain of Thought (CoT) Reasoning**: Implement step-by-step reasoning for complex culinary tasks.
- **ReAct Methodology**: Combine reasoning and acting for context-aware responses and recommendations.
- **Dynamic Recipe Modification**: Adapt existing recipes to dietary restrictions or ingredient substitutions.
- **Personalized Dish Recommendations**: Generate tailored suggestions based on user preferences and recipe history.
- **Culinary Q&A System**: Provide detailed, context-aware answers to cooking-related questions.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/CookBookAI.git
cd CookBookAI
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Install Ollama and download required models:
Follow Ollama installation instructions from https://ollama.ai/
```
ollama pull llama3.1:8b-instruct-q8_0
ollama pull llama3.1:latest
```

## Usage

1. Place PDF cookbooks in the `cookbooks/` directory.

2. Follow the CookBookAI.ipynb notebook.

3. Follow the interactive prompts to:
- Process PDF cookbooks
- Generate new recipes
- Modify existing recipes
- Get dish recommendations
- Ask culinary questions

## Project Structure

- `cookbook_ai.py`: Core CookBookAI class and CLI interface
- `cookbooks/`: Storage for PDF cookbooks
- `cookbook_vectordb/`: FAISS vector database for processed recipes

## Technical Deep Dive

### PDF Processing and Indexing
- Utilizes PyPDFLoader for efficient PDF parsing
- Implements RecursiveCharacterTextSplitter for optimal text chunking
- Enhances metadata with source PDF, page numbers, and unique chunk IDs

### Vector Embedding and Storage
- Employs HuggingFaceEmbeddings with the "sentence-transformers/all-MiniLM-L6-v2" model
- Utilizes FAISS for high-performance similarity search and retrieval

### Retrieval Augmented Thoughts (RAT) Process
You can read all about RAT in [this paper](https://arxiv.org/abs/2403.05313)
1. Initial recipe plan generation using the instruction-following LLM
2. Iterative refinement through multiple cycles of:
- Relevant information retrieval from the vector store
- Creative expansion and improvement using the higher-temperature LLM

### Advanced Prompt Engineering
- Implements structured JSON outputs for easy parsing and utilization
- Utilizes markdown formatting for improved prompt readability
- Incorporates few-shot learning examples in prompts for better context

### Chain of Thought (CoT) Reasoning
- Breaks down complex tasks into step-by-step thought processes
- Enhances recipe generation, modification, and question-answering capabilities

### ReAct Methodology
- Combines reasoning steps with action steps for more dynamic and context-aware responses
- Improves the system's ability to handle multi-step culinary tasks and queries

## Dependencies

The dependencies are listed in the `reqirements.txt` file

## Future Enhancements

- Integration with computer vision models for recipe generation from food images
- Incorporation of nutritional analysis and meal planning features
- Development of a web-based user interface for broader accessibility
- Implementation of multi-modal recipe representation (text, images, videos)
- Enhancement of the recommendation system with collaborative filtering techniques

## Disclaimer

CookBookAI is an AI-assisted tool and should be used as a creative aid and reference. Always exercise caution and common sense when following AI-generated recipes, especially regarding food safety and allergies.