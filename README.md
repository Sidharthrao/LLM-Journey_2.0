# LLM-Journey_2.0

A comprehensive learning repository covering essential Large Language Model (LLM) concepts, implementations, and practical applications. This repository contains hands-on notebooks, code examples, and tutorials for understanding and working with modern LLM technologies.

## 📚 Table of Contents

- [Overview](#overview)
- [Modules](#modules)
  - [1. Chunking Strategy](#1-chunking-strategy)
  - [2. RAG](#2-rag)
  - [3. FineTuning](#3-finetuning)
  - [3.a Advanced FineTuning](#3a-advanced-finetuning)
  - [4. LangGraph](#4-langgraph)
  - [4.a AI Agents - SQL](#4a-ai-agents---sql)
  - [4.b AI Agents - Competitor Analysis](#4b-ai-agents---competitor-analysis)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository serves as a practical guide for learning and implementing various LLM techniques. Each module focuses on a specific aspect of LLM development, from basic text processing to advanced agent-based systems. The content is designed for both beginners and intermediate practitioners looking to deepen their understanding of LLM technologies.

## Modules

### 1. Chunking Strategy

Text chunking is a fundamental technique for processing large documents in LLM applications. This module demonstrates various chunking methods:

- **Character Text Splitting**: Fixed-size character-based chunking
- **Recursive Character Text Splitting**: Hierarchical splitting that preserves semantic structure
- **Semantic Chunking**: Embedding-based chunking that splits at semantic boundaries
- **Agentic Chunking**: LLM-powered intelligent chunking that groups related propositions

**Key Files:**
- `advance_chunking.ipynb`: Comprehensive notebook covering all chunking methods
- `Ver1-Agentic_Chunking.py`: Implementation of agentic chunking with semantic grouping
- `0. DifferentMethods of Chunking.csv`: Reference guide for chunking methods

### 2. RAG

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation for more accurate and context-aware responses.

- **Basic RAG**: Implementation with web scraping and PDF processing
- **Multimodal RAG**: Extending RAG to handle multiple data types (text, images, etc.)

**Key Files:**
- `1. Basic_RAG implementation - Web scrapping and PDF file.ipynb`: Basic RAG implementation
- `2. Multimodal_RAG.executed.ipynb`: Multimodal RAG examples

### 3. FineTuning

Fine-tuning pre-trained models for specific tasks using Parameter-Efficient Fine-Tuning (PEFT) and LoRA techniques.

- **PEFT/LoRA Fine-Tuning**: Efficient fine-tuning with Google's Flan-T5 model
- **Evaluation Metrics**: ROUGE score evaluation for model performance
- **Dataset**: Indian Food Dataset for domain-specific fine-tuning

**Key Files:**
- `1. FineTuning & Eval.ipynb`: Fine-tuning workflow and evaluation
- `2. Fine tuning_PEFT_LoRA - Google's Flaunt T5.ipynb`: PEFT/LoRA implementation
- `requirements.txt`: Dependencies for fine-tuning

### 3.a Advanced FineTuning

Advanced fine-tuning techniques using TRL (Transformer Reinforcement Learning) and Supervised Fine-Tuning (SFT).

- **TinyLlama Fine-Tuning**: Fine-tuning TinyLlama-1.1B-Chat model
- **TRL Integration**: Using TRL library for efficient training
- **WandB Integration**: Experiment tracking and monitoring

**Key Files:**
- `AdvanceFineTuning.ipynb`: Advanced fine-tuning notebook
- `instruction-response.csv`: Training dataset
- `requirements.txt`: Dependencies for advanced fine-tuning

### 4. LangGraph

Comprehensive tutorial on building stateful, multi-actor applications with LangGraph.

- **State Management**: Understanding state in LangGraph applications
- **Nodes and Edges**: Building workflow graphs
- **Message Handling**: Managing conversation history and state updates

**Key Files:**
- `langgraph_comprehensive_tutorial.ipynb`: Complete LangGraph tutorial

### 4.a AI Agents - SQL

Building AI agents capable of generating and executing SQL queries.

- **SQL Query Generation**: Using LLMs to generate database queries
- **Database Integration**: Connecting to PostgreSQL databases
- **Agent Architecture**: Building reusable agent classes

**Key Files:**
- `SQL Agent.ipynb`: SQL agent implementation
- `Class File - sql_agent.ipynb`: Reusable SQL agent class
- `requirements.txt`: Dependencies for SQL agents

### 4.b AI Agents - Competitor Analysis

Building intelligent agents for competitor analysis using web scraping and search.

- **Web Scraping**: Extracting competitor information from the web
- **Search Integration**: Using DuckDuckGo and Exa for information retrieval
- **Agent Orchestration**: Coordinating multiple tools for analysis

**Key Files:**
- `Competitior Agents.ipynb`: Competitor analysis agent
- `Class file - compitator_agent.ipynb`: Reusable competitor agent class
- `ClassFile_compitator_agent.py`: Python implementation of competitor agent
- `requirements.txt`: Dependencies for competitor analysis

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** (Python 3.10+ recommended)
- **CUDA-capable GPU** (recommended for fine-tuning modules)
- **API Keys**:
  - OpenAI API key (for most modules)
  - Anthropic API key (optional, for some agent modules)
  - LangSmith API key (optional, for tracing)
  - Hugging Face token (for model downloads)

## Installation

Each module has its own `requirements.txt` file. Install dependencies based on which modules you want to use:

### For Chunking Strategy
```bash
pip install chromadb langchain llama-index langchain_experimental langchain_openai rich
```

### For RAG
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters
```

### For FineTuning
```bash
cd "3. FineTuning"
pip install -r requirements.txt
```

### For Advanced FineTuning
```bash
cd "3.a Advanced FineTuning"
pip install -r requirements.txt
```

### For LangGraph
```bash
pip install langchain langchain-openai langgraph
```

### For SQL Agents
```bash
cd "4.a AI Agents - SQL"
pip install -r requirements.txt
```

### For Competitor Analysis Agents
```bash
cd "4.b AI Agents - Competitor analysis"
pip install -r requirements.txt
```

## Usage

1. **Set up environment variables**: Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   LANGSMITH_API_KEY=your_langsmith_key
   HUGGINGFACE_TOKEN=your_hf_token
   ```

2. **Navigate to the module** you want to explore

3. **Open the Jupyter notebook** and follow along:
   ```bash
   jupyter notebook
   ```

4. **Run cells sequentially** to understand the concepts and see the implementations in action

### Example: Running Basic RAG
```bash
cd "2. RAG"
jupyter notebook "1. Basic_RAG implementation - Web scrapping and PDF file.ipynb"
```

### Example: Fine-Tuning a Model
```bash
cd "3. FineTuning"
jupyter notebook "1. FineTuning & Eval.ipynb"
```

## Project Structure

```
LLM-Journey_2.0/
├── 1. Chunking Strategy/
│   ├── 0. DifferentMethods of Chunking.csv
│   └── Advance/
│       ├── advance_chunking.ipynb
│       └── Ver1-Agentic_Chunking.py
├── 2. RAG/
│   ├── 1. Basic_RAG implementation - Web scrapping and PDF file.ipynb
│   └── 2. Multimodal_RAG.executed.ipynb
├── 3. FineTuning/
│   ├── 1. FineTuning & Eval.ipynb
│   ├── 2. Fine tuning_PEFT_LoRA - Google's Flaunt T5.ipynb
│   ├── DataSet/
│   │   └── IndianFoodDataset.csv
│   ├── peft-dialogue-summary-checkpoint-local/
│   └── requirements.txt
├── 3.a Advanced FineTuning/
│   ├── AdvanceFineTuning.ipynb
│   ├── instruction-response.csv
│   ├── llama-chatbot-finetuned/
│   ├── wandb/
│   └── requirements.txt
├── 4. LangGraph/
│   └── langgraph_comprehensive_tutorial.ipynb
├── 4.a AI Agents - SQL/
│   ├── SQL Agent.ipynb
│   ├── Class File - sql_agent.ipynb
│   └── requirements.txt
├── 4.b AI Agents - Competitor analysis/
│   ├── Competitior Agents.ipynb
│   ├── Class file - compitator_agent.ipynb
│   ├── ClassFile_compitator_agent.py
│   └── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Fine-tuning examples use [Hugging Face Transformers](https://huggingface.co/transformers)
- Agent implementations inspired by [LangGraph](https://github.com/langchain-ai/langgraph)
- PEFT/LoRA techniques from [PEFT](https://github.com/huggingface/peft) library

---

**Note**: This repository is for educational purposes. Make sure to comply with API usage policies and model licensing terms when using commercial APIs and models.
