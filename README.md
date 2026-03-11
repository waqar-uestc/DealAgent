## DealAgent: A Trustworthy Agentic AI Framework for End-to-End Deal Optimization in Consumer Electronics Environments


Artificial intelligence has significantly transformed online shopping through conversational assistants and personalized recommendation systems. Existing approaches primarily provide interaction support while relying on users for decision making and task execution, limiting autonomy in dynamic consumer environments. This paper presents \textit{DealAgent}, a trustworthy agentic AI framework for end-to-end shopping that emphasizes human-centric interaction and adaptive decision support in consumer electronics ecosystems. We formulate autonomous shopping as a constrained sequential decision-making problem and propose a layered architecture that integrates semantic task understanding, hybrid autonomous planning, multi-attribute deal optimization, dynamic preference learning, and secure execution governance. An *initial version of deal agent* is available here with a local AI-powered web app that fetches deals from RSS feeds and analyzes them using LLM evaluation, price prediction, and RAG Q&A.

---

## Setup

1. Add your API key(s) to a `.env` file in the project root:
```env
   OPENAI_API_KEY=your-key-here      # Optional
   DEEPSEEK_API_KEY=your-key-here    # Optional
   GEMINI_API_KEY=your-key-here      # Optional
```
   > At least one API key is required.

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

- **Start**: Run `start.bat`  — opens the app
- **Stop**: Run `stop.bat`


We are preparing additional code segments to upload, will be available soon
