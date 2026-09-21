🤖 RAG Assistant — AI Document Assistant
An AI-powered Research & Document Assistant built with Python, Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq LLMs.

RAG Assistant allows users to upload and interact with multiple documents, ask questions using text or voice, perform web research, generate AI-powered document analysis reports, manage private conversations, open websites through voice/chat commands, and generate images using natural-language prompts.

✨ Features
📄 1. Multi-Format Document Upload
Upload and process multiple documents.

Supports up to 10 files at a time.

Supports:

PDF
DOC
DOCX
TXT
CSV
XLS
XLSX
PPTX
PPT
RTF
ODT
Markdown
JSON
HTML
🧠 2. RAG-Based Question Answering
Retrieval-Augmented Generation (RAG).
Automatic document text extraction.
Intelligent text chunking.
Hugging Face sentence-transformer embeddings.
FAISS vector database.
Semantic similarity-based retrieval.
Context-aware answers using Groq LLM.
Answers grounded in uploaded document content.
🔎 3. Intelligent Query Routing
Automatically determines the best source for answering a question:

📄 Document Search
🌐 Web Search
🔄 Document + Web Search
🤖 Direct LLM response when retrieval is not required
This allows the assistant to decide whether a question should be answered from uploaded documents, external web information, or both.

🌐 4. Web Search & Research
Web search using DuckDuckGo/DDGS.
Research current and general information.
Searches the web when required information is not available in uploaded documents.
Combines web results with document context when required.
Provides web-based research responses alongside document-based answers.
📚 5. Source-Aware Answers
Displays sources used to generate answers.

Provides relevant document references such as:

File name
Page number
Slide number
Excel sheet
Helps users verify AI-generated responses against the original content.

📊 6. AI-Powered Document Analysis
Automatically analyzes different types of documents, including:

Research Papers
Resumes / CVs
Project Reports
Technical Documentation
Datasets
Excel / CSV Files
Presentations
Legal Documents
Business Documents
Academic Notes
Certificates
User Manuals
General Documents
Features include:

Automatic document classification.
AI-powered summarization.
Key information extraction.
Structured analysis reports.
Document-specific analysis.
Downloadable analysis reports.
💬 7. Chat & Conversation Management
Create new conversations.
Persistent chat history.
Continue previous conversations.
User-specific conversations.
Recent chat management.
Delete conversations.
Clear chat history.
Persistent conversation storage.
Conversation export.
🔐 8. User Authentication & Session Management
User registration.
User login/logout.
Gmail-based email validation.
Name validation.
Strong password validation.
Duplicate email prevention.
PBKDF2-SHA256 password hashing.
Unique random password salts.
Secure random session tokens.
Persistent login sessions.
Session validation.
Session expiration.
Logout from all devices.
Expired session cleanup.
🎙️ 9. Voice Assistant
Voice-based interaction.
Speech-to-text using Faster-Whisper.
Voice-based document questions.
Voice-based web research.
Voice command recognition.
Noise calibration.
Voice Activity Detection (VAD).
Speech recognition hallucination filtering.
Fuzzy command matching using RapidFuzz.
Supports website and application opening commands.
Example commands:

Open YouTube
Open Google
Open Gmail
Open GitHub
Open LinkedIn
Open Naukri
Open ChatGPT
🌐 10. Website & Application Launcher
Users can open supported websites directly through voice or chat commands.

Examples:

"Open YouTube"
"Open Google"
"Open Gmail"
"Open GitHub"
"Open LinkedIn"
"Open Naukri"
"Open ChatGPT"
The assistant detects the user's intent, identifies the requested website/application, and opens it directly in the browser.

🔊 11. Text-to-Speech
Converts AI responses into speech.
Supports pyttsx3.
Supports Edge TTS.
Cleans Markdown content before speech generation.
Provides a more natural voice interaction experience.
Enables voice-based responses after AI processing.
🖼️ 12. AI Image Generation
Generate images from text prompts.
OpenAI Image Generation API integration.
Supports gpt-image-1.
Converts natural-language descriptions into generated visuals.
⚡ 13. Performance & Reliability
Streamlit caching for expensive resources.
Cached LLM initialization.
Cached embedding models.
Persistent FAISS vector database.
Response-time tracking.
API rate-limit handling.
User-friendly error handling.
Retry handling for transient API failures.
Relevance/similarity filtering for retrieved content.
🔒 14. Privacy & Data Isolation
User-specific authentication.
User-specific conversations.
Session-based access control.
Passwords are never stored in plain text.
PBKDF2-SHA256 password hashing.
Sensitive API credentials stored through environment variables.
.env support for API credentials.
Separate conversation storage for users.
🚀 15. AI Research Assistant Workflow
The application combines document understanding, web research, voice interaction, and AI generation into a single workflow.

                    User
                     │
            ┌────────┴────────┐
            │                 │
           Chat             Voice
            │                 │
            └────────┬────────┘
                     ↓
              Query Processing
                     ↓
           Intelligent Query Routing
                     │
          ┌──────────┼──────────┐
          │          │          │
          ↓          ↓          ↓
      Documents     Web        Both
       Search      Search     Sources
          │          │          │
          └──────────┼──────────┘
                     ↓
                  Groq LLM
                     ↓
              AI Generated Answer
                     ↓
              Sources / References
🏗️ RAG Pipeline
Upload Documents
       ↓
Document Loading
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Database
       ↓
User Question
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
Groq LLM
       ↓
Grounded AI Answer
       ↓
Source References
🎙️ Voice Assistant Pipeline
Microphone
    ↓
Noise Calibration
    ↓
Voice Activity Detection
    ↓
Faster-Whisper
    ↓
Speech → Text
    ↓
Query / Command Detection
    ↓
┌───────────────┬────────────────┐
│               │                │
RAG Query     Web Search      Website Command
│               │                │
└───────────────┴────────────────┘
        ↓
    AI Response
        ↓
   Text-to-Speech
🧰 Technology Stack
Programming & UI
Python
Streamlit
AI & LLM
Groq API
LangChain
Hugging Face Sentence Transformers
OpenAI Image Generation API
RAG & Search
FAISS
DuckDuckGo / DDGS
LangChain Document Processing
Voice
Faster-Whisper
RapidFuzz
SoundDevice
Voice Activity Detection
Text-to-Speech
Edge TTS
pyttsx3
Authentication & Storage
SQLite
PBKDF2-SHA256
Secure Session Tokens
Document Processing
PyPDF
python-docx
Pandas
OpenPyXL
python-pptx
Configuration
python-dotenv
📁 Project Structure
RAG Assistant/
│
├── app.py                      # Main Streamlit application
│
├── auth.py                     # Authentication and session logic
├── auth_ui.py                  # Login and signup UI
├── session_state.py            # Streamlit session-state helpers
│
├── chat_manager.py             # Chat history management
├── share_utils.py              # Sharing/export utilities
│
├── ingest.py                   # Document loading and preprocessing
├── rag.py                      # RAG functionality
├── rag_engine.py               # Core retrieval and generation pipeline
├── qa_handler.py               # Question-answering orchestration
│
├── analysis_report.py          # Document analysis/report generation
│
├── voice_assistant.py          # Voice assistant functionality
├── tts.py                      # Text-to-speech utilities
│
├── image_generator.py          # AI image generation
│
├── conversations/              # Per-user conversation data
├── users.db                    # SQLite authentication database
│
├── requirements.txt            # Python dependencies
├── .env                        # API credentials
├── .gitignore                  # Git ignored files
└── README.md
🚀 Installation & Setup
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
2. Create a Virtual Environment
Windows
python -m venv rag_env
Activate the environment:

rag_env\Scripts\activate
Linux / macOS
python3 -m venv rag_env
source rag_env/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY is required only for the AI image-generation functionality.

⚠️ Never commit .env or API keys to GitHub.

5. Run the Application
streamlit run app.py
The application will be available at:

http://localhost:8501
💡 Example Usage
📄 Ask Questions About Documents
Upload a document and ask:

Summarize this document.
What methodology was used?
What are the key findings?
Explain this document in simple terms.
🌐 Perform Web Research
Ask questions such as:

What are the latest developments in this technology?
If the required information is not available in the uploaded documents, the assistant can use web search.

🎙️ Use Voice Commands
You can interact with the assistant using voice commands:

Open YouTube
Open Google
Open Gmail
Open GitHub
Open LinkedIn
Open Naukri
You can also ask document and research questions through voice.

💬 Use Chat Commands
The same website-opening functionality is available through text chat:

Open YouTube
Open GitHub
Open Gmail
The assistant detects the command and opens the requested website.

📊 Document Analysis
The document analysis system can automatically identify and analyze different types of content.

Example workflow:

Upload Document
       ↓
Document Classification
       ↓
Content Analysis
       ↓
Information Extraction
       ↓
AI Summarization
       ↓
Structured Analysis Report
       ↓
Download Report
🔐 Authentication & Data Storage
The application uses SQLite for authentication and local storage.

User Database
users.db
Stores:

User accounts
Hashed passwords
Authentication/session information
Conversation Storage
conversations/
Stores user-specific conversation history.

Example:

conversations/
│
├── user1/
│   ├── conversation_1.json
│   └── conversation_2.json
│
└── user2/
    └── conversation_1.json
🛡️ Security
The authentication system includes:

PBKDF2-SHA256 password hashing
Unique password salts
Secure random session tokens
Password validation
Email validation
Session expiration
User-specific conversation storage
Environment-based API credentials
For production deployment, additional security measures such as HTTPS, secure cookie configuration, CSRF protection, centralized secret management and production-grade authentication should be implemented.

📤 Conversation Export
Users can export their conversations for offline use.

Supported export formats include:

PDF
TXT
This allows users to save important research discussions and AI-generated responses.

⚙️ Groq LLM
The application uses Groq-hosted LLMs for fast AI responses.

Configured models include:

llama-3.3-70b-versatile
llama-3.1-8b-instant
The model can be selected/configured depending on the task and API availability.

🗃️ FAISS Vector Store
FAISS is used for efficient semantic similarity search.

The process is:

Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
FAISS Index
    ↓
Similarity Search
    ↓
Relevant Context
This enables the assistant to retrieve the most relevant document content before generating an answer.

🔄 Query Routing Logic
The assistant can determine whether a query requires:

📄 Document Search
Used when the answer should come from uploaded documents.

"What is the conclusion of this research paper?"
🌐 Web Search
Used when external/current information is required.

"What are the latest developments in AI?"
🔄 Document + Web
Used when both uploaded content and external information are useful.

"Compare the technology in my document with the latest developments."
🤖 Direct LLM
Used for general questions that don't require document or web retrieval.

🧪 Troubleshooting
Missing API Key
Make sure your .env file contains:

GROQ_API_KEY=your_groq_api_key
For image generation:

OPENAI_API_KEY=your_openai_api_key
Restart Streamlit after modifying .env.

Dependency Issues
Make sure the virtual environment is activated:

rag_env\Scripts\activate
Then reinstall dependencies:

pip install -r requirements.txt
Voice Assistant Issues
Ensure the required voice dependencies are installed:

pip install faster-whisper rapidfuzz sounddevice
Additional system-level audio dependencies may be required depending on the operating system.

FAISS / Vector Store Issues
If document retrieval is not working:

Restart the application.
Re-upload the documents.
Rebuild the vector store if required.
Check that the embedding dependencies are installed correctly.
🔮 Future Development
A separate backend architecture is also being developed to evolve this Streamlit application into a more scalable system.

Planned architecture:

Frontend
   ↓
FastAPI Backend
   ↓
PostgreSQL
   ↓
pgvector
   ↓
RAG / AI Services
Potential improvements include:

FastAPI backend
PostgreSQL database
SQLAlchemy
Alembic migrations
pgvector-based vector search
Dedicated frontend
Production-grade authentication
Scalable deployment
Improved document processing
Advanced retrieval and reranking
📌 Project Highlights
✓ Multi-format document processing
✓ Multi-document RAG
✓ Semantic document search
✓ Intelligent query routing
✓ Web research
✓ Source-aware answers
✓ AI document analysis
✓ Persistent private chat history
✓ PDF/TXT conversation export
✓ Secure authentication
✓ Voice-based Q&A
✓ Voice command support
✓ Website launcher through voice & chat
✓ Text-to-speech
✓ AI image generation
✓ FAISS vector search
✓ Groq LLM integration
✓ API rate-limit handling
📜 License
This is a personal/academic project and currently does not specify an open-source license.

⭐ Support
If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

RAG Assistant — Chat with your documents, research the web, and interact with AI using text or voice.
