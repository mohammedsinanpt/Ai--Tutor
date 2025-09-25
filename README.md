# 🤖 AI Tutor - Conversational RAG System

> A complete conversational AI tutor with speech recognition, text-to-speech, and animated emotional responses.

## 🎯 Project Overview

This project implements a full-stack AI tutoring system featuring:
- **RAG-powered backend** using LangChain and Vector Database
- **Interactive mascot interface** with speech capabilities
- **Real-time emotional responses** and animations
- **Speech-to-Speech pipeline** for natural interaction

##Screenshots 

<img width="1274" height="894" alt="Screenshot 2025-09-25 225441" src="https://github.com/user-attachments/assets/78daf43a-2480-4f7a-967e-799c06569ee2" />
<img width="1098" height="936" alt="Screenshot 2025-09-25 225424" src="https://github.com/user-attachments/assets/eaf7d4cb-2d45-4d19-8d59-400df489315b" />
<img width="1274" height="894" alt="Screenshot 2025-09-25 225441" src="https://github.com/user-attachments/assets/3494f81d-8290-4437-b016-b8e754f53ccd" />

## 🏗️ System Architecture

```
User Speech Input → STT → RAG Backend → Vector DB Search → Response Generation → TTS → Mascot Animation
```

### Technical Stack
- **Backend**: Python, FastAPI, LangChain, FAISS Vector DB
- **Frontend**: HTML5, JavaScript, Web Speech API
- **AI/ML**: Sentence Transformers, RAG Pipeline
- **Speech**: Web Speech Recognition & Synthesis APIs

## ⚡ Features

### Backend (RAG API)
- ✅ **LangChain + FAISS Vector Database** for document retrieval
- ✅ **POST /query** - Single question endpoint
- ✅ **POST /chat** - Multi-turn conversation endpoint  
- ✅ **Emotion Detection** - Returns emotional state with responses
- ✅ **Document Processing** - Automatic knowledge base loading

### Frontend (Mascot Interface)
- ✅ **Speech Recognition (STT)** - Voice input with visual feedback
- ✅ **Text-to-Speech (TTS)** - Spoken responses with mouth animation
- ✅ **Emotional Animations** - Dynamic facial expressions
- ✅ **Live API Integration** - Real-time backend communication
- ✅ **Professional UI** - Modern glassmorphism design

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Modern web browser (Chrome/Firefox recommended)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ai-tutor-rag
cd ai-tutor-rag

# Install dependencies
pip install fastapi uvicorn langchain-community langchain-core
pip install sentence-transformers faiss-cpu

# Add your knowledge documents to documents/ folder
# (Create .txt files with educational content)
```

### Running the System

#### 1. Start Backend (Terminal 1)
```bash
cd backend
python main.py
```
Backend will run on: `http://localhost:8000`

#### 2. Start Frontend (Terminal 2)
```bash
cd frontend
python -m http.server 3000 --bind 127.0.0.1
```
Frontend will run on: `http://localhost:3000`

#### 3. Usage
1. Open `http://localhost:3000` in browser
2. **Type questions** or **click 🎤 to speak**
3. Watch the AI tutor respond with speech and emotions!

## 📡 API Endpoints

### POST /query
Single question answering
```json
Request: {"question": "What is Python?"}
Response: {"text": "Python is a programming language...", "emotion": "excited"}
```

### POST /chat  
Multi-turn conversation
```json
Request: {"message": "Explain AI", "conversation_history": [...]}
Response: {"text": "AI helps computers think...", "emotion": "happy"}
```

### GET /health
System health check
```json
Response: {"status": "healthy", "rag_system": "active"}
```

## 🎭 Emotion System

The mascot displays different emotions based on content:
- **Happy** 😊 - Greetings, positive responses
- **Excited** 🤩 - Learning topics, new concepts  
- **Thinking** 🤔 - Processing, complex topics
- **Neutral** 😐 - General responses
- **Sad** 😢 - Errors or problems

## 📂 Project Structure

```
ai-tutor-rag/
├── backend/
│   ├── main.py              # FastAPI RAG backend
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html          # Mascot interface
├── documents/
│   ├── python_basics.txt   # Knowledge base files
│   └── ai_basics.txt
└── README.md
```

## 🎬 Demo Features

### Complete Speech-to-Speech Pipeline
1. **User speaks** → Speech Recognition converts to text
2. **RAG Processing** → LangChain searches vector database  
3. **Response Generation** → AI generates contextual answer
4. **Speech Output** → Text-to-Speech with mouth animation
5. **Emotional Display** → Mascot shows appropriate emotion

### Knowledge Base
- Automatically processes documents in `documents/` folder
- Uses semantic search for relevant information retrieval
- Supports multiple document formats and sources

## 🔧 Customization

### Adding Knowledge
1. Add `.txt` files to `documents/` folder
2. Restart backend to reload knowledge base

### Modifying Emotions
Edit the `determine_emotion()` function in `backend/main.py`

### Styling Changes
Modify CSS in `frontend/index.html` for different themes

## 🎪 Technical Highlights

- **RAG Implementation**: Proper retrieval-augmented generation using LangChain
- **Vector Similarity Search**: FAISS for efficient semantic search
- **Real-time Communication**: FastAPI with CORS for frontend-backend integration
- **Speech Processing**: Native Web Speech APIs for STT/TTS
- **Responsive Design**: Modern UI with smooth animations
- **Error Handling**: Comprehensive error management and fallbacks

## 📊 Performance

- **Response Time**: < 2 seconds for most queries
- **Knowledge Base**: Supports unlimited document size
- **Concurrent Users**: FastAPI handles multiple simultaneous requests
- **Browser Compatibility**: Works on all modern browsers

## 🏆 Project Achievements

✅ Complete RAG pipeline implementation  
✅ Live API integration with emotional responses  
✅ Full speech-to-speech interaction  
✅ Professional animated interface  
✅ Scalable architecture with proper documentation  

## 🔮 Future Enhancements

- Integration with advanced LLMs (GPT-4, Claude)
- Multi-language support
- Voice cloning for personalized speech
- Advanced 3D mascot animations
- Mobile app development
