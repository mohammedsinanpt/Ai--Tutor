# Enhanced AI Tutor Backend with Improved Error Handling and Configuration 🧠

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_tutor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangChain imports with better error handling
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain components loaded successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning(f"⚠️  LangChain not available: {e}")

# Configuration class
class Config:
    """Application configuration"""
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # RAG Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", 3))
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Document paths
    DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "../documents")
    LOG_FILE = os.getenv("LOG_FILE", "ai_tutor.log")

config = Config()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Tutor RAG Backend",
    description="Production-ready conversational AI tutor with RAG pipeline",
    version="1.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced data models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation")

class Response(BaseModel):
    text: str = Field(..., description="AI response text")
    emotion: str = Field(..., description="Mascot emotion state")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    sources: Optional[List[str]] = Field(default=None, description="Document sources used")

class HealthResponse(BaseModel):
    status: str
    rag_system: str
    documents_loaded: int
    conversations: int
    langchain_available: bool
    uptime: str

# Global variables for RAG system
vectorstore = None
embeddings = None
documents_loaded = 0
conversation_memory = []
startup_time = datetime.now()

# Enhanced emotion detection
def determine_emotion(question: str, answer: str) -> str:
    """Enhanced emotion detection with more context awareness"""
    try:
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Greeting patterns
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "thanks", "thank you"]
        if any(word in question_lower for word in greeting_words):
            return "happy"
        
        # Confusion/difficulty patterns
        difficulty_words = ["difficult", "hard", "confused", "don't understand", "help", "stuck", "problem"]
        if any(word in question_lower for word in difficulty_words):
            return "thinking"
        
        # Learning/curiosity patterns
        learning_words = ["learn", "teach", "explain", "how", "what", "why", "show me", "tell me"]
        if any(word in question_lower for word in learning_words):
            return "excited"
        
        # Error/apologetic responses
        if any(word in answer_lower for word in ["error", "sorry", "apologize", "can't", "unable"]):
            return "sad"
        
        # Positive/successful responses
        success_words = ["great", "excellent", "perfect", "correct", "well done", "good job"]
        if any(word in answer_lower for word in success_words):
            return "happy"
        
        return "neutral"
        
    except Exception as e:
        logger.warning(f"Error in emotion detection: {e}")
        return "neutral"

def load_documents_with_langchain():
    """Enhanced document loading with better error handling"""
    global vectorstore, embeddings, documents_loaded
    
    try:
        logger.info("🔄 Initializing RAG pipeline with LangChain...")
        
        # Initialize embeddings with error handling
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"✅ Embeddings model loaded: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings model: {e}")
            raise
        
        # Load documents with enhanced error handling
        documents = []
        documents_folder = Path(config.DOCUMENTS_FOLDER)
        
        if documents_folder.exists():
            logger.info(f"📂 Loading documents from: {documents_folder}")
            
            for file_path in documents_folder.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                        if content:  # Only add non-empty documents
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_path.name,
                                    "file_path": str(file_path),
                                    "file_size": file_path.stat().st_size,
                                    "loaded_at": datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                            logger.info(f"📄 Loaded document: {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {file_path}: {e}")
        
        # Add comprehensive default knowledge if no documents found
        if not documents:
            logger.info("📚 No custom documents found, loading default knowledge base...")
            default_knowledge = [
                # Python Programming
                "Python is a high-level, interpreted programming language known for its simplicity and readability. It's excellent for beginners and widely used in web development, data science, artificial intelligence, and automation.",
                
                # AI and Machine Learning
                "Artificial Intelligence (AI) is the simulation of human intelligence in machines. Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                
                # RAG and Vector Databases
                "Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It uses vector databases to store and search through documents using semantic similarity.",
                
                # Web Development
                "APIs (Application Programming Interfaces) are sets of protocols and tools for building software applications. REST APIs use HTTP methods like GET, POST, PUT, and DELETE to communicate between systems.",
                
                # LangChain Framework
                "LangChain is a framework for developing applications powered by language models. It provides tools for document loading, text splitting, embeddings, vector stores, and chaining operations together.",
                
                # FastAPI Framework
                "FastAPI is a modern, fast web framework for building APIs with Python. It's based on standard Python type hints and provides automatic API documentation, data validation, and high performance.",
                
                # Speech Technologies
                "Speech Recognition (STT) converts spoken words into text, while Text-to-Speech (TTS) converts written text into spoken words. These technologies enable voice interfaces and accessibility features.",
                
                # Vector Embeddings
                "Text embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search and similarity matching in AI applications.",
                
                # Conversational AI
                "Conversational AI systems use natural language processing to understand and respond to human language. They maintain context across multiple turns of conversation for more natural interactions.",
                
                # Data Science
                "Data Science combines statistics, programming, and domain expertise to extract insights from data. Python libraries like pandas, numpy, and scikit-learn are commonly used for data analysis and machine learning."
            ]
            
            documents = [
                Document(
                    page_content=content,
                    metadata={
                        "source": "default_knowledge",
                        "topic": f"topic_{i}",
                        "loaded_at": datetime.now().isoformat()
                    }
                ) for i, content in enumerate(default_knowledge)
            ]
        
        # Split documents into chunks with enhanced configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"📝 Created {len(splits)} document chunks")
        
        # Create vector store with error handling
        try:
            vectorstore = FAISS.from_documents(splits, embeddings)
            documents_loaded = len(splits)
            logger.info(f"✅ Vector store created with {documents_loaded} chunks")
            
            # Save vector store for future use (optional)
            # vectorstore.save_local("./vector_store")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Error in RAG setup: {e}")
        return False

def enhanced_rag_query(question: str) -> tuple[str, List[str]]:
    """Enhanced RAG implementation with source tracking"""
    
    if not vectorstore or not documents_loaded:
        return "RAG system not initialized properly.", []
    
    try:
        # Retrieve relevant documents with scores
        relevant_docs = vectorstore.similarity_search_with_score(
            question, 
            k=config.MAX_RETRIEVED_DOCS
        )
        
        if not relevant_docs:
            return "I don't have specific information about that topic in my knowledge base.", []
        
        # Extract contexts and sources
        contexts = []
        sources = []
        
        for doc, score in relevant_docs:
            contexts.append(doc.page_content)
            source = doc.metadata.get('source', 'unknown')
            if source not in sources:
                sources.append(source)
        
        # Combine context
        context = "\n".join(contexts)
        
        # Enhanced response generation
        question_lower = question.lower()
        
        # Base response from retrieved context
        if context.strip():
            response = f"Hy,Based on the relevant information: {context[:500]}..."
            
            # Add contextual enhancements
            if "python" in question_lower:
                response += "\n\nPython's versatility and extensive library ecosystem make it ideal for beginners and professionals alike!"
            elif any(word in question_lower for word in ["ai", "ml", "machine learning", "artificial intelligence"]):
                response += "\n\nAI and machine learning are rapidly transforming industries and creating new possibilities for solving complex problems!"
            elif any(word in question_lower for word in ["api", "rest", "fastapi"]):
                response += "\n\nAPIs are the backbone of modern web applications, enabling seamless integration between different services!"
            elif any(word in question_lower for word in ["rag", "vector", "embedding"]):
                response += "\n\nRAG systems like this one combine the power of retrieval and generation for more accurate and contextual responses!"
            else:
                response += "\n\nWould you like me to explain any of these concepts in more detail?"
                
        else:
            response = "I don't have specific information about that topic, but I'd be happy to discuss related concepts I do know about!"
        
        return response, sources
        
    except Exception as e:
        logger.error(f"Error in RAG processing: {e}")
        return f"I encountered an issue processing your question. Please try rephrasing it.", []

def fallback_query(question: str) -> tuple[str, List[str]]:
    """Enhanced fallback responses when LangChain isn't available"""
    
    fallback_responses = {
        "python": ("Python is an excellent programming language for AI development! It has powerful libraries like LangChain, FastAPI, NumPy, and many machine learning frameworks that make complex tasks simple.", ["fallback_knowledge"]),
        
        "ai": ("Artificial Intelligence involves creating smart computer systems that can understand, learn, and respond to human needs. It's used in everything from chatbots to self-driving cars!", ["fallback_knowledge"]),
        
        "ml": ("Machine Learning helps computers automatically learn patterns from data without explicit programming. It's like teaching a computer to recognize patterns the same way humans do!", ["fallback_knowledge"]),
        
        "api": ("APIs are like bridges that let different software programs communicate with each other. For example, this frontend uses APIs to talk to our backend server!", ["fallback_knowledge"]),
        
        "rag": ("Retrieval-Augmented Generation (RAG) combines searching through documents with generating new text. It's like having a smart assistant that looks up information and then explains it to you!", ["fallback_knowledge"]),
        
        "vector": ("Vector databases store numerical representations of text that capture meaning. They allow computers to understand that 'dog' and 'puppy' are related concepts!", ["fallback_knowledge"]),
        
        "langchain": ("LangChain is a powerful framework for building AI applications. It provides tools for document processing, embeddings, and chaining different AI operations together!", ["fallback_knowledge"]),
        
        "fastapi": ("FastAPI is a modern Python framework for building high-performance APIs. It automatically generates documentation and provides excellent developer experience!", ["fallback_knowledge"])
    }
    
    question_lower = question.lower()
    
    for key, (response, sources) in fallback_responses.items():
        if key in question_lower:
            return response, sources
    
    return ("That's an interesting question! I'm here to help you learn about AI, Python programming, machine learning, and web development. Feel free to ask about any of these topics!", ["fallback_knowledge"])

# Enhanced error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event with enhanced initialization
@app.on_event("startup")
async def startup_event():
    """Enhanced startup with comprehensive logging"""
    logger.info("🚀 Starting AI Tutor RAG Backend...")
    logger.info(f"📋 Configuration:")
    logger.info(f"   - Host: {config.API_HOST}:{config.API_PORT}")
    logger.info(f"   - Debug mode: {config.DEBUG}")
    logger.info(f"   - Embedding model: {config.EMBEDDING_MODEL}")
    logger.info(f"   - Documents folder: {config.DOCUMENTS_FOLDER}")
    
    if LANGCHAIN_AVAILABLE:
        success = load_documents_with_langchain()
        if success:
            logger.info("✅ RAG system initialized successfully")
        else:
            logger.warning("⚠️  RAG initialization failed, using fallback mode")
    else:
        logger.warning("⚠️  LangChain not available, running in fallback mode")
    
    logger.info("✅ Backend startup complete!")

# Enhanced API endpoints

@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    """Enhanced single query endpoint with better error handling"""
    try:
        logger.info(f"📝 Query received: {request.question[:100]}...")
        
        # Process with enhanced RAG or fallback
        if LANGCHAIN_AVAILABLE and vectorstore:
            answer, sources = enhanced_rag_query(request.question)
        else:
            answer, sources = fallback_query(request.question)
        
        emotion = determine_emotion(request.question, answer)
        
        response = Response(
            text=answer,
            emotion=emotion,
            sources=sources
        )
        
        logger.info(f"✅ Query processed successfully, emotion: {emotion}")
        return response
    
    except Exception as e:
        logger.error(f"❌ Error in query endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Query processing failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/chat", response_model=Response)
async def chat_endpoint(request: ChatRequest):
    """Enhanced multi-turn conversation endpoint"""
    try:
        logger.info(f"💬 Chat message received: {request.message[:100]}...")
        
        # Store conversation with enhanced metadata
        conversation_entry = {
            "message": request.message,
            "history_length": len(request.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
        conversation_memory.append(conversation_entry)
        
        # Process with context awareness
        context_aware_query = request.message
        if request.conversation_history:
            # Simple context enhancement (can be improved with more sophisticated methods)
            recent_context = " ".join([
                entry.get("message", "") for entry in request.conversation_history[-3:]
            ])
            context_aware_query = f"Context: {recent_context} Current question: {request.message}"
        
        # Process with enhanced RAG or fallback
        if LANGCHAIN_AVAILABLE and vectorstore:
            answer, sources = enhanced_rag_query(context_aware_query)
        else:
            answer, sources = fallback_query(context_aware_query)
        
        emotion = determine_emotion(request.message, answer)
        
        response = Response(
            text=answer,
            emotion=emotion,
            sources=sources
        )
        
        logger.info(f"✅ Chat processed successfully, emotion: {emotion}")
        return response
    
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health", response_model=HealthResponse)
async def enhanced_health_check():
    """Enhanced health check with detailed system information"""
    uptime = datetime.now() - startup_time
    uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
    
    return HealthResponse(
        status="healthy",
        rag_system="active" if (LANGCHAIN_AVAILABLE and vectorstore) else "fallback",
        documents_loaded=documents_loaded,
        conversations=len(conversation_memory),
        langchain_available=LANGCHAIN_AVAILABLE,
        uptime=uptime_str
    )

@app.get("/")
async def root():
    """Enhanced root endpoint with system information"""
    return {
        "message": "🤖 AI Tutor RAG Backend - Production Ready!",
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "rag_enabled": LANGCHAIN_AVAILABLE and vectorstore is not None,
            "documents_loaded": documents_loaded,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "conversation_memory": len(conversation_memory)
        },
        "endpoints": {
            "query": "POST /query - Single question answering",
            "chat": "POST /chat - Multi-turn conversation",
            "health": "GET /health - System health check",
            "docs": "GET /docs - API documentation (debug mode only)"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Tutor RAG Backend...")
    print("📋 Production-ready features:")
    print("   ✅ Enhanced error handling and logging")
    print("   ✅ Configuration management")
    print("   ✅ Comprehensive health monitoring")
    print("   ✅ Source tracking in responses")
    print("   ✅ Context-aware conversations")
    print("   ✅ Fallback systems")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info" if config.DEBUG else "warning"
    )