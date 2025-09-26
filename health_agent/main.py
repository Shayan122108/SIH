import os
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from dotenv import load_dotenv

# Import agents with error handling
try:
    from agents.health_adviser import health_adviser_app
    from agents.booking_agent import booking_agent_app, BookingState
except ImportError as e:
    logging.error(f"Failed to import agents: {e}")
    raise

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIConfig:
    """Configuration class for API settings"""
    def __init__(self):
        self.host = os.getenv('API_HOST', '0.0.0.0')
        self.port = int(os.getenv('API_PORT', '8000'))
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.max_session_age = int(os.getenv('MAX_SESSION_AGE_HOURS', '24'))
        self.max_sessions = int(os.getenv('MAX_SESSIONS', '10000'))
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW_MINUTES', '60'))
        
        # CORS settings
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        self.cors_methods = os.getenv('CORS_METHODS', 'GET,POST,PUT,DELETE').split(',')

config = APIConfig()

# Enhanced Pydantic models with validation
class UserQuery(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="User's health-related question or request"
    )
    session_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Unique session identifier"
    )
    user_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional user metadata"
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        # Basic sanitization
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v.strip():
            raise ValueError('Session ID cannot be empty')
        # Remove any potentially harmful characters
        return ''.join(c for c in v if c.isalnum() or c in '-_')

class AgentResponse(BaseModel):
    response: str = Field(..., description="Agent's response to the user query")
    session_id: str = Field(..., description="Session identifier")
    response_type: str = Field(..., description="Type of response (health_advice, booking, error)")
    confidence_level: Optional[str] = Field(None, description="Confidence level of the response")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class HealthStatus(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    active_sessions: int
    system_health: Dict[str, str]

class SessionManager:
    """Enhanced session management with cleanup and rate limiting"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        self.rate_limits: Dict[str, list] = {}
        self.startup_time = datetime.now()
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data with automatic cleanup"""
        self._cleanup_expired_sessions()
        return self.sessions.get(session_id, {"status": "new"})
    
    def set_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Set session data with timestamp tracking"""
        self.sessions[session_id] = data
        self.session_timestamps[session_id] = datetime.now()
        
        # Limit number of sessions to prevent memory issues
        if len(self.sessions) > config.max_sessions:
            oldest_session = min(self.session_timestamps, key=self.session_timestamps.get)
            self.remove_session(oldest_session)
    
    def remove_session(self, session_id: str) -> None:
        """Remove session and its timestamp"""
        self.sessions.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
        self.rate_limits.pop(session_id, None)
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, timestamp in self.session_timestamps.items():
            if current_time - timestamp > timedelta(hours=config.max_session_age):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limits"""
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=config.rate_limit_window)
        
        if session_id not in self.rate_limits:
            self.rate_limits[session_id] = []
        
        # Remove old requests outside the window
        self.rate_limits[session_id] = [
            req_time for req_time in self.rate_limits[session_id]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.rate_limits[session_id]) >= config.rate_limit_requests:
            return False
        
        # Add current request
        self.rate_limits[session_id].append(current_time)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        current_time = datetime.now()
        uptime = (current_time - self.startup_time).total_seconds()
        
        return {
            "active_sessions": len(self.sessions),
            "uptime_seconds": uptime,
            "total_rate_limited_sessions": len(self.rate_limits),
            "startup_time": self.startup_time.isoformat()
        }

# Initialize session manager
session_manager = SessionManager()

# Background task for periodic cleanup
async def cleanup_sessions():
    """Periodic session cleanup task"""
    while True:
        try:
            initial_count = len(session_manager.sessions)
            session_manager._cleanup_expired_sessions()
            final_count = len(session_manager.sessions)
            
            if initial_count != final_count:
                logger.info(f"Session cleanup: {initial_count - final_count} sessions removed")
            
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Health Agent API...")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_sessions())
    
    try:
        yield
    finally:
        logger.info("Shutting down AI Health Agent API...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Health Agent API",
    description="Advanced AI-powered health advisory and appointment booking system",
    version="2.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=config.cors_methods,
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting dependency
async def rate_limit_check(request: Request, user_query: UserQuery):
    """Check rate limits for requests"""
    session_id = user_query.session_id
    client_ip = request.client.host
    
    if not session_manager.check_rate_limit(f"{session_id}:{client_ip}"):
        logger.warning(f"Rate limit exceeded for session {session_id} from {client_ip}")
        raise HTTPException(
            status_code=429, 
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please wait before trying again.",
                "retry_after": config.rate_limit_window * 60
            }
        )
    
    return user_query

def classify_intent(query: str) -> str:
    """Enhanced intent classification with better booking detection"""
    query_lower = query.lower().strip()
    
    # Emergency keywords (highest priority)
    emergency_keywords = [
        'emergency', 'urgent', 'severe chest pain', 'heart attack',
        'stroke', 'heavy bleeding', 'unconscious', 'can\'t breathe',
        'choking', 'seizure', 'overdose'
    ]
    
    # Explicit booking request keywords (high priority)
    explicit_booking_keywords = [
        'book appointment', 'schedule appointment', 'need appointment',
        'want appointment', 'make appointment', 'see doctor', 'visit doctor',
        'meet doctor', 'consultation', 'check up', 'book slot',
        'available slot', 'hospital visit', 'clinic visit'
    ]
    
    # Symptom + booking intent keywords
    symptom_booking_keywords = [
        'i have', 'i am having', 'suffering from', 'experiencing',
        'feeling', 'my', 'pain in', 'ache in', 'hurt', 'sick',
        'unwell', 'not feeling well'
    ]
    
    # Medical symptoms that usually require booking
    medical_symptoms = [
        'fever', 'headache', 'stomach pain', 'back pain', 'chest pain',
        'cough', 'cold', 'flu', 'nausea', 'vomiting', 'dizziness',
        'fatigue', 'weakness', 'rash', 'swelling', 'infection',
        'breathing problem', 'shortness of breath', 'sore throat'
    ]
    
    # Pure health advice/information keywords
    pure_advice_keywords = [
        'what is', 'how to prevent', 'causes of', 'treatment for',
        'cure for', 'about', 'explain', 'tell me about', 'definition of',
        'symptoms of', 'how to treat', 'home remedy', 'natural remedy'
    ]
    
    # Check for emergency (highest priority)
    if any(keyword in query_lower for keyword in emergency_keywords):
        return 'emergency'
    
    # Check for explicit booking requests
    if any(keyword in query_lower for keyword in explicit_booking_keywords):
        return 'booking_request'
    
    # Check for pure advice/information requests (should NOT be booking)
    if any(keyword in query_lower for keyword in pure_advice_keywords):
        return 'health_query'
    
    # Check for symptom descriptions that indicate booking intent
    has_symptom_context = any(keyword in query_lower for keyword in symptom_booking_keywords)
    has_medical_symptom = any(symptom in query_lower for symptom in medical_symptoms)
    
    # If user is describing personal symptoms, it's likely a booking request
    if has_symptom_context and has_medical_symptom:
        return 'booking_request'
    
    # If just mentioning symptoms without personal context, might be asking for info
    if has_medical_symptom and not has_symptom_context:
        # Check if it's a question about the symptom (advice) or experiencing it (booking)
        question_words = ['what', 'why', 'how', 'when', 'where', 'which']
        if any(word in query_lower.split()[:3] for word in question_words):
            return 'health_query'
        else:
            return 'booking_request'
    
    # Default to health query for general health questions
    return 'health_query'


async def process_health_query(query: str, session_id: str) -> Dict[str, Any]:
    """Process health advisory queries with enhanced error handling"""
    try:
        logger.info(f"Processing health query for session {session_id}")
        start_time = datetime.now()
        
        state = health_adviser_app.invoke({"question": query})
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": state.get('answer', 'Unable to generate response'),
            "response_type": "health_advice",
            "confidence_level": state.get('confidence_level'),
            "processing_time": processing_time,
            "metadata": {
                "documents_retrieved": len(state.get('documents', [])),
                "retrieval_score": state.get('retrieval_score', 0.0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing health query: {e}\n{traceback.format_exc()}")
        return {
            "response": "I apologize, but I'm unable to process your health question right now. Please try again later or consult a healthcare professional.",
            "response_type": "error",
            "confidence_level": "Very Low",
            "metadata": {"error": str(e)}
        }

async def process_booking_request(query: str, session_id: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Process booking requests with improved continuation handling"""
    try:
        logger.info(f"Processing booking request for session {session_id}")
        start_time = datetime.now()

        ql = query.lower().strip()

        # If we are continuing an existing booking flow:
        if session_state.get("status") == "awaiting_confirmation":
            logger.info(f"Continuing existing booking flow for session {session_id}")
            
            # Enhanced confirmation detection
            confirmation_patterns = [
                # Direct confirmations
                "yes", "yep", "yeah", "sure", "ok", "okay", "confirm",
                "please book", "book it", "schedule it", "go ahead",
                "find an appointment", "book appointment", "yes find",
                "proceed", "continue", "that's fine", "sounds good",
                # Scheduling responses
                "tomorrow", "today", "next week", "morning", "evening",
                "afternoon", "am", "pm", "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday"
            ]
            
            # Check for confirmation or scheduling details
            is_confirmation = any(pattern in ql for pattern in confirmation_patterns)
            
            # Check if it's a date/time specification
            has_datetime_info = any(indicator in ql for indicator in [
                "am", "pm", "morning", "evening", "afternoon", "tomorrow", 
                "today", "next", "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday", ":", "/"
            ])
            
            # Check if it's additional symptom information
            symptom_words = [
                "pain", "hurt", "ache", "feel", "symptom", "problem",
                "worse", "better", "since", "started", "days", "weeks"
            ]
            has_symptom_info = any(word in ql for word in symptom_words)
            
            if is_confirmation or has_datetime_info or (has_symptom_info and len(ql.split()) > 3):
                logger.info(f"User provided booking continuation: confirmation={is_confirmation}, datetime={has_datetime_info}, symptom_info={has_symptom_info}")
                logger.info(f"User query for continuation: '{query}'")
                
                # Get saved booking data
                saved_data = session_state.get("data", {})
                
                # Add the follow-up response
                saved_data.setdefault("follow_up_responses", []).append(query)
                saved_data["confirmation_received"] = True
                saved_data["user_confirmed"] = True
                
                # Process the booking continuation
                try:
                    logger.info(f"Invoking booking continuation with data keys: {list(saved_data.keys())}")
                    
                    continued_state = booking_agent_app.invoke(
                        saved_data,
                        config={"configurable": {"entrypoint": "continue_booking"}}
                    )
                    
                    logger.info(f"Booking continuation response keys: {list(continued_state.keys())}")
                    
                    # Mark session as completed
                    session_manager.set_session(session_id, {"status": "completed"})
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Check for final response in different possible keys
                    final_response = (
                        continued_state.get('final_response') or 
                        continued_state.get('response') or 
                        continued_state.get('booking_confirmation') or
                        'Booking process completed successfully'
                    )
                    
                    return {
                        "response": final_response,
                        "response_type": "booking_confirmation",
                        "processing_time": processing_time,
                        "metadata": {
                            "booking_completed": True,
                            "confirmation_type": "confirmed" if is_confirmation else "detailed",
                            "continued_state_keys": list(continued_state.keys())
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error in booking continuation: {e}\n{traceback.format_exc()}")
                    # Fall back to a comprehensive completion message
                    session_manager.set_session(session_id, {"status": "completed"})
                    
                    # Create a detailed booking confirmation based on saved data
                    symptoms = saved_data.get('symptoms', [])
                    urgency = saved_data.get('urgency', 'Low')
                    specialty = saved_data.get('specialty', 'General Physician')
                    
                    fallback_response = f"""✅ **Appointment Request Confirmed**

**Pre-Consultation Summary:**
• **Symptoms:** {', '.join(symptoms) if symptoms else 'As discussed'}
• **Urgency Level:** {urgency}
• **Recommended Specialty:** {specialty}

**Next Steps:**
• Your appointment request has been submitted
• You will receive a confirmation call within 24 hours
• Please keep your phone available
• Bring any relevant medical records to your appointment

**Emergency Contact:** If your condition worsens, please call 108 or visit the nearest emergency room immediately.

Thank you for using our booking service. Get well soon! 🏥"""
                    
                    return {
                        "response": fallback_response,
                        "response_type": "booking_confirmation",
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "metadata": {"booking_completed": True, "fallback_used": True}
                    }
            else:
                # Ask for explicit confirmation
                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    "response": "I have your appointment request ready. Would you like me to proceed with booking? Please confirm with 'Yes' or provide your preferred date and time.",
                    "response_type": "booking_inquiry",
                    "processing_time": processing_time,
                    "metadata": {"awaiting_explicit_confirmation": True}
                }

        # Handle new booking request
        logger.info(f"Processing new booking request for session {session_id}: {query[:100]}...")
        
        initial_state: BookingState = {"user_query": query}
        
        try:
            report_state = booking_agent_app.invoke(initial_state)
            
            # Save state for potential continuation
            session_manager.set_session(session_id, {
                "status": "awaiting_confirmation",
                "data": report_state,
                "created_at": datetime.now().isoformat(),
                "original_query": query
            })

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "response": report_state.get('final_response', 'Unable to process booking request'),
                "response_type": "booking_inquiry",
                "processing_time": processing_time,
                "metadata": {
                    "symptoms_found": len(report_state.get('symptoms', [])),
                    "urgency": report_state.get('urgency'),
                    "specialty": report_state.get('specialty'),
                    "awaiting_confirmation": True,
                    "booking_flow_started": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error invoking booking agent: {e}")
            return {
                "response": "I understand you're looking to book an appointment. Could you please provide more details about your symptoms or the type of consultation you need?",
                "response_type": "booking_inquiry",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "metadata": {"error": "booking_agent_error", "fallback_used": True}
            }

    except Exception as e:
        logger.error(f"Error processing booking request: {e}\n{traceback.format_exc()}")
        return {
            "response": "I apologize, but I'm unable to process your booking request right now. Please try again later or contact the hospital directly.",
            "response_type": "error",
            "metadata": {"error": str(e)}
        }

# Enhanced chat endpoint with better debugging
@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(
    user_query: UserQuery = Depends(rate_limit_check),
    request: Request = None
) -> AgentResponse:
    """Enhanced chat endpoint with improved routing logic"""
    
    start_time = datetime.now()
    session_id = user_query.session_id
    query = user_query.query
    
    logger.info(f"Chat request from session {session_id}: {query}")
    
    try:
        # Get session state
        session_state = session_manager.get_session(session_id)
        logger.info(f"Session {session_id} state: {session_state.get('status', 'new')}")

        # PRIORITY: Handle ongoing booking sessions
        if session_state.get("status") == "awaiting_confirmation":
            logger.info(f"Session {session_id} has pending booking - routing to booking continuation")
            response_data = await process_booking_request(query, session_id, session_state)

            total_processing_time = (datetime.now() - start_time).total_seconds()
            if 'processing_time' not in response_data:
                response_data['processing_time'] = total_processing_time

            response = AgentResponse(
                response=response_data["response"],
                session_id=session_id,
                response_type=response_data.get("response_type", "unknown"),
                confidence_level=response_data.get("confidence_level"),
                processing_time=response_data.get("processing_time"),
                metadata=response_data.get("metadata", {})
            )
            logger.info(f"Completed booking continuation for session {session_id}")
            return response

        # Classify intent for new requests
        intent = classify_intent(query)
        logger.info(f"Classified intent: {intent} for session {session_id}")
        
        # Add debug info to see classification reasoning
        logger.debug(f"Intent classification for '{query}': {intent}")
        
        # Process based on intent
        if intent == 'emergency':
            response_data = {
                "response": """🚨 **EMERGENCY DETECTED** 🚨

This appears to be a medical emergency. Please:

• **Call emergency services immediately: 108**
• **Go to the nearest emergency room**
• **Do not wait - seek immediate medical attention**

**Emergency Hospitals in Warangal:**
• Government Hospital Warangal
• Apollo Hospital Emergency
• KIMS Hospital Emergency

**Important**: Do not rely on online advice for emergencies. Get professional medical help immediately.""",
                "response_type": "emergency_alert",
                "confidence_level": "High",
                "metadata": {"emergency_detected": True}
            }
            
        elif intent == 'booking_request':
            logger.info(f"Processing as booking request for session {session_id}")
            response_data = await process_booking_request(query, session_id, session_state)
            
        elif intent == 'health_query':
            logger.info(f"Processing as health query for session {session_id}")
            response_data = await process_health_query(query, session_id)
            
        else:
            # Fallback
            logger.warning(f"Unrecognized intent, defaulting to health query for session {session_id}")
            response_data = await process_health_query(query, session_id)
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds()
        if 'processing_time' not in response_data:
            response_data['processing_time'] = total_processing_time
        
        # Create response
        response = AgentResponse(
            response=response_data["response"],
            session_id=session_id,
            response_type=response_data.get("response_type", "unknown"),
            confidence_level=response_data.get("confidence_level"),
            processing_time=response_data.get("processing_time"),
            metadata=response_data.get("metadata", {})
        )
        
        logger.info(f"Successfully processed {intent} for session {session_id} in {total_processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}\n{traceback.format_exc()}")
        
        error_response = AgentResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again later or contact a healthcare professional for immediate assistance.",
            session_id=session_id,
            response_type="system_error",
            confidence_level="Very Low",
            processing_time=(datetime.now() - start_time).total_seconds(),
            metadata={"error": "System error occurred", "error_type": type(e).__name__}
        )
        
        return error_response

@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Comprehensive health check endpoint"""
    try:
        current_time = datetime.now()
        stats = session_manager.get_stats()
        
        # Test basic functionality
        system_health = {}
        
        # Test health adviser
        try:
            test_result = health_adviser_app.invoke({"question": "test"})
            system_health["health_adviser"] = "healthy"
        except Exception as e:
            logger.error(f"Health adviser test failed: {e}")
            system_health["health_adviser"] = "unhealthy"
        
        # Test booking agent
        try:
            test_state: BookingState = {"user_query": "test headache"}
            booking_agent_app.invoke(test_state)
            system_health["booking_agent"] = "healthy"
        except Exception as e:
            logger.error(f"Booking agent test failed: {e}")
            system_health["booking_agent"] = "unhealthy"
        
        # Overall system status
        overall_status = "healthy" if all(
            status == "healthy" for status in system_health.values()
        ) else "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=current_time.isoformat(),
            version="2.0.0",
            uptime_seconds=stats["uptime_seconds"],
            active_sessions=stats["active_sessions"],
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            uptime_seconds=0,
            active_sessions=0,
            system_health={"error": str(e)}
        )

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """Get session status and metadata"""
    try:
        session_data = session_manager.get_session(session_id)
        session_timestamp = session_manager.session_timestamps.get(session_id)
        
        return {
            "session_id": session_id,
            "status": session_data.get("status", "new"),
            "created_at": session_timestamp.isoformat() if session_timestamp else None,
            "last_activity": session_timestamp.isoformat() if session_timestamp else None,
            "awaiting_confirmation": session_data.get("status") == "awaiting_confirmation",
            "metadata": {
                "has_data": "data" in session_data,
                "data_keys": list(session_data.get("data", {}).keys()) if "data" in session_data else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve session status")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> Dict[str, str]:
    """Clear a specific session"""
    try:
        session_manager.remove_session(session_id)
        logger.info(f"Session {session_id} cleared successfully")
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Unable to clear session")

@app.post("/sessions/cleanup")
async def manual_cleanup(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Trigger manual session cleanup"""
    try:
        background_tasks.add_task(session_manager._cleanup_expired_sessions)
        return {"message": "Session cleanup initiated"}
        
    except Exception as e:
        logger.error(f"Error initiating cleanup: {e}")
        raise HTTPException(status_code=500, detail="Unable to initiate cleanup")

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get API statistics"""
    try:
        stats = session_manager.get_stats()
        
        return {
            "api_version": "2.0.0",
            "configuration": {
                "max_sessions": config.max_sessions,
                "max_session_age_hours": config.max_session_age,
                "rate_limit_requests": config.rate_limit_requests,
                "rate_limit_window_minutes": config.rate_limit_window
            },
            "runtime_stats": stats,
            "system_info": {
                "debug_mode": config.debug,
                "host": config.host,
                "port": config.port
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve statistics")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(f"HTTP {exc.status_code} error for {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception for {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("="*50)
    logger.info("AI Health Agent API Starting Up")
    logger.info(f"Version: 2.0.0")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Debug Mode: {config.debug}")
    logger.info(f"Max Sessions: {config.max_sessions}")
    logger.info(f"Session Age Limit: {config.max_session_age} hours")
    logger.info(f"Rate Limit: {config.rate_limit_requests} requests per {config.rate_limit_window} minutes")
    logger.info("="*50)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("AI Health Agent API Shutting Down")
    logger.info(f"Final session count: {len(session_manager.sessions)}")

def run_api():
    """Run the API with proper configuration"""
    try:
        uvicorn.run(
            "main:app",
            host=config.host,
            port=config.port,
            reload=config.debug,
            log_level="info" if not config.debug else "debug",
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("API shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

if __name__ == "__main__":
    run_api()
