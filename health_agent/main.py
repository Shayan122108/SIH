from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from agents.health_adviser import health_adviser_app
from agents.booking_agent import booking_agent_app, BookingState

load_dotenv()

app = FastAPI(title="AI Health Agent API")

class UserQuery(BaseModel):
    query: str
    session_id: str

class AgentResponse(BaseModel):
    response: str
    session_id: str

# In-memory session store to hold conversation state
sessions = {}

@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(user_query: UserQuery):
    query = user_query.query
    session_id = user_query.session_id

    # Retrieve the session state, or create a new one
    session_state = sessions.get(session_id, {"status": "new"})
    
    final_response = ""
    
    # Check if we are waiting for a booking confirmation
    if session_state.get("status") == "awaiting_confirmation" and "yes" in query.lower():
        print(f"Session [{session_id}] - Continuing booking")
        # Restore the state and continue the booking flow from the new entry point
        continued_state = booking_agent_app.invoke(
            session_state["data"],
            config={"run_name": "continue_booking"}
        )
        final_response = continued_state.get('final_response')
        sessions[session_id] = {"status": "closed"} # End the booking flow

    # Standard routing for new queries
    else:
        # Simple intent detection
        if "book" in query.lower() or "symptom" in query.lower() or "fever" in query.lower() or "headache" in query.lower() or "appointment" in query.lower():
            intent = 'booking_request'
        else:
            intent = 'health_query'
        
        print(f"Session [{session_id}] - Intent: {intent}")

        if intent == 'health_query':
            state = health_adviser_app.invoke({"question": query})
            final_response = state.get('answer')
            sessions[session_id] = {"status": "new"} # Reset status
        
        elif intent == 'booking_request':
            initial_state: BookingState = {"user_query": query}
            # Run the first part of the booking agent (diagnosis)
            report_state = booking_agent_app.invoke(initial_state)
            final_response = report_state.get('final_response')
            # Save the state and mark that we are waiting for confirmation
            sessions[session_id] = {
                "status": "awaiting_confirmation",
                "data": report_state
            }

    return AgentResponse(response=final_response, session_id=session_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)