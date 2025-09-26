import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI

# Import ALL your tools
from tools.sql_tools import get_available_slots, book_appointment, find_diseases_by_symptoms
from tools.mongo_tools import save_consultation_details

load_dotenv()

class BookingState(TypedDict):
    user_query: str
    symptoms: List[str]
    urgency: str
    specialty: str
    potential_diseases: List[str]
    available_slots: List[dict]
    final_response: str

llm = ChatOpenAI(model="gpt-4o", temperature=0)
classification_parser = JsonOutputParser()
symptom_parser = CommaSeparatedListOutputParser()

# --- Node Definitions (Functions remain the same as before) ---

# In agents/booking_agent.py

def extract_symptoms_node(state: BookingState) -> BookingState:
    """Extracts symptoms from the user query into a list."""
    print("---NODE: EXTRACTING SYMPTOMS---")
    prompt = ChatPromptTemplate.from_template(
        "Extract the key medical symptoms from the following text. Respond with only a comma-separated list of symptoms. If only one symptom is found, respond with just that symptom without a comma.\n\nText: {query}"
    )
    # Using a simple StrOutputParser and splitting manually is more reliable
    from langchain_core.output_parsers import StrOutputParser
    
    chain = prompt | llm | StrOutputParser()
    
    # Manually split the string response to guarantee a list
    symptoms_str = chain.invoke({"query": state['user_query']})
    symptoms_list = [s.strip().capitalize() for s in symptoms_str.split(',')]
    
    state['symptoms'] = symptoms_list
    print(f"Extracted Symptoms: {state['symptoms']}")
    return state
    
def classify_issue_node(state: BookingState) -> BookingState:
    print("---NODE: CLASSIFYING ISSUE---")
    prompt = ChatPromptTemplate.from_template("""Analyze symptoms and classify them into 'urgency' and 'specialty'.
        Urgency: "Low", "Medium", "High", "Emergency".
        Specialty: "Cardiology", "Dermatology", "General Physician", "Orthopedics".
        Respond with a JSON object.
        Symptoms: {symptoms}""")
    chain = prompt | llm | classification_parser
    classification = chain.invoke({"symptoms": ", ".join(state['symptoms'])})
    state.update(classification)
    return state

def find_potential_diseases_node(state: BookingState) -> BookingState:
    print("---NODE: FINDING POTENTIAL DISEASES---")
    state['potential_diseases'] = find_diseases_by_symptoms.invoke({"symptoms_list": state['symptoms']})
    return state

def present_findings_node(state: BookingState) -> BookingState:
    print("---NODE: PRESENTING FINDINGS REPORT---")
    report = [f"### Pre-Consultation Summary\n- **Symptoms**: {', '.join(state['symptoms'])}\n- **Urgency**: {state['urgency']}\n- **Specialty**: {state['specialty']}"]
    if state['potential_diseases']:
        report.append(f"- **Potential Conditions**: {', '.join(state['potential_diseases'])}")
    report.append("\n**Disclaimer**: This is not a medical diagnosis.\n\nWould you like me to find an appointment? Reply with **'yes, find an appointment'** to continue.")
    state['final_response'] = "\n".join(report)
    return state

def find_slots_node(state: BookingState) -> BookingState:
    print("---NODE: FINDING SLOTS---")
    state['available_slots'] = get_available_slots.invoke({"hospital": "Apollo Hospital", "specialty": state['specialty']})
    return state

def book_appointment_node(state: BookingState) -> BookingState:
    print("---NODE: BOOKING APPOINTMENT---")
    slot_to_book = state['available_slots'][0]
    book_appointment.invoke({"slot_id": slot_to_book['slot_id']})
    confirmation = [f"### ✅ Appointment Confirmed!\n- **Booking ID**: {slot_to_book['slot_id']}\n- **Hospital**: {slot_to_book['hospital']}\n- **Specialty**: {slot_to_book['specialty']}\n- **Time**: {slot_to_book['time']}"]
    state['final_response'] = "\n".join(confirmation)
    return state

def emergency_protocol_node(state: BookingState) -> BookingState:
    state['final_response'] = "Your symptoms may indicate an emergency. Please seek immediate medical attention."
    return state

def no_slots_found_node(state: BookingState) -> BookingState:
    state['final_response'] = "I'm sorry, no appointment slots are available. Please try again later."
    return state

# --- Conditional Edges ---
def decide_after_classification(state: BookingState):
    return "emergency" if state['urgency'].lower() == "emergency" else "find_potential_diseases"

def decide_after_finding_slots(state: BookingState):
    return "no_slots" if not state['available_slots'] else "book_slot"

# --- Assemble the Graph ---
workflow = StateGraph(BookingState)

# Add all nodes
nodes = {
    "extract_symptoms": extract_symptoms_node, "classify_issue": classify_issue_node,
    "find_potential_diseases": find_potential_diseases_node, "present_findings": present_findings_node,
    "find_slots": find_slots_node, "book_appointment": book_appointment_node,
    "emergency_protocol": emergency_protocol_node, "no_slots_found": no_slots_found_node
}
for name, node in nodes.items():
    workflow.add_node(name, node)

# Define edges for the initial diagnosis part of the workflow
workflow.set_entry_point("extract_symptoms")
workflow.add_edge("extract_symptoms", "classify_issue")
workflow.add_conditional_edges(
    "classify_issue",
    decide_after_classification,
    {"emergency": "emergency_protocol", "find_potential_diseases": "find_potential_diseases"}
)
workflow.add_edge("find_potential_diseases", "present_findings")
workflow.add_edge("present_findings", END)
workflow.add_edge("emergency_protocol", END)

# Define edges for the booking confirmation part of the workflow
# This part is entered with pre-populated state
workflow.add_node("continue_booking", find_slots_node) # Create a specific entry point for continuing
workflow.add_conditional_edges(
    "continue_booking",
    decide_after_finding_slots,
    {"no_slots": "no_slots_found", "book_slot": "book_appointment"}
)
workflow.add_edge("book_appointment", END)
workflow.add_edge("no_slots_found", END)

booking_agent_app = workflow.compile()