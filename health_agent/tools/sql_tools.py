import os
import sqlite3
from langchain.tools import tool

# --- Define an absolute path to the database ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "diseases.db")

# In-memory mock data for available hospital slots
MOCK_HOSPITAL_DATA = [
    {"slot_id": 1, "hospital": "Apollo Hospital", "specialty": "Dermatology", "time": "2025-09-26 10:00", "booked": False},
    {"slot_id": 2, "hospital": "Apollo Hospital", "specialty": "Dermatology", "time": "2025-09-26 11:00", "booked": False},
    {"slot_id": 3, "hospital": "Apollo Hospital", "specialty": "General Physician", "time": "2025-09-26 12:00", "booked": False},
    {"slot_id": 4, "hospital": "KIMS Hospital", "specialty": "Orthopedics", "time": "2025-09-26 14:00", "booked": False},
    {"slot_id": 5, "hospital": "Apollo Hospital", "specialty": "Cardiology", "time": "2025-09-26 15:00", "booked": True},
]

@tool
def get_available_slots(hospital: str, specialty: str) -> list:
    """Finds available, unbooked slots for a given hospital and specialty from the mock database."""
    print(f"🔍 Searching MOCK DB for slots at {hospital} for {specialty}...")
    
    available_slots = []
    for slot in MOCK_HOSPITAL_DATA:
        if (slot["hospital"].lower() == hospital.lower() and
            slot["specialty"].lower() == specialty.lower() and
            not slot["booked"]):
            available_slots.append(slot)
            
    return available_slots

@tool
def book_appointment(slot_id: int) -> str:
    """Books an appointment by its slot_id in the mock database."""
    print(f"✍️ Attempting to book slot_id {slot_id} in MOCK DB...")
    
    for slot in MOCK_HOSPITAL_DATA:
        if slot["slot_id"] == slot_id:
            if not slot["booked"]:
                slot["booked"] = True  # "Book" the slot
                return f"Success! Appointment for slot {slot_id} has been booked."
            else:
                return f"Error: Slot {slot_id} is already booked."
                
    return f"Error: Slot with ID {slot_id} not found."

@tool
def find_diseases_by_symptoms(symptoms_list: list) -> list:
    """
    Queries the database to find diseases that match all symptoms in a given list.
    Returns a list of disease names.
    """
    if not symptoms_list:
        return []

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        placeholders = ', '.join('?' for _ in symptoms_list)
        query = f"""
            SELECT d.name
            FROM diseases d
            JOIN disease_symptoms ds ON d.disease_id = ds.disease_id
            JOIN symptoms s ON ds.symptom_id = s.symptom_id
            WHERE s.name IN ({placeholders})
            GROUP BY d.name
            HAVING COUNT(DISTINCT s.name) = ?
        """
        params = symptoms_list + [len(symptoms_list)]
        
        cur.execute(query, params)
        results = [row[0] for row in cur.fetchall()]
        con.close()
        
        return results
    except sqlite3.OperationalError as e:
        print(f"Database Error: {e}")
        print(f"Please ensure you have run 'run_sql_script.py' and the database '{DB_PATH}' is correctly populated.")
        return []