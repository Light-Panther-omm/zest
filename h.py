import streamlit as st
import google.generativeai as genai
import json
import os
import logging
from datetime import datetime
import re
from transformers import pipeline
import base64

# Page Config
st.set_page_config(page_title="Gemini Healthcare Assistant", layout="centered")

# Load Base64 Font
def load_font_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

font_base64 = load_font_base64("Greenos.ttf")

# Inject External CSS
def inject_custom_css(file_path, font_base64=""):
    with open(file_path, "r") as file:
        css = file.read().replace("{font_base64}", font_base64)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_custom_css("style.css", font_base64)

# Load NER model
PRETRAINED = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"
ers = pipeline(task="ner", model=PRETRAINED, tokenizer=PRETRAINED)

# Enable debugging
st.set_option("client.showErrorDetails", True)
logging.basicConfig(level=logging.DEBUG)

# Gemini setup
GEMINI_API_KEY = "AIzaSyBxpTHcJP3dmR9Pqppp4zmc2Tfut6nic6A"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Users
users = {
    "a": "2",
    "doctor": "health123",
    "": "",
}

CHAT_RECORD_FILE = "chat_records.json"

def load_records():
    if os.path.exists(CHAT_RECORD_FILE):
        with open(CHAT_RECORD_FILE, "r") as f:
            return json.load(f)
    return []

def save_record(chat_history):
    records = load_records()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records.append({"timestamp": timestamp, "history": chat_history})
    with open(CHAT_RECORD_FILE, "w") as f:
        json.dump(records, f, indent=2)

def export_chat(chat_history):
    return json.dumps(chat_history, indent=2).encode("utf-8")

# Health Keywords
health_keywords = [
    "fever", "cold", "headache", "pain", "diabetes", "pressure", "bp", "covid",
    "infection", "symptom", "cancer", "flu", "aids", "allergy", "disease", "vomit", "asthma",
    "medicine", "tablet", "ill", "sick", "nausea", "health", "injury", "cough", "treatment",
    "doctor", "hospital", "clinic", "vaccine", "antibiotic", "therapy", "mental health", "stress",
    "anxiety", "depression", "diet", "nutrition", "fitness", "exercise", "weight loss", "cholesterol",
    "thyroid", "migraine", "burn", "fracture", "wound", "emergency", "blood sugar", "sugar", "heart", "lungs"
]

def is_health_related(text):
    return any(re.search(rf"\b{re.escape(word)}\b", text.lower()) for word in health_keywords)

def extract_diseases(text):
    entities = ers(text)
    return set(ent['word'] for ent in entities if 'disease' in ent.get('entity', '').lower())

def highlight_diseases(text):
    diseases = extract_diseases(text)
    for disease in diseases:
        text = re.sub(fr"\b({re.escape(disease)})\b", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(e)
        return "⚠️ An unexpected error occurred. Please try again."

# Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_tag" not in st.session_state:
    st.session_state.selected_tag = "General"

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.markdown('<div class="login-hide"></div>', unsafe_allow_html=True)
    st.title("🔐 Login")
    st.markdown("Please enter your credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

# ---------------- MAIN CHAT APP ----------------
else:
    st.title("🩺Healthcare Assistant")

    with st.sidebar:
        st.header("LOGGED IN")
        st.write(f"**{st.session_state.username}**")
        name = st.text_input("Name (optional for health-related questions)")
        age = st.text_input("Age (optional for health-related questions)")
        gender = st.selectbox("Gender (optional)", ["", "Male", "Female", "Other"])
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()

    tab1, tab2 = st.tabs(["💬 Chat", "📜 Records"])

    # ---------- TAB 1 ----------
    with tab1:
        st.markdown("#### 🏷️ Choose a topic (optional):")
        tags = ["General", "Mental Health", "Diet", "Fitness", "Stress"]
        st.session_state.selected_tag = st.radio("", tags, horizontal=True, index=0)

        for sender, msg in st.session_state.chat_history:
            cls = "user-bubble" if sender == "You" else "bot-bubble"
            display_msg = highlight_diseases(msg) if sender != "You" else msg
            st.markdown(f'<div class="chat-bubble {cls}"><strong>{sender}:</strong><br>{display_msg}</div>',
                        unsafe_allow_html=True)

        chat = st.chat_input("Type your message...")

        if chat:
            st.session_state.chat_history.append(("You", chat))

            if is_health_related(chat):
                if not (name and age and gender):
                    response = "⚠️ Please complete your user info for health-related questions."
                elif not age.isdigit() or not (0 <= int(age) <= 120):
                    response = "⚠️ Please enter a valid age between 0 and 120."
                else:
                    prompt = f"""You are a helpful AI healthcare assistant. Provide simple, safe, general health-related answers without diagnoses or prescriptions.

User Info:
Name: {name}
Age: {age}
Gender: {gender}

Topic: {st.session_state.selected_tag}

User's Question: {chat}
"""
                    response = ask_gemini(prompt)
            else:
                prompt = f"""You are a friendly, polite assistant.
Respond naturally and supportively.

Topic: {st.session_state.selected_tag}

User's Message: {chat}
"""
                response = ask_gemini(prompt)

            st.session_state.chat_history.append(("Gemini", response))
            save_record(st.session_state.chat_history)
            st.rerun()

        if st.session_state.chat_history:
            st.download_button("⬇️ Export Chat", export_chat(st.session_state.chat_history),
                               "chat_history.json", "application/json")

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("Past Conversations")
        records = load_records()
        if records:
            for idx, record in enumerate(records[::-1], 1):
                with st.expander(f"Chat Record #{idx} – {record['timestamp']}"):
                    for item in record["history"]:
                        if isinstance(item, list) and len(item) == 2:
                            sender, message = item
                            st.markdown(f"**{sender}:** {message}")
                        else:
                            st.warning("⚠️ Invalid chat format.")
        else:
            st.info("No past records found.")
