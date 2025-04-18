import streamlit as st
import google.generativeai as genai
import json
import os
import logging
from datetime import datetime
import re
from transformers import pipeline

# --- CUSTOM FONT FOR TITLE ---
import base64

st.set_page_config(page_title="Gemini Healthcare Assistant", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }

    [data-testid="stSidebar"] {
        background-color: #fbeec1;
    }

    /* White background and styling for input boxes */
    div[data-testid="textInput"] input {
        background-color: #ffffff !important;
        color: black !important;
        border-radius: 5px;
    }

    /* White background for select box */
    div[data-testid="stSelectbox"] > div {
        background-color: #ffffff !important;
        color: black !important;
        border-radius: 5px;
    }

    /* Optional: white textarea too */
    textarea {
        background-color: #ffffff !important;
        color: black !important;
        border-radius: 5px;

    }
    [data-testid="stForm"] > div:first-child > div:first-child > div:first-child {{
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def load_font_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


font_base64 = load_font_base64("Greenos.ttf")  # <-- using .otf now

st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'CustomTitleFont';
        src: url(data:font/otf;base64,{font_base64}) format('truetype');
    }}
    h1 {{
        font-family: 'CustomTitleFont', sans-serif !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load NER model
PRETRAINED = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"
ers = pipeline(task="ner", model=PRETRAINED, tokenizer=PRETRAINED)

# Enable debugging
st.set_option("client.showErrorDetails", True)
logging.basicConfig(level=logging.DEBUG)

# Load Gemini API key from environment variable
GEMINI_API_KEY = "AIzaSyBxpTHcJP3dmR9Pqppp4zmc2Tfut6nic6A"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# User credentials
users = {
    "a": "2",
    "doctor": "health123",
    "": "",
}

# File to store chat records
CHAT_RECORD_FILE = "chat_records.json"


# Load/save records
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


# Keyword check
health_keywords = [
    "fever", "cold", "headache", "pain", "diabetes", "pressure", "bp", "covid",
    "infection", "symptom", "cancer", "flu", "aids", "allergy", "disease", "vomit", "asthma",
    "medicine", "tablet", "ill", "sick", "nausea", "health", "injury", "cough", "treatment",
    "doctor", "hospital", "clinic", "vaccine", "antibiotic", "therapy", "mental health", "stress",
    "anxiety", "depression", "diet", "nutrition", "fitness", "exercise", "weight loss", "cholesterol",
    "thyroid", "migraine", "burn", "fracture", "wound", "emergency", "blood sugar", "sugar", "heart", "lungs"
]


def is_health_related(text):
    return any(re.search(rf"\\b{re.escape(word)}\\b", text.lower()) for word in health_keywords)


def extract_diseases(text):
    entities = ers(text)
    diseases = [ent['word'] for ent in entities if 'disease' in ent.get('entity', '').lower()]
    return set(diseases)


def highlight_diseases(text):
    diseases = extract_diseases(text)
    for disease in diseases:
        text = re.sub(fr"\\b({re.escape(disease)})\\b", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text


def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(e)
        return "⚠️ An unexpected error occurred. Please try again."


# Page config and styles


st.markdown("""
    <style>
    .chat-bubble {
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #2f2f2f;
        color: white;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #e0f7fa;
        color: black;
        margin-right: auto;
    }
    mark {
        background-color: #ffeb3b;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_tag" not in st.session_state:
    st.session_state.selected_tag = "General"

# LOGIN PAGE
if not st.session_state.logged_in:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔐 Login")
    st.markdown("Please enter your credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted or (username and password and st.session_state.get("autologin", False)):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

# MAIN CHAT APP
else:
    st.title("🩺Healthcare Assistant")

    # Sidebar with logged-in username and user info inputs
    with st.sidebar:
        st.header("LOGGED IN")
        st.write(f"**{st.session_state.username}**")
        name = st.text_input("Name (optional for health-related questions)")
        age = st.text_input("Age (optional for health-related questions)")
        gender = st.selectbox("Gender (optional for health-related questions)", ["", "Male", "Female", "Other"])
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()

    tab1, tab2 = st.tabs(["💬 Chat", "📜 Records"])

    with tab1:
        st.markdown("#### 🏷️ Choose a topic (optional):")
        predefined_tags = ["General", "Mental Health", "Diet", "Fitness", "Stress"]
        st.session_state.selected_tag = st.radio("", predefined_tags, horizontal=True, index=0)

        for sender, msg in st.session_state.chat_history:
            bubble_class = "user-bubble" if sender == "You" else "bot-bubble"
            formatted_msg = highlight_diseases(msg) if sender != "You" else msg
            st.markdown(f'<div class="chat-bubble {bubble_class}"><strong>{sender}:</strong><br>{formatted_msg}</div>',
                        unsafe_allow_html=True)

        with st._bottom:
            left_col, right_col = st.columns([4, 1])
            with left_col:
                chat = st.chat_input("Type your message...")
            with right_col:
                if st.button("🧹 Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

        if chat:
            st.session_state.chat_history.append(("You", chat))

            if is_health_related(chat):
                if not (name and age and gender):
                    response = "⚠️ Please complete your user info for health-related questions."
                elif not age.isdigit() or not (0 <= int(age) <= 120):
                    response = "⚠️ Please enter a valid age between 0 and 120."
                else:
                    prompt = f"""
You are a helpful AI healthcare assistant.
Provide simple, safe, general health-related answers without diagnoses or prescriptions.

User Info:
Name: {name}
Age: {age}
Gender: {gender}

Topic: {st.session_state.selected_tag}

User's Question: {chat}
"""
                    response = ask_gemini(prompt)
            else:
                prompt = f"""
You are a friendly, polite assistant.
Respond naturally and supportively.

Topic: {st.session_state.selected_tag}

User's Message:
{chat}
"""
                response = ask_gemini(prompt)

            st.session_state.chat_history.append(("Gemini", response))
            save_record(st.session_state.chat_history)
            st.rerun()

        if st.session_state.chat_history:
            st.download_button(
                "⬇️ Export Chat",
                export_chat(st.session_state.chat_history),
                file_name="chat_history.json",
                mime="application/json"
            )

    with tab2:
        st.subheader("Past Conversations")
        records = load_records()
        if records:
            for idx, record in enumerate(records[::-1], 1):
                if isinstance(record, dict) and "timestamp" in record and "history" in record:
                    with st.expander(f"Chat Record #{idx} – {record['timestamp']}"):
                        for chat_item in record["history"]:
                            if isinstance(chat_item, list) and len(chat_item) == 2:
                                sender, message = chat_item
                                st.markdown(f"**{sender}:** {message}")
                            else:
                                st.warning("⚠️ Invalid chat item format.")
                else:
                    st.warning("⚠️ Invalid record format.")
        else:
            st.info("No past records found.")
