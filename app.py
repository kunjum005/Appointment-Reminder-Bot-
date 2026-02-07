import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import nltk
import pytz 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# NLP Setup
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

# -----------------------------
# Load Data
# -----------------------------
# Load appointments CSV (Kaggle dataset)
appointments = pd.read_csv("data/appointments.csv", encoding="utf-8-sig")

# Load patient preferences CSV
preferences = pd.read_csv("data/patient_preferences.csv", encoding="utf-8-sig")

# -----------------------------
# Ensure PatientId exists in appointments
# -----------------------------
# If appointments CSV does not have PatientId, create it
if "PatientId" not in appointments.columns:
    appointments["PatientId"] = range(1, len(appointments) + 1)

# -----------------------------
# Merge appointments + preferences
# -----------------------------
appointments = appointments.merge(preferences, on="PatientId", how="left")

# Fill missing preferences with defaults
appointments["preferred_time"].fillna("morning", inplace=True)
appointments["reminder_type"].fillna("SMS", inplace=True)

# Convert date columns
appointments["ScheduledDay"] = pd.to_datetime(appointments["ScheduledDay"])
appointments["AppointmentDay"] = pd.to_datetime(appointments["AppointmentDay"], utc=True)

# Feature: waiting days
appointments["waiting_days"] = (appointments["AppointmentDay"] - appointments["ScheduledDay"]).dt.days

if "SMS_received" not in appointments.columns:
    appointments["SMS_received"] = 0  # Default 0 if missing
    
# -----------------------------
# Load ML model for no-show prediction
# -----------------------------
reminder_model = joblib.load("model/reminder_model.pkl")

# -----------------------------
# NLP Intent Dataset
# -----------------------------
intent_data = {
    "text": [
        "show appointments",
        "list appointments",
        "send reminders",
        "remind patients",
        "who will miss appointment",
        "predict no show"
    ],
    "intent": [
        "view",
        "view",
        "remind",
        "remind",
        "predict",
        "predict"
    ]
}

intent_df = pd.DataFrame(intent_data)
intent_df["processed"] = intent_df["text"].apply(preprocess)

vectorizer = TfidfVectorizer()
X_intent = vectorizer.fit_transform(intent_df["processed"])
y_intent = intent_df["intent"]
intent_model = MultinomialNB()
intent_model.fit(X_intent, y_intent)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Smart Appointment Reminder Bot", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("ℹ️ Project Info")
st.sidebar.markdown("""
**Smart Appointment Reminder Bot**  
This app integrates appointment data, patient preferences, and ML models to:  

- View upcoming appointments  
- Predict high-risk (likely no-show) patients  
- Send smart reminders automatically  

**Features:**  
- ML-based no-show prediction  
- Personalized reminder types & times  
- Command-based interface with NLP intent detection  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed using Streamlit, Python, Pandas & Scikit-learn")

# ---------------- Main UI ----------------

st.title("⏰ Smart Appointment Reminder Bot")
st.write("Compressed appointment data + patient preferences + smart reminders using ML")

# Add session state to store user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def clear_input():
    st.session_state.user_input = ""

user_input = st.text_input("💬 Enter command", key="user_input")
st.button("Clear", on_click=clear_input)

# Function to display messages like WhatsApp chat
def chat_message(message, sender="bot"):
    if sender == "bot":
        st.markdown(
            f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px 0; width:fit-content;'>{message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#FFF; border:1px solid #ECE5DD; padding:10px; border-radius:10px; margin:5px 0; width:fit-content;'>{message}</div>",
            unsafe_allow_html=True
        )
        

if user_input:
    processed = preprocess(user_input)
    vector = vectorizer.transform([processed])
    intent = intent_model.predict(vector)[0]
    chat_message(f"Detected Intent: {intent}", sender="bot")

    # ---------------- VIEW / Compressed Data ----------------
    # ---------------- VIEW / Compressed Data ----------------
    if intent == "view":
        # Using the actual date range logic
        today = appointments['AppointmentDay'].min()
        next_3_days = today + timedelta(days=3)
        
        # Filter and select columns
        compressed = appointments[
            (appointments["AppointmentDay"] >= today) &
            (appointments["AppointmentDay"] <= next_3_days)
        ].copy()
        
        compressed['Date']= compressed['AppointmentDay'].dt.strftime('%Y-%m-%d')
        
        display_df = compressed[[
            "Date", "PatientId", "preferred_time", "reminder_type"
        ]].rename(columns={
            "preferred_time": "Time Slot",
            "reminder_type": "Method"
        })
     
        
        st.subheader("📋 Appointments for the next 3 days")
        
        if display_df.empty:
            st.warning("No appointments found") # Fixed: st.waiting is not a standard streamlit command
        else:
            # Removed .head(10) to show ALL entries for those 3 days
            # Removed the duplicate st.dataframe line
            st.table(display_df.reset_index(drop=True).head(3))

    # ---------------- PREDICT High-risk Patients ----------------
    elif intent == "predict":
        X_ml = appointments[["waiting_days", "SMS_received"]]
        appointments["risk"] = reminder_model.predict(X_ml)
        risky = appointments[appointments["risk"] == 1]

        st.subheader("⚠️ High-risk (likely no-show) appointments (Top 10)")
        st.dataframe(risky[["AppointmentDay", "PatientId", "preferred_time", "reminder_type"]]
                     .sort_values(by="AppointmentDay")
                    .head(10)
                    .reset_index(drop=True)
        )

    # ---------------- SMART REMINDERS ----------------
    elif intent == "remind":
        X_ml = appointments[["waiting_days", "SMS_received"]]
        appointments["risk"] = reminder_model.predict(X_ml)
        high_risk = appointments[appointments["risk"] == 1]

        if high_risk.empty:
            st.info("✅ No high-risk patients to remind")
        else:
            st.subheader("📨 Sending Smart Reminders")
            for _, row in high_risk.head(10).iterrows():
                chat_message(
                    f"Reminder sent to Patient {row['PatientId']} at {row['preferred_time']} via {row['reminder_type']}",
                    sender="bot"
                )
            st.info(f"✅ {min(len(high_risk), 10)} reminders sent (showing only 10)")