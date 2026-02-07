import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load Kaggle dataset
df = pd.read_csv("data/appointments.csv")

# Make sure patient_id exists (add one if missing)
if "PatientId" not in df.columns:
    df["PatientId"] = range(1, len(df) + 1)

# Select required columns
df = df[["ScheduledDay", "AppointmentDay", "SMS_received", "No-show", "PatientId"]]

# Convert target label
df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0})

# Convert to datetime
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

# Feature engineering
df["waiting_days"] = (
    df["AppointmentDay"] - df["ScheduledDay"]
).dt.days

# Features & target
X = df[["waiting_days", "SMS_received"]]
y = df["No-show"]

# Train ML model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model/reminder_model.pkl")

print("✅ Model trained and saved successfully")
