## Appointment-Reminder-Bot
A Streamlit-based application that uses Natural Language Processing (NLP) and Machine Learning to manage medical appointments. The bot detects user intent to display schedules, predict patient "no-shows," and send personalized reminders based on patient preferences.

## 🚀 Features
- NLP Intent Detection: Uses a Naive Bayes classifier to understand commands like "show appointments" or "predict no-shows".
- Automated Scheduling: Filters and displays upcoming appointments for a 3-day window.
- No-Show Prediction: Utilizes a Scikit-learn model to identify high-risk patients based on waiting days and SMS history.
- Smart Reminders: Simulates sending reminders via the patient's preferred method (SMS/Email) and preferred time of day (Morning/Afternoon).

## 🛠️ Tech Stack
- Frontend: Streamlit
- Data Handling: Pandas, NumPy
- Machine Learning: Scikit-learn (MultinomialNB & RandomForest/LogisticRegression)
- NLP: NLTK (Tokenization, Stopword removal)

## Project Structure
├── data/
│   ├── appointments.csv         # Kaggle No-show dataset
│   └── patient_preferences.csv   # Custom patient settings
├── model/
│   └── reminder_model.pkl       # Trained No-show prediction model
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md

## ⚙️ Installation & Setup

## 1. Clone the repository:
git clone https://github.com/kunjum005/Appointment-Reminder-Bot.git
cd smart-appointment-bot
## 2. Install dependencies:
pip install -r requirements.txt
## 3. Run the application:
streamlit run app.py

## 🤖 How to Use
Type commands into the chat interface to interact with the bot:
- "Show appointments": Displays a cleaned list of patients for the next 3 days.
- "Predict no-shows": Identifies patients likely to miss their appointments using the ML model.
- "Send reminders": Generates personalized reminder logs based on patient preferences.

## 🔮 Future Improvements
While the current version provides a solid foundation for appointment management, the following features are planned for future releases:
1. Real-time Integration: Connecting the bot to the Google Calendar API or Twilio to send actual SMS and email notifications.
2. Enhanced NLP: Implementing a Transformer-based model (like BERT) to handle more complex user queries and conversational nuances.
3. Patient Feedback Loop: Allowing the bot to receive and process cancellation requests directly from the chat interface to update the database automatically.

## 🤝 Acknowledgements
This project was made possible through the use of the following open-source tools and datasets:
- Dataset: https://www.kaggle.com/datasets/joniarroba/noshowappointments 
- scikit-learn & Streamlit communities

## ⭐ If you like this project
Give it a star ⭐ on GitHub and feel free to fork or contribute
