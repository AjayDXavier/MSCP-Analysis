!pip install imbalanced-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


df = pd.read_csv('MLDATA.csv')
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9]', '_', regex=True)


df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 0, 'female': 1})


binary_cols = [
    'panicattackhistory_yes_no_',
    'hypertensionhistory_yes_no_',
    'medicationuse_yes_no_',
    'smoking_alcohol_etc_yes_no_'
]
for col in binary_cols:
    df[col] = df[col].astype(str).str.strip().str.upper().map({'YES': 1, 'NO': 0})


df['socialinteraction_low_med_high_'] = df['socialinteraction_low_med_high_'].astype(str).str.strip().str.upper().map({'LOW': 0, 'MED': 1, 'HIGH': 2})


if 'duration' in df.columns:
    df['duration'] = df['duration'].astype(str).str.extract(r'(\d+)').astype(float)


df['qol_score'] = (
    100
    - df['painseverity_1_10_'] * 3
    - df['anxietyscore'] * 2
    - df['depressionscore'] * 2
    - df['panicattackhistory_yes_no_'] * 10
    - df['hypertensionhistory_yes_no_'] * 5
    - df['medicationuse_yes_no_'] * 5
    - df['smoking_alcohol_etc_yes_no_'] * 5
    + df['sleephours'] * 2
    + df['physicalactivity_hoursperweek_'] * 2
    + df['socialinteraction_low_med_high_'] * 5
    - df['age'] * 0.1
)


def classify_qol(score):
    if score < 40:
        return 'Low'
    elif score < 70:
        return 'Medium'
    else:
        return 'High'
df['qol_class'] = df['qol_score'].apply(classify_qol)


cat_cols = ['location', 'trigger', 'occupation']
cat_cols = [col for col in cat_cols if col in df.columns]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


df.to_csv('MLDATA_with_QoL.csv', index=False)


X = df.drop(['qol_score', 'qol_class'], axis=1)
y_cls = df['qol_class']
y_reg = df['qol_score']

# REGRESSION 
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.15, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

print("\n📈 Regression Results:")
print("R2 Score:", round(r2_score(y_test_r, y_pred_r), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 2))

# CLASSIFICATION 


X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.15, random_state=42, stratify=y_cls)


sm = SMOTE(random_state=42)
X_train_c_sm, y_train_c_sm = sm.fit_resample(X_train_c, y_train_c)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train_c_sm, y_train_c_sm)
y_pred_c = clf.predict(X_test_c)

print("\n Classification Results:")
print("Accuracy:", round(accuracy_score(y_test_c, y_pred_c), 3))
print(classification_report(y_test_c, y_pred_c))


conf_mat = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  FEATURE IMPORTANCE 
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]

plt.figure(figsize=(10, 12))
sns.barplot(x=importances[indices], y=features)
plt.title("Feature Importance - Classification Model")
plt.tight_layout()
plt.show()

# Take input and predict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



def get_user_input():
    user_input = {}

    user_input['gender'] = input("Enter gender (Male/Female): ").strip().lower()
    user_input['weight'] = float(input("Enter weight (in kg): "))
    user_input['age'] = int(input("Enter age: "))
    user_input['painseverity_1_10_'] = int(input("Pain severity (1-10): "))
    user_input['duration'] = input("How long does the pain last (e.g., '5 minutes'): ")
    user_input['location'] = input("Where is the pain located? (e.g., 'Lower Back'): ").strip()
    user_input['trigger'] = input("Trigger of the pain? (e.g., 'Prolonged Sitting'): ").strip()
    user_input['occupation'] = input("Occupation: ").strip()
    user_input['anxietyscore'] = int(input("Anxiety score (0-10): "))
    user_input['depressionscore'] = int(input("Depression score (0-10): "))
    user_input['panicattackhistory_yes_no_'] = input("Panic attack history (Yes/No): ").strip().upper()
    user_input['hypertensionhistory_yes_no_'] = input("Hypertension history (Yes/No): ").strip().upper()
    user_input['medicationuse_yes_no_'] = input("Medication use (Yes/No): ").strip().upper()
    user_input['smoking_alcohol_etc_yes_no_'] = input("Smoking/Alcohol etc. (Yes/No): ").strip().upper()
    user_input['sleephours'] = float(input("Sleep hours per day: "))
    user_input['physicalactivity_hoursperweek_'] = float(input("Physical activity hours/week: "))
    user_input['socialinteraction_low_med_high_'] = input("Social interaction (Low/Med/High): ").strip().upper()

    return user_input


def predict_qol(user_input_dict, model, columns_template):
    input_df = pd.DataFrame([user_input_dict])

    
    input_df['gender'] = input_df['gender'].map({'male': 0, 'female': 1})
    input_df['panicattackhistory_yes_no_'] = input_df['panicattackhistory_yes_no_'].map({'YES': 1, 'NO': 0})
    input_df['hypertensionhistory_yes_no_'] = input_df['hypertensionhistory_yes_no_'].map({'YES': 1, 'NO': 0})
    input_df['medicationuse_yes_no_'] = input_df['medicationuse_yes_no_'].map({'YES': 1, 'NO': 0})
    input_df['smoking_alcohol_etc_yes_no_'] = input_df['smoking_alcohol_etc_yes_no_'].map({'YES': 1, 'NO': 0})
    input_df['socialinteraction_low_med_high_'] = input_df['socialinteraction_low_med_high_'].map({'LOW': 0, 'MED': 1, 'HIGH': 2})
    input_df['duration'] = input_df['duration'].str.extract(r'(\d+)').astype(float)

    input_df = pd.get_dummies(input_df)

   
    input_df = input_df.reindex(columns=columns_template, fill_value=0)

    
    prediction = model.predict(input_df)[0]
    return prediction




df = pd.read_csv("MLDATA_with_QoL.csv")
X = df.drop(['qol_score', 'qol_class'], axis=1)
y = df['qol_class']


model = RandomForestClassifier()
model.fit(X, y)




user_input = get_user_input()
prediction = predict_qol(user_input, model, X.columns)

print("\n🔮 Predicted QoL Class:", prediction)

if prediction == 'Low':
    print("""⚠️ High Risk of Musculoskeletal Chest Pain (MSCP)
Please consult a healthcare provider immediately.
Your current health data suggests a high likelihood of musculoskeletal chest pain, which may be significantly affecting your quality of life and mental well-being. MSCP, while non-cardiac in origin, can mimic serious conditions and may be worsened by psychological factors such as anxiety, depression, panic attacks, and hypertension.

📋 What You Should Know:
MSCP is not caused by heart problems, but it can still be debilitating.

It may stem from muscle tension, poor posture, or joint dysfunction in the chest or upper back.

Mental health plays a major role—stress and anxiety can amplify the perception of pain.

Ignoring symptoms can lead to chronic pain and reduced daily function.

🩺 What You Should Do:
✅ 1. Seek Immediate Medical Evaluation
Consult a primary care physician, physiotherapist, or pain specialist.

Ensure that any serious cardiac causes are ruled out first.

✅ 2. Begin a Pain Management Plan
A doctor may recommend pain relief, muscle relaxants, or targeted physical therapy.

Avoid self-medicating without professional advice.

✅ 3. Address Mental Health
Speak to a licensed therapist about managing stress, panic, or depressive symptoms.

Consider Cognitive Behavioral Therapy (CBT) or mindfulness techniques to reduce pain-related anxiety.

✅ 4. Monitor Your Vital Signs
Track blood pressure, heart rate, and pain episodes using a wearable or monitoring device.

Keep a daily log of symptoms, mood, sleep, and activity levels.

✅ 5. Make Lifestyle Changes
Improve posture and ergonomic habits.

Reduce intake of caffeine, processed foods, and salt.

Prioritize regular sleep, hydration, and gentle exercise like stretching or walking.

📞 When to Get Emergency Help:
If you experience any of the following, seek emergency medical care immediately:

Severe or sudden chest pain

Pain radiating to the arm, jaw, or neck

Shortness of breath, dizziness, or fainting

Sweating, nausea, or irregular heartbeat

🧠 Reminder:
Even though MSCP is non-cardiac, it is a real and painful condition. Getting early treatment and support can prevent it from becoming chronic and help restore your physical and emotional well-being.""")

elif prediction == 'Medium':
    print("""
    ⚠️ Moderate QoL Detected
🩺 Your current health status indicates a moderate risk level. Taking preventive action now can help improve your quality of life and avoid future complications.
✅ Recommended Preventive Measures
🧘 Mental & Emotional Well-being
🧠 Try Cognitive Behavioral Therapy (CBT)
Manage stress and anxiety that may be worsening your symptoms.

🕯️ Practice Mindfulness or Guided Meditation
Even 10–15 minutes daily can reduce panic and boost emotional balance.

📅 Schedule Regular Mental Health Check-ins
Speak with a counselor or therapist at least once a month.

🏃 Physical Health & Pain Management
🧍‍♂️ Start Light Physiotherapy
Target chest and upper back muscles to relieve tension and pain.

🪑 Improve Posture & Ergonomics
Adjust your workspace and sitting habits to reduce physical strain.

🧘‍♀️ Incorporate Daily Stretching & Strength Exercises
Just 10 minutes daily can prevent chronic MSCP.

🍎 Lifestyle Habits
🥗 Eat a Heart-Healthy, Anti-Inflammatory Diet
Include fruits, veggies, whole grains, and omega-3s. Cut back on salt, caffeine, and processed foods.

😴 Maintain a Consistent Sleep Schedule
Aim for 7–8 hours of quality sleep per night.

🚭 Avoid Smoking and Limit Alcohol
These can worsen anxiety, panic, and blood pressure.

📱 Self-Monitoring & Tools
📊 Track Your Mood and Pain Daily
Use a mobile app or journal to identify patterns and triggers.

🩺 Monitor Blood Pressure and Heart Rate
Use a home device or smartwatch to track trends over time.

⌚ Use Wearables for Sleep and Activity Monitoring
Stay informed about your stress levels and sleep quality.

👨‍⚕️ Medical Support & Education
🏥 Attend Regular Doctor Visits
Review symptoms, blood pressure, and mental health every 4–6 weeks.

🧾 Stay Informed About Your Condition
MSCP is not life-threatening but needs proper care. Learn more to reduce unnecessary worry.

💊 Take Medications as Prescribed
Adhere to your treatment plan for pain, anxiety, and hypertension.

""")
else:
    print("""✅ Good Quality of Life
👍 Low Risk of Musculoskeletal Chest Pain (MSCP)
Your current health data indicates a low risk of musculoskeletal chest pain, and your overall quality of life appears stable and positive. Great job! Maintaining both physical and mental well-being is key to preventing future discomfort.

🌟 What This Means:
You are functioning well physically and emotionally.

There are no major signs of MSCP or related psychological distress.

Your lifestyle choices and habits are likely helping protect your health.

🛡️ Keep Up the Good Work With These Tips:
🧘 Stay Mentally Balanced
Continue healthy routines like mindfulness, relaxation techniques, or journaling.

Maintain a positive support system—talk to friends, family, or a counselor when needed.

🏃‍♂️ Stay Physically Active
Include light strength training or cardio 3–5 times per week.

Stretch regularly to keep muscles and joints flexible.

🍽️ Eat Smart
Stick to a balanced, anti-inflammatory diet.

Stay hydrated and avoid excessive caffeine, sugar, or salty foods.

😴 Sleep Well
Prioritize 7–8 hours of sleep per night.

Maintain a consistent sleep schedule for optimal recovery and mood.

🔍 Continue Monitoring
Track your mood, activity, and vitals using a health app or wearable device.

Stay proactive—early detection is always better than reactive care.

✅ Final Note:
You're on the right path. Continue these healthy habits, and remember: Prevention is better than cure. Even with low risk, staying consistent with your health practices ensures long-term wellness and resilience against stress or future discomfort.

""")
