import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🚖 NYC Taxi Trip Predictor", layout="wide")
st.title("🚖 NYC Taxi Trip Duration Predictor")
st.markdown("This app predicts trip duration (in seconds) for NYC Green Taxi rides using 2019 data.")

# === 1. Load dataset from file ===
FILE_PATH = "train.csv"

if not os.path.exists(FILE_PATH):
    st.error(f"File '{FILE_PATH}' not found. Please ensure it's in the working directory.")
    st.stop()

# Check which columns exist

sample_df = pd.read_csv(FILE_PATH, nrows=5)
print(sample_df.columns.tolist())
expected_columns = ['pickup_datetime', 'dropoff_datetime']
missing_cols = [col for col in expected_columns if col not in sample_df.columns]

if missing_cols:
    st.error(f"Missing expected columns: {missing_cols}")
    st.stop()

df = pd.read_csv(FILE_PATH, nrows=200_000, parse_dates=expected_columns)

# === 2. Feature Engineering ===
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

# Drop invalid durations
df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 7200)]

# Optional: create distance if coordinates exist
if all(col in df.columns for col in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        φ1, φ2 = np.radians(lat1), np.radians(lat2)
        dφ = φ2 - φ1
        dλ = np.radians(lon2 - lon1)
        a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    df['distance_km'] = df.apply(lambda r: haversine(
        r['pickup_latitude'], r['pickup_longitude'],
        r['dropoff_latitude'], r['dropoff_longitude']), axis=1)
else:
    df['distance_km'] = df['trip_distance']

# === 3. Train Model ===
features = ['passenger_count', 'pickup_hour', 'pickup_day', 'distance_km']
df = df.dropna(subset=features)
X = df[features]
y = np.log1p(df['trip_duration'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmsle = np.sqrt(mean_squared_log_error(y_test, preds))

st.subheader("📊 Model Performance")
st.write(f"**RMSLE:** {rmsle:.3f}")
st.write(f"**R² Score:** {r2:.3f}")

# === 4. Actual vs Predicted Plot ===
fig, ax = plt.subplots()
ax.scatter(np.expm1(y_test), np.expm1(preds), alpha=0.2, s=10)
ax.plot([0, 3600], [0, 3600], '--r')
ax.set_xlabel("Actual duration (s)")
ax.set_ylabel("Predicted (s)")
st.pyplot(fig)

# === 5. Custom Prediction ===
st.subheader("🔮 Try Custom Prediction")
col1, col2, col3, col4 = st.columns(4)
passenger = col1.number_input("Passenger Count", min_value=1, max_value=6, value=1)
hour = col2.slider("Pickup Hour", 0, 23, 9)
day = col3.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
distance = col4.number_input("Trip Distance (km)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)

if st.button("Predict Trip Duration"):
    input_arr = np.array([[passenger, hour, day, distance]])
    pred = model.predict(input_arr)
    duration = np.expm1(pred[0])
    st.success(f"🚕 Estimated Trip Duration: **{duration:.0f} seconds** (~{duration/60:.1f} minutes)")
