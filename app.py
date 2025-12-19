import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("taxi_trip_pricing.csv")

# Handle missing values (same as notebook)
df.fillna(method="ffill", inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("Trip_Price", axis=1)
y = df["Trip_Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚕 Taxi Trip Price Prediction")
st.write("Predict taxi trip price using Machine Learning")

st.sidebar.header("Enter Trip Details")

trip_distance = st.sidebar.slider(
    "Trip Distance (km)",
    float(X["Trip_Distance_km"].min()),
    float(X["Trip_Distance_km"].max()),
    float(X["Trip_Distance_km"].mean())
)

trip_duration = st.sidebar.slider(
    "Trip Duration (minutes)",
    float(X["Trip_Duration_Minutes"].min()),
    float(X["Trip_Duration_Minutes"].max()),
    float(X["Trip_Duration_Minutes"].mean())
)

passenger_count = st.sidebar.selectbox(
    "Passenger Count",
    sorted(X["Passenger_Count"].unique())
)

base_fare = st.sidebar.slider(
    "Base Fare",
    float(X["Base_Fare"].min()),
    float(X["Base_Fare"].max()),
    float(X["Base_Fare"].mean())
)

per_km_rate = st.sidebar.slider(
    "Per Km Rate",
    float(X["Per_Km_Rate"].min()),
    float(X["Per_Km_Rate"].max()),
    float(X["Per_Km_Rate"].mean())
)

per_min_rate = st.sidebar.slider(
    "Per Minute Rate",
    float(X["Per_Minute_Rate"].min()),
    float(X["Per_Minute_Rate"].max()),
    float(X["Per_Minute_Rate"].mean())
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    df["Time_of_Day"].unique()
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    df["Day_of_Week"].unique()
)

traffic = st.sidebar.selectbox(
    "Traffic Conditions",
    df["Traffic_Conditions"].unique()
)

weather = st.sidebar.selectbox(
    "Weather",
    df["Weather"].unique()
)

# Encode inputs
input_data = pd.DataFrame([[
    trip_distance,
    time_of_day,
    day_of_week,
    passenger_count,
    traffic,
    weather,
    base_fare,
    per_km_rate,
    per_min_rate,
    trip_duration
]], columns=X.columns)

for col in input_data.select_dtypes(include="object").columns:
    input_data[col] = le.fit_transform(input_data[col])

# Prediction
if st.button("Predict Trip Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Trip Price: ₹ {prediction:.2f}")
