
import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ------------------- Setup Cache -------------------
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ------------------- Load Historical Race Data -------------------
print("🔄 Loading 2024 Belgian GP race session (Spa)...")
race_session = fastf1.get_session(2024, "Belgium", "R")
race_session.load()
print("✅ Race session loaded.")

# ------------------- Prepare Lap Data -------------------
laps = race_session.laps[["Driver", "LapTime"]].copy()
laps.dropna(subset=["LapTime"], inplace=True)
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# ------------------- 2025 Qualifying Data -------------------
# These should reflect the latest qualifying session if available
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
        "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
        "Fernando Alonso", "Lance Stroll"
    ],
    "QualifyingTime (s)": [
        102.324, 102.450, 102.200, 102.560, 103.010,
        103.120, 102.875, 102.980, 103.240, 103.000,
        103.300, 103.400
    ]
})

# ------------------- Driver Code Mapping -------------------
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Fernando Alonso": "ALO", "Lance Stroll": "STR"
}
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# ------------------- Merge and Prepare Dataset -------------------
merged = qualifying_2025.merge(laps, left_on="DriverCode", right_on="Driver")
X = merged[["QualifyingTime (s)"]]
y = merged["LapTime (s)"]

if X.empty:
    raise ValueError("❌ No data after merging qualifying and race data. Check driver codes or session data.")

# ------------------- Train Machine Learning Model -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ------------------- Predict 2025 Belgian GP Race Times -------------------
qualifying_2025["PredictedRaceTime (s)"] = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025.sort_values(by="PredictedRaceTime (s)", inplace=True)

# ------------------- Output Results -------------------
print("\n🏁 Predicted 2025 Belgian GP Race Results 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# ------------------- Evaluate Model -------------------
y_test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_pred)
print(f"\n📊 Model Evaluation - MAE: {mae:.2f} seconds\n")

