import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load dataset
df = pd.read_csv("petrol_price_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print("✅ Dataset Loaded! Total Records:", len(df))
print(df.tail())

# Feature Engineering
df["Is_Friday"] = (df["Day_of_Week"] == "Fri").astype(int)
df["Near_Mid_Month"] = df["Date"].dt.day.between(12, 18).astype(int)
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Simple ML-like Scoring Function
def predict_score(row):
    score = 0
    score += row["Is_Friday"] * 4
    score += row["Is_Revision_Date"] * 5
    score += row["Near_Mid_Month"] * 3
    score += (1 if row["Global_Oil_Trend"] > 2 else 0) * 4
    score += (1 if row["Prev_Change"] > 10 else 0) * 3
    return score

df["Increase_Score"] = df.apply(predict_score, axis=1)
df["Prediction"] = np.where(df["Increase_Score"] >= 8, "Negative (Increase)", "Positive (Stable/Decrease)")

# === 6-Month Projection ===
future_dates = [datetime(2026, 5, 15) + timedelta(days=15*i) for i in range(12)]
future_df = pd.DataFrame({
    "Date": future_dates,
    "Is_Revision_Date": [1 if d.day in [1,15] else 0 for d in future_dates],
    "Is_Friday": [1 if d.strftime("%a") == "Fri" else 0 for d in future_dates],
    "Near_Mid_Month": [1 if d.day.between(12,18) else 0 for d in future_dates],
    "Global_Oil_Trend": np.random.uniform(1.5, 5.0, 12),
    "Prev_Change": [15, 8, -5, 12, 20, -10, 18, 6, 22, -8, 14, 10]
})

future_df["Increase_Score"] = future_df.apply(predict_score, axis=1)
future_df["Prediction"] = np.where(future_df["Increase_Score"] >= 8, "Increase", "Stable/Decrease")

print("\n=== 6-Month Projection (May–Nov 2026) ===")
print(future_df[["Date", "Prediction"]].head(6))

# === Visualization ===
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Price"], label="Historical Price", marker='o', color='blue')
plt.plot(future_df["Date"][:6], [415 + i*8 for i in range(6)], label="Projected Price (Trend)", linestyle='--', color='red')
plt.title("Pakistan Petrol Price Trend & 6-Month Projection")
plt.xlabel("Date")
plt.ylabel("Price (PKR)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save projection
future_df.to_csv("6_month_projection.csv", index=False)
print("\nProjection saved to 6_month_projection.csv")