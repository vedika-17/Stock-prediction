import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

sns.set(style="whitegrid", palette="deep")

path = r"C:\Users\karth\OneDrive\Desktop\PROJECTS\sp500.csv"
df = pd.read_csv(path)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

os.makedirs("outputs", exist_ok=True)

if "Adj Close" in df.columns:
    df.drop("Adj Close", axis=1, inplace=True)

df["Volume"] = df["Volume"].astype(float)
df = df.dropna()

df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Return"] = df["Close"].pct_change()

for lag in range(1, 6):
    df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

df["Target"] = df["Close"].shift(-1)
df = df.dropna().reset_index(drop=True)

df.to_csv("outputs/cleaned_dataset.csv", index=False)

features = [
    "Open","High","Low","Close","Volume",
    "SMA_10","SMA_20","EMA_12","EMA_26","MACD",
    "Return"
] + [f"Close_lag_{i}" for i in range(1,6)]

X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)


mae = mean_absolute_error(y_test, predicted)
rmse = np.sqrt(mean_squared_error(y_test, predicted))
r2 = r2_score(y_test, predicted)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Value": [mae, rmse, r2]
})
metrics_df.to_csv("outputs/metrics.csv", index=False)

pred_df = pd.DataFrame({
    "Date": df["Date"].iloc[-len(predicted):].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predicted
})
pred_df.to_csv("outputs/predicted_prices.csv", index=False)

plt.figure(figsize=(12,5))
sns.lineplot(x="Date", y="Actual_Close", data=pred_df, label="Actual")
sns.lineplot(x="Date", y="Predicted_Close", data=pred_df, label="Predicted", linestyle="dotted")
plt.title("Actual vs Predicted Stock Price")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

plt.figure(figsize=(12,5))
sns.lineplot(x="Date", y="Actual_Close", data=pred_df)
plt.title("Actual Closing Prices")
plt.tight_layout()
plt.savefig("outputs/actual_prices.png")
plt.show()

plt.figure(figsize=(12,5))
sns.lineplot(x="Date", y="Predicted_Close", data=pred_df, color="orange")
plt.title("Predicted Closing Prices")
plt.tight_layout()
plt.savefig("outputs/predicted_prices.png")
plt.show()

residuals = pred_df["Predicted_Close"] - pred_df["Actual_Close"]
plt.figure(figsize=(12,4))
sns.lineplot(x=pred_df["Date"], y=residuals)
plt.axhline(0, color='black')
plt.title("Residual Errors Over Time")
plt.tight_layout()
plt.savefig("outputs/residual_plot.png")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Error Histogram")
plt.tight_layout()
plt.savefig("outputs/error_histogram.png")
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x="Actual_Close", y="Predicted_Close", data=pred_df, alpha=0.6)
plt.plot(
    [pred_df["Actual_Close"].min(), pred_df["Actual_Close"].max()],
    [pred_df["Actual_Close"].min(), pred_df["Actual_Close"].max()],
    'r--'
)
plt.title("Actual vs Predicted (Scatter)")
plt.tight_layout()
plt.savefig("outputs/scatter_plot.png")
plt.show()

print("All graphs displayed AND saved in: outputs/")
