# 📈 Stock Price Prediction System (Python + Machine Learning)

This project is a **Python-based Stock Price Prediction System** developed for academic submission.  
It predicts next-day stock closing prices using **Linear Regression**, technical indicators, lag features, and full preprocessing.  
All cleaned datasets, prediction outputs, evaluation metrics, and graphs are automatically saved into an `outputs/` folder.

---

## Project Features

### **1. Complete Data Preprocessing**
- Reads S&P 500 historical price data  
- Converts `Date` column to datetime  
- Sorts data chronologically  
- Removes unused `Adj Close` column  
- Converts `Volume` to numeric  
- Removes missing values  
- Saves cleaned dataset to:  
  `outputs/cleaned_dataset.csv`

---

### **2. Technical Indicator Generation**
The script automatically computes multiple stock market indicators:

- **SMA_10** – 10-day Simple Moving Average  
- **SMA_20** – 20-day Simple Moving Average  
- **EMA_12** – 12-day Exponential Moving Average  
- **EMA_26** – 26-day Exponential Moving Average  
- **MACD** – EMA₁₂ − EMA₂₆  
- **Return** – Daily percentage price change  

---

### **3. Lag Feature Engineering**
To incorporate time-series information, the following lag features are created:



These represent the previous 1–5 days’ closing prices.

---

### **4. Linear Regression Model**
Using scikit-learn:

- Dataset is split into **80% training** and **20% testing** (chronologically)  
- Trains a **Linear Regression** model  
- Predicts next-day closing price  
- Evaluates using:
  - **MAE** (Mean Absolute Error)  
  - **RMSE** (Root Mean Squared Error)  
  - **R² Score**  

Metrics saved to:



---

### **5. Prediction Output**
A final CSV containing:

- Date  
- Actual Closing Price  
- Predicted Closing Price  

Saved as:



---

### **6. Visualization Output**
Multiple charts are automatically generated and saved:

| Plot Name | Description | File |
|----------|-------------|------|
| Actual vs Predicted | Comparison of real vs predicted prices | actual_vs_predicted.png |
| Actual Prices | Real closing price timeline | actual_prices.png |
| Predicted Prices | Prediction trend | predicted_prices.png |
| Residual Plot | Error over time | residual_plot.png |
| Error Histogram | Error distribution | error_histogram.png |
| Scatter Plot | Actual vs predicted points | scatter_plot.png |

All stored inside the `outputs/` folder.

---

## Technologies Used
- Python  
- pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- Jupyter Notebook / VS Code (optional)

---

## Setup Requirements

### **1. Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn


