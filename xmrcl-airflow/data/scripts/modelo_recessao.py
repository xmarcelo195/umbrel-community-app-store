# Check if packages are already installed
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

to_install = [
    'pandas',
    'numpy',
    'fredapi',
    'scikit-learn',
    'matplotlib',
    'plotly',
    'imbalanced-learn',]

for package in to_install:
    install(package)

# Start
import pandas as pd
from fredapi import Fred
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import plotly.express as px
import os


api_key = os.getenv('FRED_API_KEY')

fred = Fred(api_key=api_key)  # Replace with your API key

# Define indicators (exclude JTSJOL to extend dataset)
indicators = {
    'USREC': 'Recession Indicator',
    'T10Y3M': '10Y-3M Yield Spread',
    'UNRATE': 'Unemployment Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'PAYEMS': 'Nonfarm Payrolls',
    'UMCSENT': 'Consumer Sentiment',
    'USSLIND': 'Leading Economic Index',
    'HOUST': 'Housing Starts',
    'INDPRO': 'Industrial Production',
    'VIXCLS': 'VIX'
}

# Fetch data
data = {}
for code, name in indicators.items():
    series = fred.get_series(code)
    data[name] = series

# Print raw data ranges for diagnostics
print("Raw Data Ranges:")
for name, series in data.items():
    print(f"{name}: {series.index.min()} to {series.index.max()}")

# Create a DataFrame
df = pd.DataFrame(data)

# Resample to monthly frequency
df = df.resample('M').mean()

# Print initial shape and missing values
print("\nInitial DataFrame Shape:", df.shape)
print("Initial Missing Values:\n", df.isna().sum())

# Impute missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Print shape after imputation
print("\nShape After Imputation:", df.shape)
print("Missing Values After Imputation:\n", df.isna().sum())

# Shift recession indicator to predict 1 month ahead
df['Recession_1M_Ahead'] = df['Recession Indicator'].shift(-1).fillna(0)

# Calculate percentage changes and smoothed features
df['CPI_Inflation'] = df['Consumer Price Index'].pct_change() * 100
df['Payrolls_Growth'] = df['Nonfarm Payrolls'].pct_change() * 100
df['Consumer_Sentiment_Change'] = df['Consumer Sentiment'].pct_change() * 100
# df['LEI_Change'] = df['Leading Economic Index'].pct_change() * 100
df['Housing_Starts_Change'] = df['Housing Starts'].pct_change() * 100
df['INDPRO_Change'] = df['Industrial Production'].pct_change() * 100
df['VIX_Change'] = df['VIX'].pct_change() * 100

# Smoothed features
df['Payrolls_Growth_Smoothed'] = df['Payrolls_Growth'].rolling(3).mean()
df['CPI_Inflation_Smoothed'] = df['CPI_Inflation'].rolling(3).mean()

# Calculate momentum for yield spread (3-month vs. 12-month MA)
df['Yield_Spread_Momentum'] = df['10Y-3M Yield Spread'].rolling(3).mean() / df['10Y-3M Yield Spread'].rolling(12).mean()

# Impute missing values after feature engineering
df = df.fillna(method='ffill').fillna(method='bfill')

# Print final dataset range and recession months
print("\nFinal Dataset Range After Feature Engineering:", df.index.min(), "to", df.index.max())
print("Shape After Feature Engineering Imputation:", df.shape)
print("Recession Months in y:", df['Recession_1M_Ahead'].sum())

# Define features and target
features = [
    '10Y-3M Yield Spread', 'Unemployment Rate', 'CPI_Inflation', 'Payrolls_Growth',
    'Yield_Spread_Momentum', 'Consumer Sentiment', 'Consumer_Sentiment_Change',
    'Housing Starts', 'Housing_Starts_Change',
    'Industrial Production', 'INDPRO_Change',
    'VIX', 'VIX_Change', 'Payrolls_Growth_Smoothed', 'CPI_Inflation_Smoothed'
]
X = df[features]
y = df['Recession_1M_Ahead'].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

# Print class distribution
print("y_train Shape:", y_train.shape)
print("Class distribution in y_train:", np.bincount(y_train))
print("Class distribution in y:", np.bincount(y))

# Apply SMOTE if both classes are present
if len(np.unique(y_train)) > 1:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied")
else:
    print("SMOTE skipped: Only one class in y_train")

# Print test set range for diagnostics
print("\nTest Set Range:", X_test.index.min(), "to", X_test.index.max())

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set (for evaluation)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Predict recession probability for the next month
latest_data = X.tail(1)
latest_date = latest_data.index[0]
next_month = latest_date + pd.offsets.MonthEnd(1)
next_month_proba = model.predict_proba(latest_data)[:, 1][0]
print(f"\nRecession Probability for {next_month.strftime('%B %Y')}: {next_month_proba:.2%}")

# Predict probabilities for the entire dataset (to plot full range)
y_pred_proba_full = model.predict_proba(X)[:, 1]

# Create results DataFrame for the entire dataset
results = pd.DataFrame({
    'Date': X.index,
    'Actual': y,
    'Predicted_Probability': y_pred_proba_full
})

# Plot recession probability forecast for the full dataset
fig = px.line(results, x='Date', y='Predicted_Probability', title='Recession Probability Forecast (1 Month Ahead, Monthly)')
fig.add_scatter(x=results[results['Actual'] == 1]['Date'],
                y=results[results['Actual'] == 1]['Predicted_Probability'],
                mode='markers', name='Actual Recession', marker=dict(color='red', size=10))
fig.update_layout(yaxis_title='Probability of Recession', xaxis_title='Date')
fig.show()

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance in Recession Prediction')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

df
# results[results['Date']<= '2020-03-01']

# Fetch data
data = {}
for code, name in indicators.items():
    print(code,name)
    series = fred.get_series(code)
    data[name] = series

# Print raw data ranges for diagnostics
print("Raw Data Ranges:")
for name, series in data.items():
    print(f"{name}: {series.index.min()} to {series.index.max()}")

# Create a DataFrame
df = pd.DataFrame(data)

df.index.max()