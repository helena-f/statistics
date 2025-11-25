import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Read the data file
# Skip header lines and read the data starting from line 28
# Note: line 29 is blank, so we skip 28 lines total
data = pd.read_csv('mpg.dat', sep='\t', skiprows=28, header=0, 
                   names=['MAKE', 'MODEL', 'VOL', 'HP', 'MPG', 'SP', 'WT'])

# Clean the data - remove any rows with missing values
data = data.dropna()

# Convert numeric columns to float (in case they're read as strings)
numeric_cols = ['VOL', 'HP', 'MPG', 'SP', 'WT']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove any rows that couldn't be converted
data = data.dropna()

# Extract features (X) and target (y)
# Features: VOL, HP, SP, WT
# Target: MPG
X = data[['VOL', 'HP', 'SP', 'WT']]
y = data['MPG']

# Fit multiple linear regression using scikit-learn
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Print results
print("Multiple Linear Regression Model to Predict MPG")
print("=" * 50)
print(f"\nCoefficients:")
print(f"  Intercept: {model.intercept_:.4f}")
print(f"  VOL (cab space): {model.coef_[0]:.4f}")
print(f"  HP (horsepower): {model.coef_[1]:.4f}")
print(f"  SP (top speed): {model.coef_[2]:.4f}")
print(f"  WT (weight): {model.coef_[3]:.4f}")
print(f"\nR-squared: {r2:.4f}")

# Fit using statsmodels for detailed statistics
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()

print("\n" + "=" * 50)
print("Detailed Statistics (using statsmodels):")
print("=" * 50)
print(model_sm.summary())

