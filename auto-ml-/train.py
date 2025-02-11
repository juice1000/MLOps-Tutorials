import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

# Load the dataset
df = pd.read_csv("data/preprocessed.csv")
X = df.drop(columns=["Item_Outlet_Sales"])
y = df["Item_Outlet_Sales"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize TPOT
tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)

# Fit TPOT to find the best model
tpot.fit(X_train, y_train)

# Evaluate the best model
score = tpot.score(X_test, y_test)
print(f"Best model accuracy: {score:.4f}")

# Export the optimized pipeline as a Python script
tpot.export("./tpot_item_outlet_pipeline.py")

# Print the best found pipeline
print("\nBest pipeline:")
print(tpot.fitted_pipeline_)
