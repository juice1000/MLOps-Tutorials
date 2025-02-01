import pandas as pd
from evidently import ColumnMapping
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["target"] = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[iris.feature_names], data["target"], test_size=0.2, random_state=42
)

# Combine X_train and y_train into a single DataFrame
train_data = X_train.copy()
train_data["target"] = y_train

# Create a column mapping for evidently
column_mapping = ColumnMapping(target="target", numerical_features=iris.feature_names)

# Create a report with the DataBalanceMetric
data_stability = TestSuite(
    tests=[
        DataStabilityTestPreset(),
    ]
)
data_stability.run(
    reference_data=train_data, current_data=train_data, column_mapping=column_mapping
)

# Print the report (only works in jupyter notebooks)
data_stability.show()
