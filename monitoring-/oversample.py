import pandas as pd
from imblearn.over_sampling import RandomOverSampler
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

# Balance the dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Combine the resampled features and target into a single DataFrame
balanced_train_data = pd.DataFrame(X_resampled, columns=iris.feature_names)
balanced_train_data["target"] = y_resampled

# Print the class distribution before and after balancing
print("Class distribution before balancing:")
print(train_data["target"].value_counts())
print("\nClass distribution after balancing:")
print(balanced_train_data["target"].value_counts())
