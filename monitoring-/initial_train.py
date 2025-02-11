import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from train import train

iris = load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_frame["target"] = iris.target

reference_data = iris_frame.iloc[:100]  # Reference data

X_train, X_test, y_train, y_test = train_test_split(
    reference_data[iris.feature_names],
    reference_data["target"],
    test_size=0.2,
    random_state=42,
)

train(X_train, y_train, X_test, y_test)
