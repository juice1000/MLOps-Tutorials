# https://docs.greatexpectations.io/docs/core/introduction/try_gx
# Version for demo 1.0.0

# Prequisites: Python version 3.8 to 3.11

# pip install pandas
# pip install great_expectations


# 1- Import the following libraries
import great_expectations as gx

# Pandas ->
import pandas as pd
from great_expectations import expectations as gxe
from sklearn.datasets import load_iris

# -> Line to check gx core version:
print(gx.__version__)

# 2- Download and read the sample data into a Pandas DataFrame.
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 3- A Data Context object serves as the entrypoint for interacting with GX components.
context = gx.get_context(mode="file")


# 4- Connect to data and create a Batch.
# Define a Data Source, Data Asset, Batch Definition, and Batch. The Pandas DataFrame is provided to the Batch Definition at runtime to create the Batch.
data_source_name = "iris"
asset_name = "iris"
batch_definition_name = "iris_batch_definition"
validation_definition_name = "iris_validation_definition"
try:
    data_source = context.data_sources.add_pandas(data_source_name)
except:
    data_source = context.data_sources.get(data_source_name)


try:
    data_asset = data_source.add_dataframe_asset(name=asset_name)
except:
    data_asset = data_source.get_asset(name=asset_name)

try:
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        batch_definition_name
    )
except:
    batch_definition = data_asset.get_batch_definition(batch_definition_name)

batch_parameters = {"dataframe": df}
batch = batch_definition.get_batch(batch_parameters=batch_parameters)


# Create an Expectation.
# Expectations are a fundamental component of GX. They allow you to explicitly define the state to which your data should conform.

# Build Expectations and add to expectation suite
try:
    suite = context.suites.get("iris_suite")
except:
    suite = context.suites.add(gx.ExpectationSuite(name="iris_suite"))

expectation = gxe.ExpectColumnValuesToBeBetween(
    column="sepal length (cm)", min_value=2, max_value=10
)

# Get a validation definition
validation_definition = gx.ValidationDefinition(
    data=batch_definition, suite=suite, name=validation_definition_name
)
try:
    validation_definition = context.validation_definitions.get(
        validation_definition_name
    )
except:
    validation_definition = context.validation_definitions.add(validation_definition)

validation_results = validation_definition.run(batch_parameters=batch_parameters)

# Run and get the results!
# validation_result = batch.validate(expectation)

# Action on the validation result
validation_definitions = [validation_definition]  # can be multiple definitions

if validation_results.success:
    print("Validation passed")
else:
    raise ValueError("Validation failed", validation_results)
