# Setup XGBoost Estimator
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


def init_training_step(train_args):

    return TrainingStep(
        name="AbaloneTrain",
        step_args=train_args,
    )


def setup_xgboost_model(pipeline_session, step_process):
    region = sagemaker.Session().boto_region_name
    role = sagemaker.get_execution_role()
    instance_type = "ml.m5.xlarge"
    default_bucket = sagemaker.Session().default_bucket()

    model_path = f"s3://{default_bucket}/AbaloneTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        output_path=model_path,
        role=role,
        sagemaker_session=pipeline_session,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )

    train_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )
    return train_args
