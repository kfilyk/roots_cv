from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
from sagemaker.tuner import HyperparameterTuner
from sagemaker import (
    get_execution_role, image_uris, model_uris, script_uris
)


OD_METRIC_DEFINITIONS = [
    {
        "Name": "val_cross_entropy",
        "Regex": r"Val_CrossEntropy=(\d+\.\d+), Val_SmoothL1=\d+\.\d+"
    },
    {
        "Name": "val_smooth_l1",
        "Regex": r"Val_CrossEntropy=\d+\.\d+, Val_SmoothL1=(\d+\.\d+)"
    },
]


IC_METRIC_DEFINITIONS = [
    {
        "Name": "val_loss",
        "Regex": r"\d+\/\d+ - \d+m?s - loss: \d+\.\d+ - accuracy: \d+\.\d+ - val_loss: (\d+\.\d+) - val_accuracy: \d+\.\d+ - \d+m?s\/epoch - \d+m?s\/step"
    },
    {
        "Name": "val_accuracy",
        "Regex": r"\d+\/\d+ - \d+m?s - loss: \d+\.\d+ - accuracy: \d+\.\d+ - val_loss: \d+\.\d+ - val_accuracy: (\d+\.\d+) - \d+m?s\/epoch - \d+m?s\/step"
    },
    {
        "Name": "best_val_accuracy",
        "Regex": r"\d+\/\d+ - \d+m?s - loss: \d+\.\d+ - accuracy: (\d+\.\d+) - \d+m?s\/epoch - \d+m?s\/step"
    },
]


class BaseEstimator:
    def __init__(
        self, model_id, model_version, hyperparameters, instance_type, instance_count,
        max_run, output_path, model_uri=None, base_job_name=None
    ):
        aws_role = get_execution_role()
        
        # Retrieve the docker image
        image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            model_id=model_id,
            model_version=model_version,
            image_scope="training",
            instance_type=instance_type,
        )

        # Retrieve the training script
        source_uri = script_uris.retrieve(
            model_id=model_id,
            model_version=model_version,
            script_scope="training"
        )

        if not model_uri:
            # Retrieve the pre-trained model tarball to further fine-tune
            model_uri = model_uris.retrieve(
                model_id=model_id,
                model_version=model_version,
                model_scope="training"
            )
            
        if not base_job_name:
            base_job_name = name_from_base(model_id)
        self.base_job_name = base_job_name
        
        # required to properly save model to s3 bucket
        hyperparameters["model_dir"] = "/opt/ml/model"
        self._estimator = None

        
    def fit(self, *args, **kwargs):
        self._estimator.fit(*args, **kwargs)
        

class ObjectDetectionEstimator(BaseEstimator):
    def __init__(
        self, model_id, model_version, hyperparameters, instance_type, instance_count,
        max_run, output_path, model_uri=None, base_job_name=None
    ):
        super().__init__(
            model_id, model_version, hyperparameters, instance_type, instance_count,
            max_run, output_path, model_uri, base_job_name
        )
        # Create SageMaker Estimator instance
        self._estimator = Estimator(
            role=aws_role,
            image_uri=image_uri,
            source_dir=source_uri,
            model_uri=model_uri,
            entry_point="transfer_learning.py",
            instance_count=instance_count,
            instance_type=instance_type,
            max_run=max_run,
            hyperparameters=hyperparameters,
            enable_sagemaker_metrics=True,
            metric_definitions=OD_METRIC_DEFINITIONS,
            output_path=output_path,
            base_job_name=base_job_name,
        )
        

class ImageClassificationEstimator(BaseEstimator):
    def __init__(
        self, model_id, model_version, hyperparameters, instance_type, instance_count,
        max_run, output_path, model_uri=None, base_job_name=None
    ):
        super().__init__(
            model_id, model_version, hyperparameters, instance_type, instance_count,
            max_run, output_path, model_uri, base_job_name
        )
        # Create SageMaker Estimator instance
        self._estimator = Estimator(
            role=aws_role,
            image_uri=image_uri,
            source_dir=source_uri,
            model_uri=model_uri,
            entry_point="transfer_learning.py",
            instance_count=instance_count,
            instance_type=instance_type,
            max_run=max_run,
            hyperparameters=hyperparameters,
            enable_sagemaker_metrics=True,
            metric_definitions=IC_METRIC_DEFINITIONS,
            output_path=output_path,
            base_job_name=base_job_name,
        )
        

class BaseHPTuner:
    def __init__(self, estimator, hyperparameter_ranges, max_jobs, max_parallel_jobs, base_job_name=None):
        if not base_job_name:
            base_job_name = estimator.base_job_name
        self._estimator = None

        
class ImageClassificationHPTuner(BaseHPTuner):
    def __init__(self, estimator, hyperparameter_ranges, max_jobs, max_parallel_jobs, base_job_name=None):
        super().__init__(
            estimator, hyperparameter_ranges, max_jobs, max_parallel_jobs, base_job_name
        )
        self._estimator = HyperparameterTuner(
            estimator=estimator,
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            metric_definitions=IC_METRIC_DEFINITIONS,
            objective_metric_name="best_val_accuracy",
            objective_type="Maximize",
            early_stopping_type="Auto",
            base_tuning_job_name=base_job_name,
        )
        
        
class ObjectDetectionHPTuner(BaseHPTuner):
    def __init__(self, estimator, hyperparameter_ranges, max_jobs, max_parallel_jobs, base_job_name=None):
        super().__init__(
            estimator, hyperparameter_ranges, max_jobs, max_parallel_jobs, base_job_name
        )
        self._estimator = HyperparameterTuner(
            estimator=estimator,
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            metric_definitions=OD_METRIC_DEFINITIONS,
            objective_metric_name="val_cross_entropy",
            objective_type="Minimize",
            early_stopping_type="Auto",
            base_tuning_job_name=base_job_name,
        )