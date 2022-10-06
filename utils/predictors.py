import json
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
from sagemaker import get_execution_role, image_uris, script_uris, model_uris


class BasePredictor:
    def __init__(self, model_id, model_version, instance_type, model_uri=None, endpoint_name=None):
        aws_role = get_execution_role()
        
        # Retrieve the inference docker container uri
        image_uri = image_uris.retrieve(
            region=None,
            framework=None,  # automatically inferred from model_id
            image_scope="inference",
            model_id=model_id,
            model_version=model_version,
            instance_type=instance_type,
        )

        # Retrieve the inference script uri. This includes scripts for model loading, inference handling etc.
        source_uri = script_uris.retrieve(
            model_id=model_id,
            model_version=model_version,
            script_scope="inference"
        )

        if not model_uri:
            # Retrieve the base model uri
            model_uri = model_uris.retrieve(
                model_id=model_id,
                model_version=model_version,
                model_scope="inference"
            )
            
        if not endpoint_name:
            endpoint_name = name_from_base(model_id)

        # Create the SageMaker model instance
        self._model = Model(
            image_uri=image_uri,
            source_dir=source_uri,
            model_data=model_uri,
            entry_point="inference.py",  # entry point file in source_dir and present in ic_source_uri
            role=aws_role,
            predictor_cls=Predictor,
            name=endpoint_name,
        )
        
        self._deployed = False  # keeps track of whether the model has been deployed or not
        
        
    def deploy(self, instance_type, instance_count):
        """
        Deploy the Model. Note that we need to pass Predictor class when we deploy model through Model class,
        for being able to run inference through the sagemaker API.
        """
        if self._deployed:
            return
        
        self._model_predictor = self._model.deploy(
            instance_type=instance_type,
            initial_instance_count=instance_count,
            predictor_cls=Predictor,
        )
        
        self._deployed = True
        
    
    def predict(self, data, args=None):
        """ Predict data using predictive model. Implementation changes depending on model. """
        pass
        
    
    def delete(self):
        """ Delete the SageMaker endpoint and the attached resources. """
        if not self._deployed:
            return
        
        self._model_predictor.delete_model()
        self._model_predictor.delete_endpoint()
        
        self._deployed = False
        
        
class ObjectDetectionPredictor(BasePredictor):
    def predict(self, data, args=None):
        if not args:
            args = {
                "ContentType": "application/x-image",
                "Accept": "application/json;verbose"
            }
            
        query_response = self._model_predictor.predict(data, args)
        model_predictions = json.loads(query_response)
        normalized_boxes, classes, scores, labels = (
            model_predictions["normalized_boxes"],
            model_predictions["classes"],
            model_predictions["scores"],
            model_predictions["labels"]
        )
        # Substitute the classes index with the classes name
        class_names = [labels[int(idx)] for idx in classes]
        return normalized_boxes, class_names, scores, labels
    

class ImageClassificationPredictor(BasePredictor):
    def predict(self, data, args=None):
        if not args:
            args = {
                "ContentType": "application/x-image",
                "Accept": "application/json;verbose",
            }
            
        query_response = self._model_predictor.predict(data, args)
        model_predictions = json.loads(query_response)
        labels, probabilities = model_predictions["labels"], model_predictions["probabilities"]
        
        # sort labels and probabilities by descending order of probabilities
        sorted_idx = sorted(
            range(len(probabilities)), key=lambda index: probabilities[index], reverse=True
        )
        labels = [labels[idx] for idx in sorted_idx]
        probabilities = [probabilities[idx] for idx in sorted_idx]
        
        return labels, probabilities