# AVA Roots Computer Vision

## Setup environment
Run the following to copy the template environment file:
```
cp template.env env
```
The environment variables inside `env` will be used in the pipeline's jupyter notebooks.


## 1. Data Processing
### Download GBIF Images
Run `scripts/data_processing/download_GBIF_images.ipynb` to download images from GBIF. The metadata files were downloaded from https://www.gbif.org/occurrence/search.

### Crop Plant Images
Run `scripts/data_processing/crop_plant_images.ipynb` to crop plant images from images.

### Split Object Detection Data Labelling Output Manifest
Run `scripts/data_processing/split_object_detection_output_manifest.ipynb` to manipulate the file `output.manifest` from SageMaker's Ground Truth object detection data job labelling. It will split records containing bounding boxes for both plants and leaves into two different files, e.i. one file `plant_output.manifest` will contain bounding boxes for plants and the other file `leaf_output.manifest` will contain bounding boxes for leaves.


## 2. Training Classification Model
### Train Pre-Trained Classification Model
Run `scripts/training/training_classification.ipynb` to train pre-trained classification model. Different pre-trained models can be found at https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html. Search for `-ic` to get a list of the various pre-trained classification models available.

### Incremental Training of Classification Model
Run `scripts/training/incremental_training_classification.ipynb` to perform incremental training on a model that has already been trained.


## 3. Inference
### Inference: Image Classification
Run `scripts/inference/inference_classification.ipynb` to perform image classification on a S3 bucket's prefix (directory).

### Inference: Object Detection & Image Classification
Run `scripts/inference/inference_object_detection+classification.ipynb` to perform object detection then image classification on a S3 bucket's image key.