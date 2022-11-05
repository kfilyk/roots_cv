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

### Clean S3 Image Directory
Run `scripts/data_processing/clean_s3_image_directory.ipynb` to convert all PNG, WEBP, and GIF files to JPEG images within a specified S3 path. Furthermore, it deletes any duplicate images.

### Create Images from Bounding Boxes
Run `scripts/data_processing/images_from_bounding_boxes.ipynb` to create new images from bounding boxes.

### Split Output Manifest
Run `scripts/data_processing/split_output_manifest.ipynb` to manipulate the file `output.manifest` from SageMaker's Ground Truth object detection data job labelling. It will split records containing bounding boxes for plants, leaves, flowers, and fruits into multiple files for each present category.

### Output Manifest to Annotations
Run `scripts/data_processing/output_manifest_to_annotations.ipynb` to convert `output.manifest` from SageMaker's Ground Truth object detection data job labelling to create an `annotations.json` file. This format is needed to train object detection models.


## 2. Training Object Detection Model
### Train Pre-Trained Object Detection Model
Run `scripts/training/training_object_detection.ipynb` to train pre-trained object detection model. Different pre-trained models can be found at https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-tensorflow-Models.html.


## 3. Training Classification Model
### Train Pre-Trained Classification Model
Run `scripts/training/training_classification.ipynb` to train pre-trained classification model. Different pre-trained models can be found at https://docs.aws.amazon.com/sagemaker/latest/dg/IC-TF-Models.html.

### Incremental Training of Classification Model
Run `scripts/training/incremental_training_classification.ipynb` to perform incremental training on a model that has already been trained.


## 4. Inference
### Inference: Object Detection
Run `scripts/inference/inference_object_detection.ipynb` to perform object detection on a S3 bucket's prefix (directory).

### Inference: Image Classification
Run `scripts/inference/inference_classification.ipynb` to perform image classification on a S3 bucket's prefix (directory).

### Inference: Object Detection & Image Classification
Run `scripts/inference/inference_object_detection+classification.ipynb` to perform object detection then image classification on a S3 bucket's image key.