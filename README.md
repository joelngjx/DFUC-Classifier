# DFUC-Classifier
**This repository is a Diabetic Foot Ulcer (DFU) classification project made using Python, utilising various common libraries in Machine Learning including PyTorch, albumentations, segmentation_models_pytorch and more. Explanations for important files in the repository are provided below:**

## `test.py`: Main project file 
* Preprocesses DFU images from a dataset of 2000 images and masks using `albumentations`
* Instantiates a UNet model for segmentation using the library `segmentation_models_pytorch`
* Runs a train-test split on the dataset
* Trains the model
* Calculates evaluation metrics (Loss calculated using a combination of `DiceLoss` and `FocalLoss`; IoU calculated using a custom function)
    - IoU is calculated using the formula `IoU = Area of Intersection/Area of Union`
* Saves the model for future use in a GUI under `models/DFUCmodelV0.pt`


## `gui.py`: GUI file implementing streamlit
* Loads the saved DFU classification model
* Prompts the user to upload the original DFU image along with its mask
* Preprocesses both images
* Calculates and outputs evaluation metrics
* Creates a mask image based on predictions