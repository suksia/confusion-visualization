# Overview
After training a neural network for binary segmentation, it will produce a grayscale prediction map where each pixel represents the probability of belonging to the positive class (i.e. 0 is negative, and 1 is positive). 

Given the human label, which acts as the ground-truth, the correctness of each pixel-wise prediction can be classified using the confusion matrix:

- True positive
- True negative
- False positive (type 1 error)
- False negative (type 2 error)

It may be informative to see where predictions were correct/incorrect given the human label, and to further see the type of error made by the machine. This visualizes the disagreement between human and machine and helps show biased labelling.

# Usage

The script ```src/demo.py``` performs a demo of the program using some provided data in ```data/```. It can also be used with custom data by supplying arguments that define paths to the top-level data directory and its sub-directories containing training images, labels, and output predictions. The user can also define a custom path (absolute, or relative) to a directory that will contain program output. Use ```demo.py --help``` to see more information about customizing supplied paths.

Predictions should be a result of binary segmentation; multiclass objectives are not supported. Each image/label/prediction pair should also have the same shape and filenames, though different (but consistent) extensions can be used.

# Output Results

The program creates a new directory that contains three sub-directories:

- ```confusion_masks/```
    - Standalone images with each pixel colored according to the confusion mask
- ```confusion_compare/```
    - Raw images, human labels, and confusion masks plotted together with the F1 found from Otsu thresholding
- ```prediction_compare/```
    - Raw images, human labels, and raw output predictions plotted together

