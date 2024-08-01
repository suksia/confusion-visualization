# Overview
After training a neural network for binary segmentation, it will produce a grayscale prediction map where each pixel represents the probability of belonging to the positive class (i.e. 0 is negative, and 1 is positive). 

Given the human label, which acts as the ground-truth, the correctness of each pixel-wise prediction can be classified using the confusion matrix:

- True positive
- True negative
- False positive (type 1 error)
- False negative (type 2 error)

It may be informative to see where predictions were correct/incorrect given the human label, and to further see the type of error made by the machine. This visualizes the disagreement between human and machine.

# 
