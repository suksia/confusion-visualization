import cv2 as cv
from sklearn.metrics import f1_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_img(img_path, flag = cv.IMREAD_GRAYSCALE):
    # in: 
    #     path to image 
    #     reading flag (color depth / # of channels)
    # out: 
    #     image read as a numpy array
    
    return cv.imread(img_path, flag)
    

def binarize_img(img):
    # in:
    #     image array
    # out:
    #     binarized image using otsu thresholding
    
    img_threshd = cv.threshold(img.copy(), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    img_binary = (img_threshd/255).astype(int)
    return img_binary


def get_f1(label_binary, pred_binary):
    # in:
    #     binarized label and prediction
    # out:
    #     f1-score between provided label and prediction

    # arrays must be flattened to use scikit-learns f1-score function
    label_binary_flatten = np.ndarray.flatten(label_binary)
    pred_binary_flatten = np.ndarray.flatten(pred_binary)

    return f1_score(label_binary_flatten, pred_binary_flatten)


def get_mask(label_binary, pred_binary, compare_condition = (1,1), px_val = 255):
    # in:
    #     binarized label and prediction images
    #     comparison values for label and prediction (e.g. 1,1 is true positive)
    # out:
    #     mask where positive pixel values (0, 1/255) satisfy the comparison condition

    # use comparison conditions to get pixels were they are both satisfied
    mask = np.logical_and(label_binary==compare_condition[0], pred_binary==compare_condition[1])

    # convert from bool to int
    mask = mask.astype('uint8')

    # optionally scale from 1 to 255
    mask = mask*px_val

    return mask

def get_comparison(imgs: list, titles: list, destination):
    # in:
    #     set of images to visually compare
    #     set of associated titles
    #     path and name of output file
    # out:
    #     display a plot showing each image next to each other

    num_images = len(imgs)

    fig, axs = plt.subplots(1, num_images, figsize=(50, num_images*40))

    subplot_counter = 0
    for ax in axs:
        ax.imshow(imgs[subplot_counter], cmap='gray')
        ax.axis('off')
        ax.set_title(titles[subplot_counter], fontsize=40)

        subplot_counter += 1

    plt.subplots_adjust(wspace=0.05)

    fig.savefig(destination, bbox_inches="tight")
