import cv2 as cv
from sklearn.metrics import f1_score
import numpy as np
from os.path import split, splitext, isdir, join
from os import mkdir, listdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

def get_confusion(img_path, label_path, pred_path, prefix_dir=''):
    # in:
    #     path to image, label, and prediction
    # out:
    #     mask with each pixel prediction classified according to the confusion matrix
    
    # get image name
    filename_w_ext = split(img_path)[1]
    filename = splitext(filename_w_ext)[0]
    
    # open images
    img = load_img(img_path)
    label = load_img(label_path)
    pred = load_img(pred_path)
    
    # binarize prediction
    pred_binary = binarize_img(pred)
    label_binary = binarize_img(label)
    
    # generate masks with pixels classified according to confusion matrix
    tp_mask = get_mask(label_binary, pred_binary, compare_condition=(1,1))
    tn_mask = get_mask(label_binary, pred_binary, compare_condition=(0,0), px_val=1)
    fp_mask = get_mask(label_binary, pred_binary, compare_condition=(0,1))
    fn_mask = get_mask(label_binary, pred_binary, compare_condition=(1,0))
    
    # let true-negatives be the original image pixels
    base_channel = np.multiply(img, tn_mask)
    
    # define color channels as true negative filter overlayed with masks
    b_channel = np.add(base_channel, fp_mask)
    g_channel = np.add(base_channel, tp_mask)
    r_channel = np.add(base_channel, fn_mask)
    
    # create composite mask and move image channel axis to end for correct shape
    mask = np.array([r_channel, g_channel, b_channel])
    mask = np.moveaxis(mask, 0, -1)
    
    # get the f1-score for the binarized image
    pred_f1 = get_f1(label_binary, pred_binary)
    pred_f1 = round(pred_f1, 4)
    
    # define destination for saving masks 
    mask_dir = join(prefix_dir, 'confusion_masks')
    
    if not isdir(mask_dir):
        mkdir(mask_dir)
    
    # save confusion mask
    cv.imwrite(join(mask_dir, filename+'.png'), mask)
    
    # define destination for saving comparisons   
    confusion_compare_dir = join(prefix_dir, 'confusion_compare')
    if not isdir(confusion_compare_dir):
        mkdir(confusion_compare_dir)
    prediction_compare_dir = join(prefix_dir, 'prediction_compare')
    if not isdir(prediction_compare_dir):
        mkdir(prediction_compare_dir)
        
    # compare raw image, human label, and unprocessed AI prediction   
    get_comparison([img, label, pred],
                   ["Raw Image", "Human Label", "AI Prediction"],
                   join(prediction_compare_dir, filename+'.png'))
    
    # compare raw image, human label, and confusion representation
    get_comparison([img, label, mask],
                   ["Raw Image", "Human Label", f"AI Prediction (F1-Score = {pred_f1})"],
                   join(confusion_compare_dir, filename+'.png'))