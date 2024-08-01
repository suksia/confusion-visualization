import confusion
import argparse
from glob import glob
from os.path import join
import sys

PATH = sys.path[0]

parser = argparse.ArgumentParser(description = 'This is a demo of the confusion-visualization tool')

parser.add_argument('--image_path', 
                    metavar='i', 
                    type=str, 
                    help='path to raw images used for training', 
                    default=join(PATH, '../data/images/'))

parser.add_argument('--label_path', 
                    metavar='l', 
                    type=str, 
                    help='path to ground truth labels used for training', 
                    default=join(PATH, '../data/labels/'))

parser.add_argument('--prediction_path', 
                    metavar='p', 
                    type=str, 
                    help='path to prediction maps generated after training', 
                    default=join(PATH, '../data/predictions/'))

parser.add_argument('--results_path',
                    metavar='r',
                    type=str,
                    help='path to directory to save results', 
                    default=join(PATH, '../results/'))

args = parser.parse_args()

def main():
    image_set = sorted(glob(join(args.image_path, '*')))
    label_set = sorted(glob(join(args.label_path, '*')))
    prediction_set = sorted(glob(join(args.prediction_path, '*')))

    for img_idx in range(len(image_set)):
        print(image_set[img_idx])
        print(label_set[img_idx])
        print(prediction_set[img_idx])
        
        confusion.get_confusion(image_set[img_idx],
                    label_set[img_idx],
                    prediction_set[img_idx],
                    prefix_dir=args.results_path)

if __name__ == '__main__':
    main()