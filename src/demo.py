import confusion
import argparse
import sys
from utils import get_img_fns
from os.path import join
import time

PATH = sys.path[0]

parser = argparse.ArgumentParser(description = 'This is a demo of the confusion-visualization tool')

parser.add_argument('--data_path', 
                    metavar='D', 
                    type=str, 
                    help='path to the directory containing the data', 
                    default=join(PATH, '../data/'))

parser.add_argument('--image_path', 
                    metavar='I', 
                    type=str, 
                    help='path (relative to data_path) to the images used for training', 
                    default=join(PATH, '../data/images/'))

parser.add_argument('--label_path', 
                    metavar='L', 
                    type=str, 
                    help='path (relative to data_path) to the ground truth labels used for training', 
                    default=join(PATH, '../data/labels/'))

parser.add_argument('--prediction_path', 
                    metavar='P', 
                    type=str, 
                    help='path (relative to data_path) to the predictions generated after training', 
                    default=join(PATH, '../data/predictions/'))

parser.add_argument('--results_path',
                    metavar='R',
                    type=str,
                    help='path to directory to save results', 
                    default=join(PATH, '../results/'))

args = parser.parse_args()

def main():
    image_set = get_img_fns(join(args.data_path, args.image_path))
    label_set = get_img_fns(join(args.data_path, args.label_path))
    prediction_set = get_img_fns(join(args.data_path, args.prediction_path))

    for img_idx in range(len(image_set)):
        
        progress = int((img_idx / len(image_set))*100)

        confusion.get_confusion(image_set[img_idx],
                                label_set[img_idx],
                                prediction_set[img_idx],
                                prefix_dir=args.results_path)
        
        print(f"Progress: {progress}%", end='\r')
        
    time.sleep(1)
    print(f"Progress: 100%")
    
if __name__ == '__main__':
    main()