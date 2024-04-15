import argparse


def train_args():
    # python train.py --data './dataset/cat_dog' --epoch 1 --project './results'
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--size' , type =int, help ='Image size', default=32)
    parent_parser.add_argument('--data' , type =str, help ='Path you your main data folder', required=True)
    parent_parser.add_argument('--config' , type =str, help ='Path you your project', default='./config/config.yaml')

    parent_parser.add_argument('--epoch', type =int, help ='Set epoch number', required=True)
    parent_parser.add_argument('--batch', type =int, help ='Set epoch number', default=32)
    parent_parser.add_argument('--opt'  , type =str, help ='Set optimizer (use keras/tf document)',default='Adam')

    parent_parser.add_argument('--project' , type =str, help ='Path you your project', required=True)

    args = parent_parser.parse_args()

    
    return args