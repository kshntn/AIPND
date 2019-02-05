import numpy as np

import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import argparse
from PIL import Image

import json



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # model = models.densenet121(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # class_to_idx = checkpoint['class_to_idx']
    # # hidden_units=checkpoint['hidden_units']
    # arch=checkpoint['arch']
    # model, _, _ = create_model(arch,class_to_idx,hidden_units)
    #
    # # model.classifier = classifier



    hidden_units=checkpoint['hidden_units']
    arch=checkpoint['arch']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)

    if arch=="vgg16" or arch=="vgg19":
        input_size = model.classifier[0].in_features
    elif arch=="densenet121":
        input_size = model.classifier.in_features


    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    print('model design')
    print(model)
    return model


def process_image(image):

    print(image.size)
    width, height = image.size

    if width > height:
        image.thumbnail((width, 256))
    else:
        image.thumbnail((256, height))

    width, height = image.size

    print(width, height)
    width, height = image.size
    left = (width - 224) / 2
    bottom = (height - 224) / 2
    right = left + 224
    top = bottom + 224

    crop_image = image.crop((left, bottom, right, top))
    print(crop_image.size)
    np_image = np.array(crop_image)
    np_image = np_image.astype(np.float32)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_image = (np_image - mean) / std

    pro_image = new_image.transpose((2, 0, 1))

    return pro_image

def predict(image_path, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(args.labels, 'r') as f:
        cat_to_name = json.load(f)
    model=load_checkpoint(args.checkpoint)
    model.to('cpu')
    model.eval()
    img = Image.open(image_path)
    img=process_image(img)

    img = torch.from_numpy(img).type(torch.FloatTensor)
    img.unsqueeze_(0)
    output=model.forward(img)

    ps = torch.exp(output)
    print(ps)

    probablity, index = torch.topk(ps, topk)
    index=index.numpy()
    index=index[0]
    probablity=probablity.detach().numpy()
    probablity=probablity[0]
    print(probablity)
# print(model.class_to_idx.items())

    indices={val: key for key, val in model.class_to_idx.items()}
    print('indices',indices)
    top_labels = [indices[ind] for ind in index]
    print(top_labels)
    top_flowers=[cat_to_name[key] for key in top_labels]
    print(top_flowers)
    # TODO: Implement the code to predict the class from an image file
    return probablity,top_labels

def main():
    predict(args.image,args.top_k)
    print('finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to the test image')
    parser.add_argument('--top_k', type=int, help='Top classes to return', default=3)
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--labels', type=str, help='file for label names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='GPU/CPU')
    args = parser.parse_args()
    main()