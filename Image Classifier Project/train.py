import numpy as np

import torch
from torch import nn, optim
from torchvision import models
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse


def create_model(arch,class_to_index,hidden_units,learning_rate=0.001):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if arch=="vgg16" or arch=="vgg19":
        input_size = model.classifier[0].in_features
    elif arch=="densenet121":
        input_size = model.classifier.in_features

    print('input size=',input_size)
    print('learning rate=',learning_rate)
    print('hidden units=',hidden_units)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    print('working')
    criterion = nn.NLLLoss()
    model.class_to_index=class_to_index
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, optimizer, criterion


def validation(model, dataloaders,criterion):
    test_loss = 0
    accuracy = 0
    if args.gpu and torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    for images, labels in dataloaders['validloader']:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def save_checkpoint(arch, state_dict, class_to_idx,hidden_units):
    return {'arch': arch,
            'class_to_idx': class_to_idx,
            'state_dict': state_dict,
            'hidden_units':hidden_units}


def train_model(class_to_idx,dataloaders, arch,learning_rate,hidden_units,checkpoint):
    # train_data = dataloaders['trainloader']
    # validation_data = dataloaders['validloader']
    model, optimizer, criterion = create_model(arch,class_to_idx,hidden_units,learning_rate)
    epochs = args.epochs
    print_every = 40
    steps = 0
    if args.gpu and torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['trainloader']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            print("training")

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model,dataloaders, criterion)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss / len(dataloaders['validloader'])),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(dataloaders['validloader'])))

                running_loss = 0
    print('finished')
    if checkpoint:
        checkpoint_saved = save_checkpoint(args.arch, model.state_dict(), model.class_to_index,hidden_units)
        torch.save(checkpoint_saved, checkpoint)
        print('checkpoint saved')
    return model


def main():
    print(args.data_dir)
    print(args.arch)
    if args.data_dir:
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        validation_transforms = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        data_dir = args.data_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        image_datasets = {
            "train_data": datasets.ImageFolder(train_dir, transform=train_transforms),
            "valid_data": datasets.ImageFolder(valid_dir, transform=validation_transforms),
            "test_data": datasets.ImageFolder(test_dir, transform=test_transforms)
        }
        dataloaders = {
            "trainloader": torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=32, shuffle=True),
            "validloader": torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32, shuffle=True),
            "testloader": torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=16)
        }
        class_to_idx=image_datasets['train_data'].class_to_idx
        print(class_to_idx)

        train_model(class_to_idx, dataloaders, arch=args.arch,learning_rate=args.learning_rate,hidden_units=args.hidden_units,checkpoint=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help='path to folder of images')
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='chosen number of epochs')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='number of hidden layers')
    parser.add_argument('--save_dir', type=str, help='determine whether to save checkpoint')
    parser.add_argument('--gpu', action='store_true', help='GPU/CPU')
    parser.add_argument('--epochs', type=int, default=1,help='training rounds')
    args = parser.parse_args()

    main()
