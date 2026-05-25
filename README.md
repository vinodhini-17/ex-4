# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

<img width="407" height="109" alt="image" src="https://github.com/user-attachments/assets/007e567b-5ada-41b9-8caf-2170188befff" />



## Neural Network Model

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d43e5227-d911-4186-b071-3b68b96e5d3f" />

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transform

### STEP 2: 

Load training and testing datasets using ImageFolder.


### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (CrossEntropyLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 


Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name:vinodhinin k
### Register Number: 212223230245

```python

import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchsummary import summary

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

!unzip -qq ./chip_data.zip -d data
dataset_path='./data/dataset'
train_dataset=datasets.ImageFolder(root=f"{dataset_path}/train",transform=transform)
test_dataset=datasets.ImageFolder(root=f"{dataset_path}/test",transform=transform)

def show_sample_images(dataset,num_images=5):
  fig,axes=plt.subplots(1,num_images,figsize=(5,5))
  for i in range(num_images):
    image,label=dataset[i]
    image=image.permute(1,2,0)
    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label])
    axes[i].axis("off")
  plt.show()
print("Number of training samples:",len(train_dataset))
first_image,label=train_dataset[0]
print("Image shape:",first_image.shape)

print("Number of testing samples:",len(test_dataset))
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)
device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
summary(model,input_size=(3,224,224))

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
summary(model,input_size=(3,224,224))

for param in model.features.parameters():
  param.requires_grad=False

criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

def train_model(model,train_loader,test_loader,num_epochs=100):
  train_losses=[]
  val_losses=[]
  for epoch in range(num_epochs):
    running_loss=0.0
    for images,labels in train_loader:
      images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels.unsqueeze(1).float())
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    train_losses.append(running_loss/len(train_loader))

    model.eval()
    val_loss=0.0
    with t.no_grad():
      for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        val_loss+=loss.item()
    val_losses.append(val_loss/len(test_loader))
    model.train()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
  plt.figure(figsize=(8,6))
  plt.plot(range(1,num_epochs+1),train_losses,label="Train Loss",marker="o")
  plt.plot(range(1,num_epochs+1),val_losses,label="Validation Loss",marker="s")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training and validation Loss")
  plt.legend()
  plt.show()

device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
train_model(model,train_loader,test_loader)

def test_model(model,test_loader):
  model.eval()
  correct=0
  total=0
  all_preds=[]
  all_labels=[]

  with t.no_grad():
    for images,labels in test_loader:
      images=images.to(device)
      labels=labels.float().unsqueeze(1).to(device)

      outputs=model(images)
      probs=t.sigmoid(outputs)
      predicted=(probs > 0.5).int()
      total+=labels.size(0)
      correct+=(predicted==labels.int()).sum().item()

      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy().astype(int))
  accuracy=correct/total
  print(f"Test Accuracy: {accuracy:.4f}")

  class_names=['Negative','Positive']
  cm=confusion_matrix(all_labels,all_preds)
  plt.figure(figsize=(6,5))
  sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()


  print("Classification Report :")
  print(classification_report(all_labels,all_preds,target_names=class_names))
test_model(model,test_loader)

def predict_image(model,image_index,dataset):
  model.eval()
  image,label=dataset[image_index]
  with t.no_grad():
    image_tensor = image.unsqueeze(0).to(device)
    output=model(image_tensor)
    _,predicted=t.max(output,1)
  class_names=dataset.classes

  image_to_display=transforms.ToPILImage()(image)

  plt.figure(figsize=(4,4))
  plt.imshow(image_to_display)
  plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
  plt.axis('off')
  plt.show()
predict_image(model,image_index=55,dataset=test_dataset)
predict_image(model,image_index=25,dataset=test_dataset)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="762" height="516" alt="image" src="https://github.com/user-attachments/assets/de0194e5-4f73-4961-a301-beb8aca5c9a5" />


## Confusion Matrix

<img width="677" height="607" alt="image" src="https://github.com/user-attachments/assets/618f3b1b-a70d-4104-90c1-f729f62d9786" />


## Classification Report

<img width="551" height="214" alt="image" src="https://github.com/user-attachments/assets/6934192f-e115-467b-980a-f5355f77f2d6" />


### New Sample Data Prediction

<img width="442" height="463" alt="image" src="https://github.com/user-attachments/assets/d3ca1322-7771-46d1-91b2-3b7a31cb225a" />



<img width="454" height="483" alt="image" src="https://github.com/user-attachments/assets/a246197f-ebba-402d-be31-ffd50ccdbbc7" />



## RESULT
Include your result here
