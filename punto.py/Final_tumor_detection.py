#in python some libraries need to be installed
#Is necessary have in the same path the model (.pt) and images for test

from PIL import Image
import torch
from torchvision import transforms
#tranform input image with some parameters that used for TRAIN, TEST images
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
#upload images with PIL
image1 = Image.open('2si.jpg')
image1 = transform(image1)
#I have problems with image dimension, for some reason that I do not know, the model ask for 4 dimension image
image1 = image1.unsqueeze(0)

#we will check if gpu is available in order to upload the model and work in GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
print(device)
#upload the model obtained previously
model = torch.load('my_model.pt')
#moving the image to CPU
imagen = image1.to(device)
imagen_outputs = model(imagen)
#predicting image
_, preds = torch.max(imagen_outputs.data, 1)
print(preds)
#printing the results 
if preds[0] == 0:
      print("NO")
else:
  print("SI")