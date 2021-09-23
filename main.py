from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

os.system("cls")

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

idx_to_class = {0:'Negative', 1:'Positive'}

chosen_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(size=227),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nLoading model")

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
 
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(256, 128),
    nn.Linear(128, 2),
)

# model = torch.load("model.pt", map_location=torch.device('cpu'))
model = model.to(device)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
print("\nModel loaded")

def predict(model, test_image, print_class = False):
     # it uses the model to predict on test_image...
    transform = chosen_transforms['val']
     
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available(): # checks if we have a gpu available
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        # this computes the output of the model
        out = model(test_image_tensor)
        # this computes the probability of each classes.
        ps = torch.exp(out)
        # we choose the top class. That is, the class with highest probability
        topk, topclass = ps.topk(1, dim=1)
        class_name = idx_to_class[topclass.cpu().numpy()[0][0]]
        if print_class:
            print("Output class :  ", class_name)
    return class_name

def predict_on_crops(input_image, height=227, width=227, save_crops = False):
    im = cv2.imread(input_image)
    imgheight, imgwidth, channels = im.shape
    k=0
    output_image = np.zeros_like(im)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            a = im[i:i+height, j:j+width]
            ## discard image cropss that are not full size
            predicted_class = predict(model,Image.fromarray(a))
            ## save image
            file, ext = os.path.splitext(input_image)
            image_name = file.split('/')[-1]
            folder_name = 'out_' + image_name
            ## Put predicted class on the image
            if predicted_class == 'Positive':
                color = (0,0, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(a, predicted_class, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 1, cv2.LINE_AA) 
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0)
            ## Save crops
            if save_crops:
                if not os.path.exists(os.path.join('real_images', folder_name)):
                    os.makedirs(os.path.join('real_images', folder_name))
                filename = os.path.join('real_images', folder_name,'img_{}.png'.format(k))
                cv2.imwrite(filename, add_img)
            output_image[i:i+height, j:j+width,:] = add_img
            k+=1
    ## Save output image
    cv2.imwrite(os.path.join('real_images','predictions', folder_name+ '.jpg'), output_image)
    return output_image

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/live_demo', methods = ['POST', 'GET'])
def live_demo():
    if request.method == 'POST':

        choice_model = request.form.getlist('model')[0]

        predictions = {"negative1": False,
                       "negative2": False,
                       "negative3": False,
                       "negative4": False,
                       "negative5": False,
                       "positive1": False,
                       "positive2": False,
                       "positive3": False,
                       "positive4": False,
                       "positive5": False}

        choice_images = request.form.getlist('image')
        images_selection = []

        for i in range(len(choice_images)):
            if choice_images[i][0] == 'p':
                images_selection.append('static/data/Positive/'+choice_images[i][-1]+'.jpg')
            elif choice_images[i][0] == 'n':
                images_selection.append('static/data/Negative/'+choice_images[i][-1]+'.jpg')

        if choice_model == "cnn":
            for i in range(len(choice_images)):
                predictions[choice_images[i]] = True
                plt.figure()
                output_image = predict_on_crops(images_selection[i])
                prediction = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                plt.imshow(prediction)
                plt.imsave("static/data/Predictions/"+choice_images[i]+".jpg", prediction)
        else: print(choice_model)

        return render_template('live_demo.html', predictions=predictions)

    else:
        predictions = {"negative1": False,
                       "negative2": False,
                       "negative3": False,
                       "negative4": False,
                       "negative5": False,
                       "positive1": False,
                       "positive2": False,
                       "positive3": False,
                       "positive4": False,
                       "positive5": False}
        dir = 'static/data/Predictions'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        return render_template("live_demo.html", predictions=predictions)

@app.route('/documentation', methods = ['POST', 'GET'])
def documentation():
  return render_template('documentation.html')

if __name__ == '__main__':
  app.run(debug=True)
