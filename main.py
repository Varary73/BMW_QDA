from flask import Flask, render_template, request
import os
import torch
from torchvision import transforms
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

print("\nLoading model")
model = torch.load("model.pth", map_location=torch.device('cpu'))
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

@app.route('/page1', methods = ['POST', 'GET'])
def page1():
  if request.method == 'POST':
    choice = request.form.getlist('choice')[0]
    number = request.form.get('number', type=int)
    # label = request.form['label']
    # number = request.form['number']
    plt.figure()
    output_image = predict_on_crops('static/data/'+choice+'/'+str(number)+'.jpg')
    prediction = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    plt.imshow(prediction)
    plt.imsave("static/data/prediction.jpg", prediction)
    return render_template('page1.html', prediction='data/prediction.jpg', input='data/'+choice+'/'+str(number)+'.jpg', original_input=number)
  else:
    return render_template("page1.html")

@app.route('/page2', methods = ['POST', 'GET'])
def page2():
  return render_template('page2.html')

if __name__ == '__main__':
  app.run(debug=True)
