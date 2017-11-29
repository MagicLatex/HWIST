
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.misc import imread
import matplotlib
from PIL import Image, ImageChops
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn as nn
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

plt.switch_backend('agg')


with open('./im2latex_train.lst','rb') as file:
    match = file.readlines()

### Build a dictionary to match the handwirtten and latex pair 
train_match_dict = {e.decode('utf-8').split(' ')[0] + '.png':e.decode('utf-8').split(' ')[1] + '.png' for e in match[:5000]}

test_match_dict = {e.decode('utf-8').split(' ')[0] + '.png':e.decode('utf-8').split(' ')[1] + '.png' for e in match[5000:6000]}



# **trim the white border and resize**


def trim(im):

    bg = Image.new(im.mode, im.size, (255,255,255)) # white
    diff = ImageChops.difference(im, bg) # different between all white and image
    bbox = diff.getbbox() # detect black edge and get a box
    if bbox:
        return np.array(im.crop(bbox))



def pad_img(max_width, max_height, img):
    height, width = img.shape[0], img.shape[1]
    height_diff, width_diff = max_height - height, max_width - width
    npad = ((int(height_diff/2), height_diff - int(height_diff/2) ), (int(width_diff/2), width_diff - int(width_diff/2) ), (0, 0))
    img_padded = np.lib.pad(img, npad, 'constant', constant_values=255)
    return img_padded


### To binary


threshold = 150 

def to_bin(img, up):
    img_copy = np.copy(img)
    img_copy[img_copy <= threshold] = 0
    img_copy[img_copy > threshold] = up
    return img_copy



###  Create train data 


def trim_pad(train_match_dict, max_width, max_height):
    
    hw_imgs, print_imgs = [], []
    for i,hw in enumerate(list(train_match_dict.keys())):
        
        if i%1000 == 0:
            print ("load ",i," data")
        
        #load matched filenames
        pr = train_match_dict[hw]
        
        try:
            #print
            print_img = Image.open('./formula_images_processed/'+pr)
            print_img_trim = trim(print_img)
            print_img_pad = pad_img(max_width, max_height, print_img_trim)
            print_img_bin = to_bin(print_img_pad,up=1)


            #handwritten
            hw_img = Image.open('./IM2LATEX-100K-HANDWRITTEN/images/'+hw)
            hw_img_trim = trim(hw_img)
            hw_img_pad = pad_img(max_width, max_height, hw_img_trim)
            hw_img_bin = to_bin(hw_img_pad,up=1)
        
            
            #append the first channel : they are the same
            print_imgs.append(print_img_bin[:,:,0])
            hw_imgs.append(hw_img_bin[:,:,0])

        except:
            #image size exceed max size
            pass
        
        
    return print_imgs, hw_imgs



max_width, max_height = 300, 50


print_imgs, hw_imgs = trim_pad(train_match_dict, max_width, max_height)





# **array of 2d to 3d**

print_imgs = np.array(print_imgs, dtype = 'float')
hw_imgs = np.array(hw_imgs, dtype = 'float')


print_imgs_3d = np.concatenate([arr[np.newaxis] for arr in print_imgs])
print_imgs_3d = np.expand_dims(print_imgs_3d, axis=1)


hw_imgs_3d = np.concatenate([arr[np.newaxis] for arr in hw_imgs])
hw_imgs_3d = np.expand_dims(hw_imgs_3d, axis=1)


print_imgs_3d = print_imgs_3d[:160,:,:,:]
hw_imgs_3d = hw_imgs_3d[:160,:,:,:]


features = torch.from_numpy(hw_imgs_3d).type(torch.FloatTensor)
targets = torch.from_numpy(print_imgs_3d).type(torch.FloatTensor)


batch_size = 40 #整数倍个数


train = data_utils.TensorDataset(features, targets)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)


# ### Train DVE


epoch = 10
learning_rate = 0.001


# Encoder 
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 16 x max_height x max_weight
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 16 x max_height x max_weight
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 32 x max_height x max_weight
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 32 x max_height x max_weight
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x max_height x max_weight / 4
        )
#         self.layer2 = nn.Sequential(
#                         nn.Conv2d(64,128,3,padding=1),  # batch x 64 x 14 x 14
#                         nn.ReLU(),
#                         nn.BatchNorm2d(128),
#                         nn.Conv2d(128,128,3,padding=1),  # batch x 64 x 14 x 14
#                         nn.ReLU(),
#                         nn.BatchNorm2d(128),
#                         nn.MaxPool2d(2,2),
#                         nn.Conv2d(128,256,3,padding=1),  # batch x 64 x 7 x 7
#                         nn.ReLU()
#         )
        
                
    def forward(self,x):
        out = self.layer1(x)
#         print (out.size())
#         out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out
    
encoder = Encoder()


# In[46]:

# Decoder 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
#         self.layer1 = nn.Sequential(
#                         nn.ConvTranspose2d(256,128,3,2,1,1),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(128),
#                         nn.ConvTranspose2d(128,128,3,1,1),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(128),
#                         nn.ConvTranspose2d(128,64,3,1,1),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(64),
#                         nn.ConvTranspose2d(64,64,3,1,1),
#                         nn.ReLU(),
#                         nn.BatchNorm2d(64)
#         )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(batch_size,64,int(max_height/2),int(max_width/2))
#         out = self.layer1(out)
        out = self.layer2(out)
        return out

decoder = Decoder()


# In[47]:

# Check output of autoencoder
#ignore
# for image,label in train_loader:
#     image = Variable(image)
    
#     output = encoder(image)
#     output = decoder(output)
#     print(output.size())
    #break


# In[48]:

# Noise
#ignore = torch.rand(batch_size,1,28,28)


# In[49]:

# loss func and optimizer
# we compute reconstruction after decoder so use Mean Squared Error
# In order to use multi parameters with one optimizer,
# concat parameters after changing into list

parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.BCEWithLogitsLoss(weight=0.1*torch.ones(40,1,50,300)) #40 is batch size

# m = nn.Sigmoid()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)


# In[50]:

loss_arr = []
for i in range(1):
    for image,label in train_loader:
  
        optimizer.zero_grad()
        output = encoder(Variable(image))
        output = decoder(output)
        
        loss = loss_func(output,Variable(label))
        loss.backward()
        optimizer.step()       
                
#     torch.save([encoder,decoder],'./model/deno_autoencoder.pkl')
        print("i",loss.data.numpy()[0])
        loss_arr.append(loss.data.numpy()[0])



        
        
        