import matplotlib
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import random
from model import *
from util import *
import time

train_output_path = './output/train'
validate_output_path = './output/train'
test_output_path = './output/train'
#train_dict,_ = build_dict([train_lst],'./train_dict_padded.pkl',x_dir,y_dir)
train_dict,_ = load('./train_dict_padded.pkl')
#validate_dict,_ = build_dict([validate_lst],'./validate_dict_padded.pkl',x_dir,y_dir)
validate_dict,_ = load('./validate_dict_padded.pkl')
#test_dict,_ = build_dict([test_lst],'./test_dict_padded.pkl',x_dir,y_dir)
test_dict,_ = load('./test_dict_padded.pkl')
#all_dict,_ = build_dict([train_lst,validate_lst,test_lst],'./all_dict_padded.pkl',x_dir,y_dir)
all_dict,_ = load('./all_dict_padded.pkl')
   
def getLoss(modelclass,dataloader,ifcuda,ifeval,setname):
    criterion = nn.MSELoss()
    num_iter = len(dataloader)
    loss_iter = np.zeros(num_iter*num_epochs)
    loss_epoch = np.zeros(num_epochs)
    i = 0
    j = 0
    print("Get Loss ...")
    for epoch in range(init_epochs,(init_epochs+num_epochs)):
        model = load_model(modelclass,ifcuda,'./model/model_{}.pth'.format(epoch))
        model = model.eval() if ifeval else model
        s = time.time()
        for input,target in dataloader:
            input,target = normalize(input,target)
            input = to_var(input,ifcuda)
            target = to_var(target,ifcuda)
            output = model(input)
            loss = criterion(output, target)
            loss_iter[i] = loss.data[0]
            i = i + 1
        e = time.time()
        loss_epoch[j] = np.mean(loss_iter[(epoch-init_epochs)*num_iter:(epoch-init_epochs+1)*num_iter])
        print("Elapsed Time for one epoch: %.3f" % (e-s))
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, init_epochs+num_epochs, loss_epoch[j]))
        if epoch % 5 == 0:
            _,pic = denormalize(norm_targets=output.cpu().data)
            save_image(pic, './output/'+setname+'/image_{}.png'.format(epoch))
        j = j + 1
        save([loss_iter,loss_epoch],'./output/'+setname+'/loss.pkl')
    return loss_iter,loss_epoch

def getLoss_all(modelclass):
    train_inputs,train_targets = load_data(train_dict,handwritten_size,latex_size,x_dir,y_dir)
    validate_inputs,validate_targets = load_data(validate_dict,handwritten_size,latex_size,x_dir,y_dir)
    test_inputs,test_targets = load_data(test_dict,handwritten_size,latex_size,x_dir,y_dir)

    train_inputs = torch.from_numpy(train_inputs).type(torch.FloatTensor)
    train_targets = torch.from_numpy(train_targets).type(torch.FloatTensor)
    validate_inputs = torch.from_numpy(validate_inputs).type(torch.FloatTensor)
    validate_targets = torch.from_numpy(validate_targets).type(torch.FloatTensor)
    test_inputs = torch.from_numpy(test_inputs).type(torch.FloatTensor)
    test_targets = torch.from_numpy(test_targets).type(torch.FloatTensor)

    train = data_utils.TensorDataset(train_inputs,train_targets)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    validate = data_utils.TensorDataset(validate_inputs,validate_targets)
    validate_loader = data_utils.DataLoader(validate, batch_size=batch_size, shuffle=True)
    test = data_utils.TensorDataset(test_inputs,test_targets)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    train_loss_iter,train_loss_epoch = getLoss(modelclass,train_loader,ifcuda,True,'train')
    validate_loss_iter,validate_loss_epoch = getLoss(modelclass,validate_loader,ifcuda,True,'validate')
    test_loss_iter,test_loss_epoch = getLoss(modelclass,test_loader,ifcuda,True,'test')
    
def getOptimalModel(validation_loss='./output/validate/loss.pkl'):
    loss_iter,loss_epoch = load(validation_loss)
    loss_epoch = np.convolve(loss_epoch, np.ones((5,))/5, mode='valid')
    return np.argmin(loss_epoch)

def compare_plot(input, target, output, output_overfit):
    n = len(input)
    fontsize = 5
    fig, axarr = plt.subplots(4,n, figsize = (30,30), dpi=300, gridspec_kw = {'wspace':0, 'hspace':0})   
    for i, ax in enumerate(fig.axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for i in range(n):
            axarr[0,i].imshow(input[i],cmap='gray')
            axarr[0,i].set_title(hw_name[i],fontsize=fontsize)            
            axarr[1,i].imshow(target[i],cmap='gray')
            axarr[2,i].imshow(output[i],cmap="gray")
            axarr[3,i].imshow(output_overfit[i],cmap="gray")
            if(i==2):
                axarr[0,i].set_xlabel('handwritten',fontsize=fontsize)
                axarr[1,i].set_xlabel('latex',fontsize=fontsize)
                axarr[2,i].set_xlabel('best validated model',fontsize=fontsize)
                axarr[3,i].set_xlabel('overfit model',fontsize=fontsize)
   
    #print ("# pixel outside 0,1: ", np.sum(output>1) + np.sum(output<0) )
    plt.show()


def showExample():
    overfit_model = 149
    print('Optimal model: {}'.format(optimal_model))
    print('Overfit model: {}'.format(overfit_model))
    model = load_model(modelclass,ifcuda,'./model/model_{}.pth'.format(optimal_model))
    model = model.eval() if ifeval else model
    overfit_model = load_model(modelclass,ifcuda,'./model/model_{}.pth'.format(overfit_model))
    overfit_model = overfit_model.eval() if ifeval else overfit_model

    m_inputs,std_inputs,m_targets,std_targets = load('./stats.pkl')
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(m_inputs,std_inputs),])
    transform_toPIL = transforms.ToPILImage()
    
    hw_imgs = []
    latex_imgs = []
    output_imgs = []
    output_overfit_imgs = []
    for i in range(n):
        hw = Image.open('./data/handwritten_padded/' + hw_name[i])
        latex = Image.open('./data/latex_padded/' + test_dict[hw_name[i]])
        hw_numpy = np.array(trim(hw))
        latex_numpy = np.array(trim(latex))
        hw = transform(hw).unsqueeze(1)
        latex = transform(latex).unsqueeze(1)
        input = to_var(hw,ifcuda)
        output = model(input)
        output_overfit = overfit_model(input)
        _,output = denormalize(norm_targets=output.cpu().data)
        _,output_overfit = denormalize(norm_targets=output_overfit.cpu().data)
        output_img = transform_toPIL(output[0,:,:,:])
        output_overfit_img = transform_toPIL(output_overfit[0,:,:,:])
        output_numpy = np.array(trim(output_img))
        output_overfit_numpy = np.array(trim(output_overfit_img))
        
        hw_imgs.append(hw_numpy)
        latex_imgs.append(latex_numpy)
        output_imgs.append(output_numpy)
        output_overfit_imgs.append(output_overfit_numpy)
        
    compare_plot(hw_imgs, latex_imgs, output_imgs, output_overfit_imgs)
    
train_lst = './im2latex_train.lst'
validate_lst = './im2latex_validate.lst'
test_lst = './im2latex_test.lst'
x_dir='./data/handwritten_padded/'
y_dir='./data/latex_padded/'

init_epochs = 0
num_epochs = 150
batch_size = 48
ifcuda = True
handwritten_size = (48,288)
latex_size = (48,288)
num_loss = 60 
modelclass = FC1
ifeval = True

#getLoss_all(modelclass)
#showLoss(num_loss,'train')
#showLoss(num_loss,'validate')
#showLoss(num_loss,'test')
optimal_model = getOptimalModel()
showLoss_complete(num_loss,optimal_model)
hw_name = ['14185.png','32787.png','65561.png','81940.png','81949.png']
n = len(hw_name)

showExample()


#n = 5
#random = np.array(random.sample(range(0,len(test_dict.keys())), n))
#hw_name = np.array(list(test_dict.keys()))[random]

