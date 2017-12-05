import torch
import torchvision
from torch import nn
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from util import *
from model import *
from vnet import *
import time
import matplotlib.pyplot as plt

def trainer(model,train_loader,valid_loader=None,test_loader=None,ifcuda=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    timing = []
    print("Training ...")
    dataloader = train_loader
    num_iter = len(dataloader)
    train_loss = np.zeros(num_iter*num_epochs)
    i = 0
    for epoch in range(init_epochs,(init_epochs+num_epochs)):
        s = time.time()
        for input,target in dataloader:
            input,target = normalize(input,target)
            input = to_var(input,ifcuda)
            target = to_var(target,ifcuda)
            # ===================forward=====================
            output = model(input)
            loss = criterion(output, target)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[i] = loss.data[0]
            #print("loss:%.3f" % loss.data[0])            
            i = i + 1
        # ===================log========================
        # tmp = output.cpu().data
        # _,pic = denormalize(norm_targets=tmp)
        # print(np.min(pic.numpy()))
        e = time.time()
        cur_epoch_train_loss = np.mean(train_loss[(epoch-init_epochs)*num_iter:(epoch-init_epochs+1)*num_iter])
        print("Elapsed Time for one epoch: %.3f" % (e-s))
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, init_epochs+num_epochs, cur_epoch_train_loss))
        if epoch % 5 == 0:
            _,pic = denormalize(norm_targets=output.cpu().data)
            save_image(pic, './output/train/image_{}.png'.format(epoch))
        save_model(model, './model/model_{}.pth'.format(epoch))
        save(train_loss,'./output/result.pkl')
    return
    
    
train_lst = './im2latex_train.lst'
validate_lst = './im2latex_validate.lst'
test_lst = './im2latex_test.lst'
# x_dir='./data/handwritten_resized/'
# y_dir='./data/latex_resized/'
x_dir='./data/handwritten_padded/'
y_dir='./data/latex_padded/'
init_epochs = 0
num_epochs = 200
batch_size = 48
learning_rate = 1e-3
# handwritten_size = (41,285)
# latex_size = (40,253)
handwritten_size = (48,288)
latex_size = (48,288)
ifcuda = True

#train_dict,_ = build_dict([train_lst],'./train_dict_padded.pkl',x_dir,y_dir)
train_dict,_ = load('./train_dict_padded.pkl')

#validate_dict,_ = build_dict([validate_lst],'./validate_dict_padded.pkl',x_dir,y_dir)
#validate_dict,_ = load('./validate_dict_padded.pkl')

#test_dict,_ = build_dict([test_lst],'./test_dict_padded.pkl',x_dir,y_dir)
#test_dict,_ = load('./test_dict_padded.pkl')

train_size = len(train_dict)
# validate_size = len(train_dict)
# test_size = len(train_dict)



#showLoss()

train_inputs,train_targets = load_data(train_dict,handwritten_size,latex_size,x_dir,y_dir)
#validate_inputs,validate_targets = load_data(validate_dict,handwritten_size,latex_size,x_dir,y_dir)
#test_inputs,test_targets = load_data(test_dict,handwritten_size,latex_size,x_dir,y_dir)
#getStats(train_inputs,train_targets)

m_inputs,std_inputs,m_targets,std_targets = load('./stats.pkl')
print(m_inputs,std_inputs,m_targets,std_targets)

print(train_inputs.shape)

train_inputs = torch.from_numpy(train_inputs).type(torch.FloatTensor)
train_targets = torch.from_numpy(train_targets).type(torch.FloatTensor)
# validate_inputs = torch.from_numpy(validate_inputs).type(torch.FloatTensor)
# validate_targets = torch.from_numpy(validate_targets).type(torch.FloatTensor)
# test_inputs = torch.from_numpy(test_inputs).type(torch.FloatTensor)
# test_targets = torch.from_numpy(test_targets).type(torch.FloatTensor)

train = data_utils.TensorDataset(train_inputs,train_targets)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
# valid = data_utils.TensorDataset(validate_inputs,validate_targets)
# valid_loader = data_utils.DataLoader(valid, batch_size=batch_size, shuffle=True)
# test = data_utils.TensorDataset(test_inputs,test_targets)
# test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)


model = VNet().cuda() if ifcuda else VNet()
#model = load_model(FC0,ifcuda)
trainer(model,train_loader,ifcuda=ifcuda)

