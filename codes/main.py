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


def trainer(model,train_loader,valid_loader=None,test_loader=None,ifcuda=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    train_loss = []
    timing = []
    print("Training ...")
    dataloader = train_loader
    for epoch in range(num_epochs):
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
        # ===================log========================
        # tmp = output.cpu().data
        # _,pic = denormalize(norm_targets=tmp)
        # print(np.min(pic.numpy()))
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data[0]))
        train_loss.append(loss.data[0])
        if epoch % 5 == 0:
            _,pic = denormalize(norm_targets=output.cpu().data)
            save_image(pic, './output/train/image_{}.png'.format(epoch))
    save(train_loss,'result.pkl')
    torch.save(model.state_dict(), './model/model.pth')
    return
    
    
train_lst = './im2latex_train.lst'
validate_lst = './im2latex_validate.lst'
test_lst = './im2latex_test.lst'
x_dir='./data/handwritten_processed/'
y_dir='./data/latex_processed/'
num_epochs = 100
batch_size = 10
learning_rate = 1e-3
handwritten_size = (41,285)
latex_size = (40,253)
ifcuda = False

#train_dict,_ = build_dict([train_lst],'./train_dict.pkl',x_dir,y_dir)
train_dict,_ = load('./train_dict.pkl')
#validate_dict,_ = build_dict([validate_lst],'./validate_dict.pkl',x_dir,y_dir)
validate_dict,_ = load('./validate_dict.pkl')
#test_dict,_ = build_dict([test_lst],'./test_dict.pkl',x_dir,y_dir)
test_dict,_ = load('./test_dict.pkl')

train_size = len(train_dict)
validate_size = len(train_dict)
test_size = len(train_dict)


m_inputs,std_inputs,m_targets,std_targets = load('./stats.pkl')
print(m_inputs,std_inputs,m_targets,std_targets)

train_inputs,train_targets = load_data(train_dict,handwritten_size,latex_size,x_dir,y_dir)
#validate_inputs,validate_targets = load_data(validate_dict,handwritten_size,latex_size,x_dir,y_dir)
#test_inputs,test_targets = load_data(test_dict,handwritten_size,latex_size,x_dir,y_dir)
#getStats(train_inputs,train_targets)

print(train_inputs.shape)

train_inputs = torch.from_numpy(train_inputs).type(torch.FloatTensor)
train_targets = torch.from_numpy(train_targets).type(torch.FloatTensor)
train = data_utils.TensorDataset(train_inputs,train_targets)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
# valid = data_utils.TensorDataset(norm_validate_inputs,norm_validate_targets)
# valid_loader = data_utils.DataLoader(valid, batch_size=batch_size, shuffle=True)
# test = data_utils.TensorDataset(norm_test_inputs,norm_test_targets)
# test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)


model = fcn().cuda() if ifcuda else fcn()
trainer(model,train_loader)

