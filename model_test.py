from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from torch.autograd import Variable

PATH = 'datasets/imu'
Model_File = 'summary/testing_model/generator199.pth'
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

X_data = torch.Tensor(np.arange(1,64))
BATCH_SIZE = 768
slice_length = 64
control_point = 32
class own_dataset(Dataset):
    def __init__(self, path = 'datasets/imu', train=True):
        
        if train: self.path = glob(path+'/train/*/*gyroz*.txt')
        else: self.path = glob(path+'/test/*gyroz3_jyh*.txt')
        self.data_list = []
        self.term = 64 #data length
        for data in self.path:
            data = open(data)
            whole = data.read()
            whole = whole.split(',')
            whole = [float(0) if x=='NaN' or x=='NaN\n' else float(x) for x in whole]
            slices = []
            dummy_excepted =[]
            slice_term = 1#자르는 크기
            slices+=[whole[i:i+self.term] for i in range(1,len(whole)-self.term,slice_term)]
            for slice in slices:
                if np.std(slice)>50:
                    dummy_excepted.extend([slice])
            self.data_list.extend(dummy_excepted)
            
    def __getitem__(self,idx):
        
        data = self.data_list[idx]
        #Noise = np.random.randn(slice_length)#Add Normal Noise, State(64x1)
        data,scale = normalize(data)
        #critic_part = data[middle:]
        syn_data = data[:]
        syn_data[control_point:] = [float(0)]*(slice_length-control_point) # [data,data,data...,0,0,0]
        data = data#+Noise
        syn_data = syn_data#+Noise
        data = torch.tensor(data)
        syn_data = torch.tensor(syn_data)
        data = {'idx': idx,'data': data,'syn_data':syn_data,'scale':scale}#'critic_part':critic_part,
        return data
    
    
    def __len__(self):
        return len(self.data_list)


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = C
        # Output_dim = C (number of channels)
        self.To_latent = nn.Sequential(
            #State (cx64x1) {Cx[data_size]}
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (64x32x1)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (128x16x1)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
 
            # State (256x8x1)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=3, padding=0))
            

        self.To_syn = nn.Sequential(
            # Self.To_latent's output[2]
            # State (1x1x1)
            nn.ConvTranspose1d(in_channels=channels, out_channels=512, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),

            # State (512x4x1)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),

            # State (256x8x1)
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),

            # State (128x16x1)
            nn.ConvTranspose1d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1)
            )
            # output of To_real -> #State(1x2x1)

        self.output = nn.Tanh()

    def forward(self, x):
        middle = x.size()[1]//2
        x = x.unsqueeze(1)
        front = x[:,:,:middle]
        z = self.To_latent(x)
        z = z[:,:,1].unsqueeze(1)
        z = self.To_syn(z)
        z = self.output(z)
        return torch.cat([front,z],dim=2)
    
    def extractor(self, x):
        front = x[0:x.size()[0]/2]
        z = self.To_latent(x)
        x = self.To_syn(z[2])
        x = self.output(x)
        return torch.cat([front,x])

generator = Generator(1)
if torch.cuda.is_available():
    generator.cuda()
testData = own_dataset(path = PATH, train = False)
dataloader = DataLoader(dataset = testData,batch_size=BATCH_SIZE)

def normalize(X):
        # input X: list
        mx =max(X)
        mn =min(X)
        if mx-mn != 0: result = [(x - mn) / (mx-mn) for x in X]
        else: result = X
        return result,[mx,mn]

def unscaler(X,scale):
    # input X: list
        mx = scale[0]
        mn = scale[1]
        scale = mx-mn
        for i in range(0,X.size(0)):
            X[i] = X[i][:].type(torch.float64)*scale[i]+mn[i]
        return X


def load_model(G_model_path):
    generator.load_state_dict(torch.load(G_model_path),strict=False)
    print('Generator model loaded from {}.'.format(G_model_path))

load_model(Model_File)

Y_data1 = []
Y_data2 = []

for i, data in enumerate(dataloader):

    # Configure input
    scale = data['scale']
    syn_series = Variable(data['syn_data'].type(Tensor))
    real_series = unscaler(Variable(data['data'].type(Tensor)),scale)
    fake_series = unscaler(generator(syn_series).squeeze(1),scale)
    #if i<5:
    #    fake_series = unscaler((fake_series.detach()).cpu(),[data['scale'][0],data['scale'][1]])
    ### Linde Plot ###
    Y_data1.append(real_series.cpu())
    Y_data2.append(fake_series)
np.savetxt('summary//tested_sample/real.txt',Y_data1[0][3:])
np.savetxt('summary/tested_sample/fake.txt',((Y_data2[0][3:].cpu())).detach())

