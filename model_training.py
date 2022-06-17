from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F



PATH = 'datasets/imu'

slice_length = 64
control_point = 32
class own_dataset(Dataset):
    
    def __init__(self, path = PATH, train=True):
        
        if train: self.path = glob(path+'/train/jyh/*gyroz*.txt')
        else: self.path = glob(path+'/test/*gyroz3_jyh*.txt')
        self.data_list = []
        self.term = slice_length #data length
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
        Noise = np.random.randn(slice_length)#Add Normal Noise, State(64x1)
        data,scale = normalize(data+Noise)
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



def normalize(X):
    # input X: list
        mx = max(X)
        mn =min(X)
        result = [(x - mn) / (mx-mn) for x in X]
        return result,[mx,mn]
def scaler(X,scale):
    # input X: list
        mx = scale[0]
        mn = scale[1]
        scale = mx-mn
        return [x*scale+mn for x in X]



def compute_gradient_penalty(Dis, real_samples, fake_samples):
    #gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0),1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = Dis(interpolates.unsqueeze(1)).squeeze(1)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
            # State(512x4x1)
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
            # output of To_real -> #State(1x32x1)

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
    
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [128, 256, 512]
        # Input_dim = channels (batchx1x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Series (1x64x1)
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x32x1)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x1)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        
            # State (512x8x1)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            # State (512x4x1)
            # output of main module --> State
            )

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=4, padding=0))


    def forward(self, x):
        x=self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 512*4*1)

lr = 1e-3
b1 = 0.5
b2 = 0.999
batch_size = 256
n_epochs = 200
n_critic = 5 #number of training steps for discriminator per iter\

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(1)
discriminator = Discriminator(1)

data_shape = (1, 64) #(batch_size, 1, 64)

cuda = True if torch.cuda.is_available() else False



if cuda:
    generator.cuda()
    discriminator.cuda()

trainData = own_dataset()
testData = own_dataset(path = PATH, train = False)
# Configure data loader
dataloader = torch.utils.data.DataLoader(dataset = trainData,
    batch_size=batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# ----------
#  Training
# ----------

batches_done = 0


for epoch in range(n_epochs):
    
    epoch_Gloss = 0
    epoch_Dloss = 0
    
    for i, data in enumerate(dataloader):

        # Configure input
        #real_imgs = Variable(data['data'].type(Tensor))
        syn_series = Variable(data['syn_data'].type(Tensor))
        real_series = Variable(data['data'].type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Sample noise as generator input
        fake_series = generator(syn_series).squeeze(1)
        

        # Real
        real_validity = discriminator(real_series.unsqueeze(1))
        # Fake
        fake_validity = discriminator(fake_series.unsqueeze(1))
        # Gradient penalty

        gradient_penalty = compute_gradient_penalty(discriminator, real_series, fake_series)
        L2distance = F.pairwise_distance(real_series[:,32:],fake_series[:,32:])#.view(-1,1,1)

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty 
    
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------
            # Generate a batch of images
            fake_series = generator(syn_series).squeeze(1)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_series.unsqueeze(1))
           
            L2distance = F.pairwise_distance(real_series[:,32:],fake_series[:,32:])#.view(-1,1,1)
      
            point_loss = torch.mean(Variable(L2distance).type(torch.float32))
            
            g_loss = -torch.mean(fake_validity)-point_loss*0.05

            g_loss.backward()
            optimizer_G.step()
            
            epoch_Gloss+=abs(g_loss.item())
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            if epoch>n_epochs-5:
                torch.save(generator.state_dict(),'summary/testing_model/generator'+str(epoch)+'.pth')
        
    print("*****     [Epoch %d/%d] [epoch_Gloss: %f]     *****" %(epoch, n_epochs,epoch_Gloss))

    batches_done += n_critic

# ----------
#   Testing
# ----------
