import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.detach().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def loadData(
    imageSize = 64,
    root = './data',
    batch_size = 64,
    num_workers = 2):
  print('Cargando el dataset CIFAR10')
  trainset = torchvision.datasets.CIFAR10(
  root=root, 
  train=True,
  download=True, 
  transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]))
  trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers)

  testset = torchvision.datasets.CIFAR10(
    root=root, 
    train=False,
    download=True, 
    transform=transforms.Compose([
      transforms.Resize(imageSize),
      transforms.CenterCrop(imageSize),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
  testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=num_workers)
  return [trainloader, testloader]

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(100, 32 * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(32 * 8),
      nn.ReLU(True),
      # state size. (32*8) x 4 x 4
      nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(32 * 4),
      nn.ReLU(True),
      # state size. (32*4) x 8 x 8
      nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(32 * 2),
      nn.ReLU(True),
      # state size. (32*2) x 16 x 16
      nn.ConvTranspose2d(32 * 2,     32, 4, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      # state size. (32) x 32 x 32
      nn.ConvTranspose2d(    32,      3, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (3) x 32 x 32
    )

  def forward(self, input):
    output = self.main(input)
    return output

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      # input is (3) x 64 x 64
      nn.Conv2d(3, 64, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (64) x 32 x 32
      nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (64*2) x 16 x 16
      nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (64*4) x 8 x 8
      nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (64*8) x 4 x 4
    )
    self.output = nn.Linear(64 * 8 * 4 * 4, 11)

  def forward(self, input):
    output = self.main(input)
    
    return self.output(output.flatten(1))

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv2d_1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.batch_normalization_1 = nn.BatchNorm2d(64)
    self.conv2d_2 = nn.Conv2d(64, 64, 3, 1, 1)
    self.batch_normalization_2 = nn.BatchNorm2d(64)
    self.max_pooling2d_1 = nn.MaxPool2d(4)
    self.dropout_1 = nn.Dropout(0.2)
    self.conv2d_3 = nn.Conv2d(64, 64 * 2, 3, 1, 1 )
    self.batch_normalization_3 = nn.BatchNorm2d(64 * 2)
    self.conv2d_4 = nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1 )
    self.batch_normalization_4 = nn.BatchNorm2d(64 * 2)
    self.max_pooling2d_2 = nn.MaxPool2d(4)
    self.dropout_2 = nn.Dropout(0.3)
    self.conv2d_5 = nn.Conv2d(64 * 2, 64 * 4, 3, 1, 1 )
    self.batch_normalization_5 = nn.BatchNorm2d(64 * 4)
    self.conv2d_6 = nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1 )
    self.batch_normalization_6 = nn.BatchNorm2d(64 * 4)
    self.max_pooling2d_3 = nn.MaxPool2d(4)
    self.dropout_3 = nn.Dropout(0.4)
    self.output = nn.Linear(64 * 4, 10)

  def forward(self, x):
    x = self.batch_normalization_1(F.elu(self.conv2d_1(x)))
    x = self.batch_normalization_2(F.elu(self.conv2d_2(x)))
    x = self.dropout_1(self.max_pooling2d_1(x))
    x = self.batch_normalization_3(F.elu(self.conv2d_3(x)))
    x = self.batch_normalization_4(F.elu(self.conv2d_4(x)))
    x = self.dropout_2(self.max_pooling2d_2(x))
    x = self.batch_normalization_5(F.elu(self.conv2d_5(x)))
    x = self.batch_normalization_6(F.elu(self.conv2d_6(x)))
    x = self.dropout_3(self.max_pooling2d_3(x))
    x = self.output(x.flatten(1))
    return x

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

def ganTraining(netD, netG, dataloader, device='cpu', epochs=1):
  criterion = nn.CrossEntropyLoss()
  batch_size = 64
  fixed_noise = torch.randn(batch_size, 100, 1, 1, device=device)
  optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # train with real
      netD.zero_grad()
      real_cpu = data[0].to(device)
      label = data[1].to(device)
      output = netD(real_cpu)
      errD_real = criterion(output, label)
      errD_real.backward()
      D_x = output.mean().item()

      # train with fake
      noise = torch.randn(batch_size, 100, 1, 1, device=device)
      fake = netG(noise)
      label.fill_(10)
      output = netD(fake.detach())
      errD_fake = criterion(output, label)
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ###########################
      netG.zero_grad()
      label.fill_(i%10)  # fake labels are real for generator cost
      output = netD(fake)
      errG = criterion(output, label)
      errG.backward()
      D_G_z2 = output.mean().item()
      optimizerG.step()

      print(
        '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
        % (epoch, epochs, i, len(dataloader),
        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      if i % 100 == 0:
        vutils.save_image(
          real_cpu,
          '%s/real_samples.png' % '.',
          normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
          '%s/fake_samples_epoch_%03d.png' % ('.', epoch),
          normalize=True)
      # do checkpointing
      torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('.', epoch))
      torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('.', epoch))

def cnnTraining(cnn, dataloader, device='cpu', epochs=1):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.RMSprop(cnn.parameters(), lr=0.001, weight_decay=1e-6)
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      cnn.zero_grad()
      real_cpu = data[0].to(device)
      label = data[1].to(device)
      output = cnn(real_cpu)
      err = criterion(output, label)
      err.backward()
      CNN_x = output.mean().item()
      optimizer.step()

      print(
        '[%d/%d][%d/%d] Loss: %.4f CNN(x): %.4f'
        % (epoch, epochs, i, len(dataloader),
        err.item(), CNN_x))
      # do checkpointing
      torch.save(cnn.state_dict(), '%s/cnn_epoch_%d.pth' % ('.', epoch))
    
def generateFakeSet(netD, netG, device='cpu'):
  batch_size = 64
  images = torch.empty(1, 3, 64, 64)
  labels = torch.empty(1)
  while(labels.size()[0] < 1000):
    noise = torch.randn(batch_size, 100, 1, 1, device=device)
    batch_images = netG(noise)
    _, batch_labels = torch.max(netD(images.detach()), 1)
    for i in (batch_labels==10).nonzero():
      batch_images = torch.cat([batch_images[0:i], batch_images[i+1:]])
    batch_labels = batch_labels[batch_labels!=10]
    images = torch.cat([images, batch_images])
    labels = torch.cat([labels, batch_labels.float()])
  return [images[1:], labels[1:]]

if __name__ == "__main__":
  device = 'cpu'
  epochs = 1
  trainloader, testloader = loadData()
  netD = Discriminator()
  netD.apply(weights_init)
  netG = Generator()
  netG.apply(weights_init)
  normalCNN = CNN()
  #ganTraining(netD, netG, trainloader, device, epochs)
  #cnnTraining(normalCNN, trainloader, device, epochs)
  fakeSet = generateFakeSet(netD, netG)
  imshow(torchvision.utils.make_grid(torch.tensor(fakeSet[0][0])))
