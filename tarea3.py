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
    self.output = nn.Linear(64 * 8 * 4 * 4, 10)

  def forward(self, input):
    output = self.main(input)
    
    return self.output(output.flatten(1))
    

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
  fixed_noise = torch.randn(batch_size, 100, 1, 1, device=torch.device(device))
  optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      batch_size = data[0].to(torch.device(device)).size()[0]
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # train with real
      netD.zero_grad()
      real_cpu = data[0].to(torch.device(device))
      label = data[1].to(torch.device(device))
      output = netD(real_cpu)
      errD_real = criterion(output, label)
      errD_real.backward()
      D_x = output.mean().item()

      # train with fake
      noise = torch.randn(batch_size, 100, 1, 1, device=torch.device(device))
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
  torch.save(netG.state_dict(), './netG_final.pth')
  torch.save(netD.state_dict(), './netD_final.pth')

def cnnTraining(cnn, dataloader, device='cpu', epochs=1, isNormalCNN=True):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.RMSprop(cnn.parameters(), lr=0.001, weight_decay=1e-6)
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      cnn.zero_grad()
      real_cpu = data[0].to(torch.device(device))
      label = data[1].to(torch.device(device))
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
  if isNormalCNN:
    torch.save(cnn.state_dict(), 'cnn_normal_final.pth')
  else:
    torch.save(cnn.state_dict(), 'cnn_fake_final.pth')
    
def generateFakeSet(netD, netG, device='cpu'):
  batch_size = 64
  images = torch.empty(1, 3, 64, 64)
  labels = torch.empty(1)
  while(labels.size()[0] < 1000):
    print(images.size()[0])
    #print(labels.size()[0])
    noise = torch.randn(batch_size, 100, 1, 1, device=torch.device(device))
    batch_images = netG(noise)
    _, batch_labels = torch.max(netD(batch_images.detach()), 1)
    #print(batch_images.size())
    for i in range(batch_labels.size()[0]):
      if batch_labels[i] != 10:
        images = torch.cat([images, batch_images[i].unsqueeze(0)])
        labels = torch.cat([labels, batch_labels[i].unsqueeze(0).float()])
  return [images[1:], labels[1:]]

if __name__ == "__main__":
  device = 'cuda:0'
  epochs = 1
  trainloader, testloader = loadData()
  #discriminador y generador
  netD = Discriminator()
  netG = Generator()
  netD.to(torch.device(device))
  netG.to(torch.device(device))
  shouldTrainGan = False
  try:
    netD.load_state_dict(torch.load('./netD_final.pth'))
    print('pesos de discriminador encontrados')
  except:
    print('pesos de discriminador no encontrados, inicializando pesos')
    netD.apply(weights_init)
    shouldTrainGan = True
  try:
    netG.load_state_dict(torch.load('./netG_final.pth'))
    print('pesos de generador encontrados')
  except:
    print('pesos de generador no encontrados, inicializando pesos')
    netG.apply(weights_init)
    shouldTrainGan = True
  #entrenamiento
  if shouldTrainGan:
    ganTraining(netD, netG, trainloader, device, epochs)

  #cnn normal
  normalCNN = CNN()
  try:
    normalCNN.load_state_dict(torch.load('./cnn_normal_final.pth'))
    print('pesos de cnn normal encontrados')
  except:
    print('pesos de cnn normal no encontrados, inicializando pesos')
    normalCNN.apply(weights_init)
    device = 'cuda:0'
    normalCNN.to(torch.device(device))
    cnnTraining(normalCNN, trainloader, device, epochs, True)
  #cnn fake
  fakeCNN = CNN()
  try:
    fakeCNN.load_state_dict(torch.load('./cnn_fake_final.pth'))
    print('pesos de cnn fake encontrados')
  except:
    print('pesos de cnn fake no encontrados, inicializando pesos')
    #generación de set falso
    device = 'cpu'
    netD.to(torch.device(device))
    netG.to(torch.device(device))
    fakeSet = generateFakeSet(netD, netG)
    print(fakeSet[0].size(), fakeSet[1].size())
    #fakeSet = torch.stack(fakeSet)
    fakeCNN.apply(weights_init)
    device = 'cuda:0'
    fakeCNN = CNN()
    fakeCNN.to(torch.device(device))
    print(fakeSet[1])
    #fakeloader = torch.utils.data.TensorDataset(fakeSet)
    #cnnTraining(fakeCNN, fakeloader, device, epochs, False)
    imshow(torchvision.utils.make_grid(torch.tensor(fakeSet[0][0])))
