# hyperparam.s
lr = 1.0
bsz = 100
clip = 0.5
log_interval = 10
sample_iters = 6
epochs = 2

ame_list = []
vae_list = []
loss_list = []
data_list = []

criterion = nn.BCELoss()

# get data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, 
                                          shuffle=False, num_workers=2)

for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs = torch.round(inputs)
    data_list.append(inputs.view(1,bsz,784))
print(data_list[0].size())
data_list = torch.cat(data_list)

test_list = []
for i, data in enumerate(testloader):
    inputs, labels = data
    inputs = torch.round(inputs)
    test_list.append(inputs.view(1,bsz,784))
print(test_list[0].size())
test_list = torch.cat(test_list)

ninp = data_list.size(-1)
mid_size = 400
latent_size = 100

with open('avme.pt', 'rb') as f:
    ame_dec_v = torch.load(f)
    
with open('avme_enc.pt', 'rb') as f:
    ame_enc_v = torch.load(f)
