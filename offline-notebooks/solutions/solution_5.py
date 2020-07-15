from torchvision.transforms.functional import to_tensor, to_pil_image

image_tensor = to_tensor(image)
input = image_tensor.unsqueeze(0)
kernel = torch.Tensor([-1,0,1]).expand(1,3,3,3)

out = torch.nn.functional.conv2d(input, kernel)
norm_out = (out - out.min()) / (out.max() - out.min())


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv_net = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('Relu1', nn.ReLU()),
            
            ('S2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('C3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('Relu3', nn.ReLU()),
            
            ('S4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('C5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('Relu5', nn.ReLU()),
        ]))
        
        self.fully_connected = nn.Sequential(OrderedDict([
            ('F6', nn.Linear(120, 84)),
            ('Relu6', nn.ReLU()),
            ('F7', nn.Linear(84, 10)),
            ('LogSoftmax', nn.LogSoftmax(dim=-1))
        ]))
        
        
    def forward(self, imgs):
        output = self.conv_net(imgs)
        output = output.view(imgs.shape[0], -1)  # imgs.shape[0] is the batch_size
        output = self.fully_connected(output)
        return output        