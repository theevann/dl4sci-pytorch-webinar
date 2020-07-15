### Examples in the MNIST dataset

n_images = len(train_dataset)
print(f"There are {n_images} images in the train dataset.")

first_datapoint = train_dataset[0]
print(f"Data points are a {type(first_datapoint)} of length {len(first_datapoint)}.")
print("The entries are the input and the correct class (the digit.)")

X, y = first_datapoint
print(f"The images are {type(X)}'s of shape {X.shape}.")


### Multi-threading in dataloaders

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
)


### Neural network implementation

import torch.nn.functional as F  # provides some helper functions like Relu's, Sigmoids, Tanh, etc.


class MyNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.linear_1 = torch.nn.Linear(input_size, 75)
        self.linear_2 = torch.nn.Linear(75, 50)
        self.linear_3 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        out = self.linear_3(out)
        return out

