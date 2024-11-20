
"""
DATA.ML.200
Week 4, title: MLP in PyTorch.

(The input is two class German Traffic Sign Recognition Benchmark (GTSRB)
dataset, a 64 × 64 RGB image that represents one of the two different
traffic signs.
The network structure is
• The 3 × 64 × 64-dimensional is flattened
• in the 1st layer there are 100 nodes.
• in the 2nd layer there are 100 nodes.
• in the 3rd (output) layer there are 2 nodes.
I defined the above network in my code. I used the class structure and
wrote the forward() function. And then I evaluated the model on the test set.)

Creator: Maral Nourimand
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


# Define the MLP model class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 100)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(100, 100)           # Hidden layer1 to hidden layer2
        self.fc3 = nn.Linear(100, 2)            # Hidden layer2 to output layer

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        #x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        #x = self.fc2(x)
        x = self.fc3(x)
        return x


def main():
    torch.manual_seed(42)

    # define data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),
        # Convert images to PyTorch tensors (automatically handles RGB channels)
    ])

    # Create dataset using ImageFolder
    dataset = datasets.ImageFolder(root="GTSRB_subset_2", transform=transform)

    # split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size,
                                                                 test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False)

    # model, loss function, and optimizer
    model = MLP()
    #summary(model, input_size=(3,64,64))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test set accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
