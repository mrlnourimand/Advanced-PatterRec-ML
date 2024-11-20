"""
DATA.ML.200
Week 7, title: Do-It-Yourself RNN, part 2: Letter prediction MLP.

(The MLP network takes a one-hot-encoded letter as input and
predicts the next letter as one-hot-encoded output:xt+1 = MLP(xt)
It uses the ’abcde.txt’ text as training data (I used my one-hot-encoding
function).
The training function forms the training input matrix X and output matrix
Y so that always the next letter is in the output matrix.
An evaluation function is also written that randomly picks a starting letter
from X, and then generates a sentence of 50 characters using the trained
network.)

Creator: Maral Nourimand
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def read_text_file(filename):
    """
    this function takes the file name as input and returns the one-hot-encoded
    data matrix, vocabulary size and ’char to int’ and ’int to char’ Python
    dictionaries to convert integers to characters and characters to integers.
    """
    # Read text from file
    with open(filename, 'r') as file:
        text = file.read()

    # Create character-to-integer and integer-to-character dictionaries
    unique_chars = sorted(set(text))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for char, i in char_to_int.items()}

    # Print total number of characters and vocabulary size
    total_chars = len(text)
    vocab_size = len(unique_chars)
    print(f"Total number of characters: {total_chars}")
    print(f"Vocabulary size: {vocab_size}")

    # Convert characters to integers
    int_text = [char_to_int[char] for char in text]

    # Convert integers to one-hot encoding
    one_hot_encoded = np.zeros((total_chars, vocab_size), dtype=np.float32)
    for i, char_index in enumerate(int_text):
        one_hot_encoded[i, char_index] = 1

    return one_hot_encoded, vocab_size, char_to_int, int_to_char


# Define the MLP model class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Function to generate training data
def generate_training_data(data_matrix):
    X = data_matrix[:-1]
    Y = data_matrix[1:]
    return X, Y


# Function to train the model
def train_model(model, X, Y, epochs, criterion, optimizer):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X)  # Forward pass
        # loss = criterion(outputs, torch.argmax(Y, dim=1))  # Compute the loss
        loss = criterion(outputs, Y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6}")


# Function to evaluate the model and generate a sequence
def evaluate_model(model, int_to_char, char_to_int, input_letter):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        start_letter = int_to_char[np.argmax(input_letter)]
        generated_sequence = [start_letter]
        for _ in range(50-1):
            input_tensor = torch.Tensor(input_letter)  # convert to PyTorch tensor
            output_probs = model(input_tensor)
            pred_index = torch.argmax(output_probs).item()
            generated_letter = int_to_char[pred_index]
            generated_sequence.append(generated_letter)

            # prepare next letter to predict
            input_letter = torch.zeros_like(input_tensor)
            input_letter[0, pred_index] = 1
        generated_sequence = ''.join(generated_sequence)
        print(f"Generated sequence by model:\n{generated_sequence}")


def main():
    # Load the one-hot encoded data and initialize the model
    data_matrix, vocab_size, char_to_int, int_to_char = read_text_file(
        'abcde.txt')
    input_size = vocab_size
    hidden_size = 128
    output_size = vocab_size
    model = MLP(input_size, hidden_size, output_size)

    # Generate training data
    X, Y = generate_training_data(data_matrix)

    # Define loss function and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    print("Training the model:")
    train_model(model, torch.Tensor(X), torch.Tensor(Y), epochs=10,
                criterion=criterion, optimizer=optimizer)

    start_index = np.random.randint(0, len(data_matrix) - 1)
    input_letter = data_matrix[start_index].reshape(1, -1)

    # Print the randomly chosen start letter
    max_index = np.argmax(input_letter)
    start_letter = [int_to_char[max_index]]
    print(20 * "#")
    print(f"Start letter: {start_letter}")

    # print True values from the txt file
    true_values = start_letter[0]
    for i in range(start_index, start_index + 49):
        true_values += int_to_char[np.argmax(Y[i])]
    print(20 * "#")
    print("True Values from the txt file:")
    print(true_values)

    # evaluate the model and generate a sequence
    print(20 * "#")
    evaluate_model(model, int_to_char, char_to_int, input_letter)


if __name__ == "__main__":
    main()
