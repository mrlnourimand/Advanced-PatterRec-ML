"""
DATA.ML.200
Week 7, title: Do-It-Yourself RNN, part 3: Letter prediction MLP V2.

(The new MLP network takes two one-hot-encoded letters as input
and predicts the next letter as one-hot-encoded output: xt+1 = MLP(xt, xt−1)
The training function forms training input matrix X and output matrix
Y so that always the next letter is in the output matrix. Inputs are always two
previous letters. My own onehot-encoding function is used from part 1.
An evaluation function is also written that randomly picks a starting letter
from X, and then generates a sentence of 50 characters using the trained network.
The code prints the training loss after each epoch and then a generated
sequence of length 50.)

Creator: Maral Nourimand
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def read_text_file(filename):
    """
    this function takes the file name as input and returns the one-hot-encoded
    data matrix, vocabulary size and ’char to int’ and ’int to char’ Python
    dictionaries to convert integers to characters and characters to integers.
    """
    with open(filename, 'r') as file:
        text = file.read()

    # character-to-integer and integer-to-character dictionaries
    unique_chars = sorted(set(text))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for char, i in char_to_int.items()}

    total_chars = len(text)
    vocab_size = len(unique_chars)
    print(f"Total number of characters: {total_chars}")
    print(f"Vocabulary size: {vocab_size}")

    # convert characters to integers
    int_text = [char_to_int[char] for char in text]

    # convert integers to one-hot encoding
    one_hot_encoded = np.zeros((total_chars, vocab_size), dtype=np.float32)
    for i, char_index in enumerate(int_text):
        one_hot_encoded[i, char_index] = 1

    return one_hot_encoded, vocab_size, char_to_int, int_to_char


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


# Function to prepare training data
def prepare_training_data(one_hot_encoded_text):
    X = []
    Y = []
    for i in range(len(one_hot_encoded_text) - 2):
        X.append(np.concatenate(
            (one_hot_encoded_text[i], one_hot_encoded_text[i + 1])))
        Y.append(one_hot_encoded_text[i + 2])
    return np.array(X), np.array(Y)


# Training function
def train_model(X, Y, model, criterion, optimizer, epochs=10, batch_size=40):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(X), batch_size):
            inputs = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            labels = torch.tensor(Y[i:i + batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Training loss: {running_loss:.6}")


# Evaluation function
def sequence(model, int_to_char, start_chars, seq_len=50-2, vocab_size=7):
    result = ''
    # concatenate the two one-hot encoded characters
    curr_char = np.concatenate(start_chars)
    model.eval()
    with torch.no_grad():
        for _ in range(seq_len):
            ins = torch.tensor(curr_char, dtype=torch.float32).unsqueeze(0)
            outs = model(ins)
            pred_index = torch.argmax(outs).item()
            pred_char = int_to_char[pred_index]
            result += pred_char
            # shift current characters and append the new one for next iteration
            curr_char = np.concatenate((curr_char[vocab_size:],
                                        np.eye(vocab_size)[pred_index]))
    return result


def main():
    # Read text file and get one-hot-encoded data
    one_hot_encoded, vocab_size, char_to_int, int_to_char = read_text_file(
        "abcde_edcba.txt")

    # Prepare training data
    X, Y = prepare_training_data(one_hot_encoded)

    # Define model, criterion, and optimizer
    input_size = vocab_size * 2
    hidden_size = 128
    output_size = vocab_size
    model = MLP(input_size, hidden_size, output_size)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_model(X, Y, model, criterion, optimizer, epochs=10)

    # Generate a random starting index within the text length
    start_index = np.random.randint(0, len(one_hot_encoded) - 2)
    initial_ints = [one_hot_encoded[start_index], one_hot_encoded[start_index + 1]]
    #initial_letters = ''
    initial_letters = []

    #for char in initial_ints: initial_letters += int_to_char[np.argmax(char)]
    for char in initial_ints: initial_letters.append(int_to_char[np.argmax(char)])
    print(20 * "#")
    print("Initial random letters to evaluate the model: ", initial_letters)

    # print True values from the txt file
    true_values = initial_letters[0] + initial_letters[1]
    for i in range(start_index, start_index + 48):
        true_values += int_to_char[np.argmax(Y[i])]
    print(20 * "#")
    print("True Values from the txt file:")
    print(true_values)

    # Evaluate the model by generating a char sequence
    generated_sequence = initial_letters[0] + initial_letters[1]
    generated_sequence += sequence(model, int_to_char, initial_ints)
    print(20 * "#")
    print("Generated Sequence by model:")
    print(generated_sequence)


if __name__ == "__main__":
    main()
