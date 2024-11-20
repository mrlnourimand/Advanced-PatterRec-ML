"""
DATA.ML.200
Week 7, title: Do-It-Yourself RNN, part 1: Letter encoding.

(This code reads the characters in the file one by one and converts them to a
 single matrix that has as many lines as characters in the file and columns
  represent one-hot-encoding. For future purposes it is written as a function
  that takes the file name as input and returns the one-hot-encoded data matrix,
   vocabulary size and ’char to int’ and ’int to char’ Python dictionaries to
    convert integers to characters and characters to integers.

Creator: Maral Nourimand
"""

import numpy as np


def read_text_file(filename):
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


def main():
    filename = 'abcde.txt'
    data_matrix, vocab_size, char_to_int, int_to_char = read_text_file(filename)


if __name__ == "__main__":
    main()

