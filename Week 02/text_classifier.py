"""
DATA.ML.200
Week 2, title: Conventional machine learning.

(In the previous exercise the vocabularity included all words found in the training
data. This made the Bag-of-Word (BoW) feature vectors unnecessarily long.
Moreover, a curated set of English ’stop words’ were used to clean uninformative
words. In this exercise, the BoW feature is investigated more in detail. We set
different number of BoW feature vectors(less than max) and see how accuracy
gradually converged to the maximum level. Next we apply TF-IDF weighting of the
feature vectors by adding TfidTransformer() after the CountVectorizer().
This applies weights to the vectors and possibly improves the results.)

Creator: Maral Nourimand
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# maximum number of BoW feature vectors created by CountVectorize()
MAX_WITH_STOP = 130107
MAX_WITHOUT_STOP = 129796


def nn_accuracy_cal(data_train, data_test, max_feature, rstop_word=True, tf=False):
    """
    this function, gets the maximum number of feature vectors which should be
    made for the data during vectorization, then applies 1NN classifiers,
    calculates its accuracy and return the accuracy.

    :param data_train: training data
    :param data_test: test data
    :param max_feature: number of Bag-of-Word (BoW) feature vectors
    :param rstop_word: when it's true, we remove stop words. Default is True
    :param tf: when it's true, we apply TfidTransformer(). Default is False.
    :return: accuracy of 1NN classifier
    """
    # vectorize and form data
    if rstop_word:  # remove stop words from the training data
        vectorizer = CountVectorizer(stop_words="english", max_features=max_feature)
    else:
        vectorizer = CountVectorizer(max_features=max_feature)

    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    if tf:
        # apply TF-IDF weighting
        tfidf_transformer = TfidfTransformer()
        X_train = tfidf_transformer.fit_transform(X_train)
        X_test = tfidf_transformer.transform(X_test)

    y_train, y_test = data_train.target, data_test.target

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_train, y_train)

    y_knn_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_knn_pred)

    return accuracy


def main():
    data_train = fetch_20newsgroups(subset='train')
    data_test = fetch_20newsgroups(subset='test')

    #########################################################
    #               without stop words
    #########################################################
    y1 = []
    vocabulary_sizes_y1 = [10, 100, 1000, 5000, 8000, 10000, MAX_WITHOUT_STOP]

    print("1NN Accuracy without stop words:")
    for size in vocabulary_sizes_y1:
        num_features = size
        accuracy = nn_accuracy_cal(data_train, data_test, num_features)
        print(10 * "#")
        print(f"1NN Accuracy for {num_features} features: {accuracy:.4f}")
        y1.append(accuracy)

    # USING ALL VOCAB (FULL)
    num_features = MAX_WITHOUT_STOP
    accuracy_full = nn_accuracy_cal(data_train, data_test, num_features)
    print(10 * "#")
    print(f"1NN Accuracy for {num_features} features (Full Vocab): "
          f"{accuracy_full:.4f}")

    #########################################################
    #               with stop words
    #########################################################
    y2 = []
    vocabulary_sizes_y2 = [10, 100, 1000, 5000, 10000, 20000, 50000,
                           MAX_WITH_STOP]

    print(20 * "+")
    print(20 * "+")
    print("1NN Accuracy with stop words:")
    for size in vocabulary_sizes_y2:
        num_features = size
        accuracy = nn_accuracy_cal(data_train, data_test, num_features,
                                   rstop_word=False)
        print(10 * "#")
        print(f"1NN Accuracy for {num_features} features: {accuracy:.4f}")
        y2.append(accuracy)

    # USING ALL VOCAB (FULL)
    num_features = MAX_WITH_STOP
    accuracyS_full = nn_accuracy_cal(data_train, data_test, num_features,
                                     rstop_word=False)
    print(10 * "#")
    print(f"1NN Accuracy for {num_features} features (Full Vocab): {accuracyS_full:.4f}")

    #########################################################
    #               with TF-IDF, without stop words
    #########################################################
    y1_tf = []
    vocabulary_sizes_y1_tf = [10, 100, 1000, 5000, 8000, 10000, 20000, 40000,
                              MAX_WITHOUT_STOP]

    print(20 * "+")
    print(20 * "+")
    print("1NN Accuracy TF-IDF without stop words:")
    for size in vocabulary_sizes_y1_tf:
        num_features = size
        accuracy = nn_accuracy_cal(data_train, data_test, num_features, tf=True)
        print(10 * "#")
        print(f"1NN Accuracy for {num_features} features: {accuracy:.4f}")
        y1_tf.append(accuracy)

    # USING ALL VOCAB (FULL)
    num_features = MAX_WITHOUT_STOP
    accuracyTF_full = nn_accuracy_cal(data_train, data_test, num_features, tf=True)
    print(10 * "#")
    print(f"1NN Accuracy for {num_features} features (Full Vocab): "
          f"{accuracyTF_full:.4f}")

    #########################################################
    #               with TF-IDF, with stop words
    #########################################################
    y2_tf = []
    vocabulary_sizes_y2_tf = [10, 100, 1000, 5000, 8000, 10000, 20000, 50000,
                              MAX_WITH_STOP]

    print(20 * "+")
    print(20 * "+")
    print("1NN Accuracy TF-IDF with stop words:")
    for size in vocabulary_sizes_y2_tf:
        num_features = size
        accuracy = nn_accuracy_cal(data_train, data_test, num_features,
                                   rstop_word=False, tf=True)
        print(10 * "#")
        print(f"1NN Accuracy for {num_features} features: {accuracy:.4f}")
        y2_tf.append(accuracy)

    # USING ALL VOCAB (FULL)
    num_features = MAX_WITH_STOP
    accuracySTF_full = nn_accuracy_cal(data_train, data_test, num_features,
                                       rstop_word=False, tf=True)
    print(10 * "#")
    print(f"1NN Accuracy for {num_features} features (Full Vocab): "
          f"{accuracySTF_full:.4f}")

    # Plotting y1
    plt.plot(vocabulary_sizes_y1, y1, marker='o',
             linestyle='-', color='b', label='Accuracy w/o Stop Words')

    # Plotting y2
    plt.plot(vocabulary_sizes_y2, y2, marker='s',
             linestyle='-', color='r', label='Accuracy w Stop Words')

    # Plotting y1_tf
    plt.plot(vocabulary_sizes_y1_tf, y1_tf, marker='o',
             linestyle='-', color='g', label='Accuracy w/o Stop Words, w TF-IDF')

    # Plotting y2_tf
    plt.plot(vocabulary_sizes_y2_tf, y2_tf, marker='o',
             linestyle='-', color='m', label='Accuracy w Stop Words, w TF-IDF')

    # Adding straight lines for accuracy_full and accuracyS_full
    plt.axhline(y=accuracy_full, color='b', linestyle='--')
    plt.axhline(y=accuracyS_full, color='r', linestyle='--')
    plt.axhline(y=accuracyTF_full, color='g', linestyle='--')
    plt.axhline(y=accuracySTF_full, color='m', linestyle='--')

    plt.xlabel('Vocabulary Size N')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy vs Vocabulary Size')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
