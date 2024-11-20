"""
DATA.ML.200
Week 1, title: Machine learning fundamentals

(1-NN classification using 20 Newsgroups dataset)

Creator: Maral Nourimand

Note: I first used euclidean() function to compute the distances one by one
which was too slow!! Then I changed to pair_distances_argmin_min() which is so
fast and compute the distances for the whole data at once.
"""

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
import timeit
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier


def nearest_neighbor_classifier(X_train, y_train, X_test):
    """
    custom function, measures Euclidean distance and classifies test samples.
    """

    # num_test_samples = X_test.shape[0]
    # num_train_samples = X_train.shape[0]
    # y_pred = np.zeros(num_test_samples)

    # this for loop is too slow!!
    # for i in range(num_test_samples):
    #     distances = np.zeros(num_train_samples)
    #     for j in range(num_train_samples):
    #         #distances[j] = compute_euclidean_distance(X_test[i], X_train[j])
    #         distances[j] = euclidean(X_test[i], X_train[j])
    #     nearest_neighbor_index = np.argmin(distances)
    #     y_pred[i] = y_train[nearest_neighbor_index]

    # This function computes for each row in X_test, the index of the row of
    # X_train which is closest. The minimal distances are also returned.
    distances = pairwise_distances_argmin_min(X_test,
                                              X_train, metric='euclidean')

    nearest_neighbors = distances[0]

    # use the class labels of the nearest neighbors as predictions
    y_pred = y_train[nearest_neighbors]

    return y_pred


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def main():
    data_train = fetch_20newsgroups(subset='train')
    data_test = fetch_20newsgroups(subset='test')

    #pprint(data_train.target_names)

    #print(
    #    f'Total of {len(data_train.data)} posts in the dataset and the total '
    #    f'size is {size_mb(data_train.data):.2f}MB')

    # vectorize and form data
    vectorizer = CountVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    y_train, y_test = data_train.target, data_test.target
    # print(y_train.shape)
    # print(X_train.shape)
    # print(X_train.shape[0])
    # print(X_test.shape)

    # classify using DummyClassifier with 'most frequent class' strategy
    dummy_classifier = DummyClassifier(strategy='most_frequent')
    start_time = timeit.default_timer()
    dummy_classifier.fit(X_train, y_train)
    predictions = dummy_classifier.predict(X_test)
    end_time = timeit.default_timer()

    # measure computation time
    computation_time = end_time - start_time

    accuracy = accuracy_score(y_test, predictions)
    print(f"Baseline Classification Accuracy: {accuracy:.4f}")
    print(f"Computation Time: {computation_time:.4f} seconds")


##############################################################
#                  Part e & f                                #
##############################################################
    # k-NN classifier by a custom implementation
    start_time = timeit.default_timer()
    num_samples_to_use = X_test.shape[0]
    # y_knn_pred_custom = nearest_neighbor_classifier(X_train.toarray(), y_train, X_test[:num_samples_to_use].toarray())

    y_knn_pred_custom = nearest_neighbor_classifier(X_train,
                                                    y_train,
                                                    X_test[:num_samples_to_use])

    end_time = timeit.default_timer()

    # Compute accuracy for custom k-NN
    knn_accuracy_custom = accuracy_score(y_test[:num_samples_to_use],
                                         y_knn_pred_custom)

    estimated_total_time = (end_time - start_time) / num_samples_to_use * len(data_test.target)

    print(20 * "#")
    print(f"Number of test samples used: {num_samples_to_use}")
    print(f"Custom Nearest Neighbor Accuracy: {knn_accuracy_custom:.4f}")
    print(f"Estimated Total Computation Time: {estimated_total_time:.4f} seconds")


##############################################################
#                  Part g & h                                #
##############################################################

    # k-NN classifier from ScikitLearn (k=1 for nearest neighbor)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)

    knn_classifier.fit(X_train, y_train)

    start_time_knn = timeit.default_timer()
    y_knn_pred = knn_classifier.predict(X_test)
    end_time_knn = timeit.default_timer()
    computation_time_knn = end_time_knn - start_time_knn

    # accuracy for k-NN
    knn_accuracy = accuracy_score(y_test, y_knn_pred)
    print(20 * "#")
    print(f"Sklearn NN Accuracy: {knn_accuracy:.4f}")
    print(f"Computation Time (1-NN): "
          f"{computation_time_knn:.4f} seconds")


if __name__ == "__main__":
    main()
