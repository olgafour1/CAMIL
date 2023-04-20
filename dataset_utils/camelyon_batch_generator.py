import numpy as np
import tensorflow as tf
import os
import h5py
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames, fold_id, args, shuffle=False, train=True, batch_size=1):
        self.filenames = filenames
        self.fold_id = fold_id
        self.dataset = args.dataset
        self.train = train
        self.batch_size = batch_size
        self.k = args.k
        self.shuffle = shuffle
        self.label_file = args.label_file
        self.on_epoch_end()

        self.enc = OneHotEncoder(handle_unknown='ignore')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames)))

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.filenames))

        if self.train == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        "returns one element from the data_set"
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.filenames[k] for k in indices]

        X, f, y = self.__data_generation(list_IDs_temp)

        return (X, f), np.array(y, dtype=np.float32)

    def __data_generation(self, filenames):
        """
        Parameters
        ----------
        batch_train:  a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        Returns
        -------
        bag_batch: a list of np.ndarrays of size (numnber of patches,h,w,d) , each of which contains the patches of an image
        neighbors: a list  of the adjacency matrices of size (numnber of patches,number of patches) of every image
        bag_label: an np.ndarray of size (number of patches,1) reffering to the label of the image
        """

        for i in range(len(filenames)):
            with h5py.File(filenames[i], "r") as hdf5_file:

                base_name = os.path.splitext(os.path.basename(filenames[i]))[0]

                features = hdf5_file['features'][:]
                neighbor_indices = hdf5_file['indices'][:]
                if self.dataset == "camelyon":
                    self.values = hdf5_file['similarities_0'.format(self.fold_id)][:]
                elif self.dataset == "tcga":
                    self.values = hdf5_file['similarities_{}'.format(self.fold_id)][:]
                elif self.dataset == "ovarian":
                    self.values = hdf5_file['similarities_{}'.format(self.fold_id)][:]
                else:
                    raise NotImplementedError

                if self.shuffle:
                    randomize = np.arange(neighbor_indices.shape[0])
                    np.random.shuffle(randomize)
                    features = features[randomize]
                    neighbor_indices = neighbor_indices[randomize]
                    self.values = self.values[randomize]

                references = pd.read_csv(self.label_file)
                if self.dataset == "camelyon":

                    if self.train:
                        try:
                            bag_label = references["train_label"].loc[references["train"] == base_name].values.tolist()[0]
                        except:
                            bag_label = references["val_label"].loc[references["val"] == base_name].values.tolist()[0]
                    else:
                        bag_label = references["test_label"].loc[references["test"] == base_name].values.tolist()[0]

                elif self.dataset == "tcga":
                    bag_label = references["slide_label"].loc[references["slide_id"] == base_name].values.tolist()[0]
                elif self.dataset == "sarcoma":
                    bag_label = references["slide_label"].loc[references["slide_id"] == base_name].values.tolist()[0]
               

        adjacency_matrix = self.get_affinity(neighbor_indices[:, :self.k + 1])

        return features, adjacency_matrix, bag_label

    def get_affinity(self, Idx):
        """
        Create the adjacency matrix of each bag based on the euclidean distances between the patches
        Parameters
        ----------
        Idx:   a list of indices of the closest neighbors of every image
        Returns
        -------
        affinity:  an nxn np.ndarray that contains the neighborhood information for every patch.
        """

        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()

        columns = Idx.ravel()

        neighbor_matrix = self.values[:, 1:]
        normalized_matrix =preprocessing.normalize(neighbor_matrix, norm="l2")

        similarities = np.exp(-normalized_matrix )

        #values = np.concatenate((np.ones(Idx.shape[0]).reshape(-1, 1), similarities), axis=1)

        values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

        values = values[:, :self.k + 1]
        values = values.ravel().tolist()

        sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
                                                   values=values,
                                                   dense_shape=[Idx.shape[0], Idx.shape[0]])
        sparse_matrix = tf.sparse.reorder(sparse_matrix)
        #     sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
        #                                            values=tf.ones(columns.shape, tf.float32),
        #                                            dense_shape=[Idx.shape[0], Idx.shape[0]])
        return sparse_matrix

