import os
import tensorflow as tf
from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from dataset_utils.dataloader import ImgIterator, load_images
from flushed_print import print
from collections import deque
from dataset_utils.data_aug_op import RandomColorAffine
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

class SIMCLR:
    def __init__(self, args, retrain_model=None):
        """
        Build the architecture of the siamese net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        """

        contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
        self.save_dir = args.simclr_path
        self.width = 512
        self.temperature = 0.1
        self.epochs = args.simclr_epochs
        self.batch_size = args.simclr_batch_size
        self.experiment_name = args.experiment_name
        self.retrain = args.retrain

        self.strategy = tf.distribute.MirroredStrategy()

        self.global_batch_size = args.simclr_batch_size * self.strategy.num_replicas_in_sync

        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        os.makedirs(self.save_dir, exist_ok=True)

        os.makedirs(os.path.join(self.save_dir, 'projection'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'encoder'), exist_ok=True)

        pro_checkpoint_path = os.path.join(self.save_dir, 'projection', "sim_clr.ckpt")
        enc_checkpoint_path = os.path.join(self.save_dir, 'encoder', "sim_clr.ckpt")


        def get_augmenter(min_area, brightness, jitter):
            zoom_factor = 1.0 - tf.sqrt(min_area)
            return tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(256, 256, 3)),
                    tf.keras.layers.experimental.preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
                    tf.keras.layers.Lambda(RandomColorAffine(zoom_factor, brightness, jitter))
                ]
            )
        with self.strategy.scope():
            base_model = ResNet50(weights="imagenet", include_top=False,
                                  input_tensor=tf.keras.Input(shape=(256, 256, 3)))

            num_ftrs = base_model.output_shape[-1]

            x = base_model.get_layer('conv5_block3_out').output
            out = GlobalAveragePooling2D()(x)

            self.encoder = Model(base_model.input, out, name='encoder')

            self.projection_head = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(num_ftrs,)),
                    tf.keras.layers.Dense(self.width, activation="relu"),
                    tf.keras.layers.Dense(self.width),
                ],
                name="projection_head",
            )

            self.contrastive_train_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
            self.contrastive_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name="c_acc"
            )
            self.contrastive_val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
            self.contrastive_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name="val_acc"
            )

            self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)

            self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pro_checkpoint_path,
                                                             monitor='val_loss',
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             mode='auto',
                                                             save_freq='epoch',
                                                             verbose=1)

            self.encoder_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=enc_checkpoint_path,
                                                                     monitor='val_loss',
                                                                     save_weights_only=True,
                                                                     save_best_only=True,
                                                                     mode='auto',
                                                                     save_freq='epoch',
                                                                     verbose=1)


            self.contrastive_optimizer = tf.keras.optimizers.Adam(0.004)

    def contrastive_loss(self, projections_1, projections_2, accuracy):

            projections_1 = tf.math.l2_normalize(projections_1, axis=1)
            projections_2 = tf.math.l2_normalize(projections_2, axis=1)
            similarities = (
                    tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
            )
            # The similarity between the representations of two augmented views of the
            # same image should be higher than their similarity with other views
            batch_size = tf.shape(projections_1)[0]
            contrastive_labels = tf.range(batch_size)
            accuracy.update_state(contrastive_labels, similarities)
            accuracy.update_state(
                contrastive_labels, tf.transpose(similarities)
            )

            loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, similarities, from_logits=True

            )
            loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, tf.transpose(similarities), from_logits=True

            )
            per_example_loss= (loss_1_2 + loss_2_1) / 2

            return per_example_loss

    def test_step(self, val_iter_1):

        a = self.contrastive_augmenter(val_iter_1, training=True)
        b = self.contrastive_augmenter(val_iter_1, training=True)

        a = preprocess_input(tf.cast(a * 255, tf.uint8))

        b = preprocess_input(tf.cast(b * 255, tf.uint8))

        features_1 = self.encoder(tf.convert_to_tensor(a), training=False)

        features_2 = self.encoder(tf.convert_to_tensor(b),training=False)

        # The representations are passed through a projection mlp
        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2, self.contrastive_val_accuracy)
        self.contrastive_val_loss_tracker.update_state(contrastive_loss)
        return self.contrastive_val_loss_tracker.result()

        # @tf.function

    def train_step(self, train_iter_1):
        with tf.GradientTape() as tape:
            a = self.contrastive_augmenter(train_iter_1, training=True)
            b = self.contrastive_augmenter(train_iter_1, training=True)

            a = preprocess_input(tf.cast(a * 255, tf.uint8))

            b = preprocess_input(tf.cast(b * 255, tf.uint8))

            features_1 = self.encoder(tf.convert_to_tensor(a), training=True)

            features_2 = self.encoder(tf.convert_to_tensor(b), training=True)

            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2, self.contrastive_train_accuracy)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_train_loss_tracker.update_state(contrastive_loss)
        return self.contrastive_train_loss_tracker.result()

    @tf.function
    def distributed_train_step(self,image_1):
        per_replica_losses = self.strategy.run(self.train_step, args=(image_1,))

        # per_replica_losses = self.strategy.experimental_local_results(per_replica_losses)
        # merged_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return per_replica_losses

    @tf.function
    def distributed_test_step(self,image_1):
        per_replica_losses = self.strategy.run(self.test_step, args=(image_1,))
        return per_replica_losses

    def train(self, train_bags, val_bags, dir, projection_head=None, encoder=None):
        """
        Train the siamese net
        Parameters
        ----------
        pairs_train : a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        check_dir   : str, specifying the directory where weights of the siamese net are going to be stored
        irun        : int reffering to the id of the experiment
        ifold       : fold reffering to the fold of the k-cross fold validation
        Returns
        -------
        A History object containing a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values

        """

        os.makedirs(dir, exist_ok=True)

        callbacks = CallbackList([self.cp_callback], add_history=True, model=self.projection_head)
        enc_callbacks = CallbackList([self.encoder_cp_callback], add_history=True, model=self.encoder)


        if self.retrain:
            self.projection_head = projection_head
            self.encoder = encoder

        logs = {}
        callbacks.on_train_begin(logs=logs)
        enc_callbacks.on_train_begin(logs=logs)

        train_hdf5Iterator = ImgIterator(train_bags, batch_size=self.global_batch_size, shuffle=False)
        train_img_loader = load_images(train_hdf5Iterator, num_child=1, batch_size=self.global_batch_size)
        train_steps_per_epoch = len(train_hdf5Iterator)-1


        val_hdf5Iterator = ImgIterator(val_bags, batch_size=self.global_batch_size, shuffle=False)
        val_img_loader = load_images(val_hdf5Iterator, num_child=1,  batch_size=self.global_batch_size)
        val_steps_per_epoch = len(val_hdf5Iterator)-1

        train_dataset = tf.data.Dataset.from_generator(lambda: train_img_loader, output_signature = tf.TensorSpec(shape=(None , 256, 256, 3 ), dtype=tf.float32))
        val_dataset = tf.data.Dataset.from_generator(lambda: val_img_loader, output_signature = tf.TensorSpec(shape=(None, 256, 256, 3 ), dtype=tf.float32))

        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

        early_stopping = 10
        loss_history = deque(maxlen=early_stopping + 1)

        for epoch in range(self.epochs):
            train_steps_done = 0
            val_steps_done = 0
            self.contrastive_train_accuracy.reset_states()
            self.contrastive_val_accuracy.reset_states()
            self.contrastive_train_loss_tracker.reset_states()
            self.contrastive_val_loss_tracker.reset_states()

            train_iter = iter(train_dataset)
            val_iter = iter(val_dataset)

            while train_steps_done< train_steps_per_epoch:

                callbacks.on_batch_begin(train_steps_done, logs=logs)
                callbacks.on_train_batch_begin(train_steps_done, logs=logs)

                self.distributed_train_step(next(train_iter))

                callbacks.on_train_batch_end(train_steps_done, logs=logs)
                callbacks.on_batch_end(train_steps_done, logs=logs)

                if train_steps_done % 500 == 0:
                            print("step: {} loss: {:.3f}".format(train_steps_done,
                            (float(self.contrastive_train_loss_tracker.result()))))
                train_steps_done += 1

            train_acc = self.contrastive_train_accuracy.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            while val_steps_done < val_steps_per_epoch:

                callbacks.on_batch_begin(val_steps_done, logs=logs)
                callbacks.on_test_batch_begin(val_steps_done, logs=logs)

                enc_callbacks.on_batch_begin(val_steps_done, logs=logs)
                enc_callbacks.on_test_batch_begin(val_steps_done, logs=logs)

                logs['val_loss'] = self.distributed_test_step(next(val_iter))

                if val_steps_done % 500 == 0:
                            print("step: {} loss: {:.3f}".format(val_steps_done,
                            (float(self.contrastive_val_loss_tracker.result()))))

                callbacks.on_test_batch_end(val_steps_done, logs=logs)
                callbacks.on_batch_end(val_steps_done, logs=logs)

                enc_callbacks.on_test_batch_end(val_steps_done, logs=logs)
                enc_callbacks.on_batch_end(val_steps_done, logs=logs)

                val_steps_done += 1

            print("Validation loss over epoch: %.4f" % (float(self.contrastive_val_loss_tracker.result()),))
            loss_history.append(self.contrastive_val_loss_tracker.result())
            callbacks.on_epoch_end(epoch, logs=logs)
            enc_callbacks.on_epoch_end(epoch, logs=logs)

            if len(loss_history) > early_stopping:
                if loss_history.popleft() < min(loss_history):
                    print(f'\nEarly stopping. No validation loss '
                          f'improvement in {early_stopping} epochs.')
                    break

            callbacks.on_train_end(logs=logs)
            enc_callbacks.on_train_end(logs=logs)

        return self.projection_head, self.encoder



