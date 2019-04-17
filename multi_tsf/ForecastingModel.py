import os
import shutil
import numpy as np
import tensorflow as tf
from multi_tsf.time_series_utils import ForecastTimeSeries

class ForecastingModel(object):
    def __init__(self,
                 name: str,
                 vector_output_mode: bool,
                 nb_steps_in: int,
                 nb_steps_out: int,
                 nb_input_features: int,
                 nb_output_features: int) -> None:
        self.name = name
        self.vector_output_mode = vector_output_mode
        self.nb_steps_in = nb_steps_in
        self.nb_steps_out = nb_steps_out
        self.nb_input_features = nb_input_features
        self.nb_output_features = nb_output_features

    def _create_model(self):
        raise NotImplementedError("Must override _create_model")

    def fit(self,
            forecast_data: ForecastTimeSeries,
            model_path: str,
            epochs: int,
            batch_size: int = 64,
            lr: float = 1e-3) -> None:

        os.makedirs(model_path, exist_ok=True)

        self.batch_size = batch_size
        self.placeholder_X = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_input_features])
        if self.vector_output_mode:
            self.placeholder_y = tf.placeholder(tf.float32, [None, self.nb_steps_out, self.nb_output_features])
        else:
            self.placeholder_y = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_output_features])


        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.train_dataset = self.train_dataset.batch(batch_size=self.batch_size).shuffle(buffer_size=100000)
        self.val_test_dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.val_test_dataset = self.val_test_dataset.batch(batch_size=self.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.train_iterator = self.iterator.make_initializer(self.train_dataset)
        self.val_test_iterator = self.iterator.make_initializer(self.val_test_dataset)

        self.data_X, self.data_y = self.iterator.get_next()
        self._create_model()

        self.optimizer = tf.train.AdamOptimizer(lr)
        train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            try:
                shutil.rmtree(model_path + '/logs/train')
                shutil.rmtree(model_path + '/logs/test')
            except FileNotFoundError as fnf_error:
                pass
            self.train_writer = tf.summary.FileWriter(model_path + '/logs/train', sess.graph)
            self.test_writer = tf.summary.FileWriter(model_path + '/logs/test')

            train_i = 0
            val_i = 0
            sess.run(self.train_iterator,
                     feed_dict={self.placeholder_X: forecast_data.train_X,
                                self.placeholder_y: forecast_data.train_y})

            for _ in range(epochs):
                while(True):
                    try:
                        _, loss, summary = sess.run([train_op, self.loss, merged])
                        self.train_writer.add_summary(summary, train_i)
                        train_i += 1
                    except tf.errors.OutOfRangeError:
                        break


                sess.run(self.val_test_iterator,
                         feed_dict={self.placeholder_X: forecast_data.val_X,
                                    self.placeholder_y: forecast_data.val_y})

                while(True):
                    try:
                        loss, summary = sess.run([self.loss, merged])
                        self.test_writer.add_summary(summary, val_i)
                        val_i += 1
                    except tf.errors.OutOfRangeError:
                        break

            self.model_path = model_path
            self.saver.save(sess, model_path + '/' + self.name, global_step=epochs)
            self.meta_path = model_path + '/' + self.name + '-%d.meta' % epochs

    def plot_historical(self, predicted_ts: np.array, actual_ts: np.array) -> None:
        raise NotImplementedError("Must override plot_historical function")

    def predict_historical(self,
                           forecast_data: ForecastTimeSeries,
                           set: str = 'Validation',
                           plot: bool =False,
                           num_features_display: int = None):
        if set == 'Validation':
            _X, _y = forecast_data.val_X, forecast_data.val_y
        elif set == 'Test':
            _X, _y = forecast_data.test_X, forecast_data.test_y
        elif set == 'Train':
            _X, _y = forecast_data.train_X, forecast_data.train_y
        else:
            _X, _y = forecast_data.val_X, forecast_data.val_y
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            sess.run(self.val_test_iterator,
                     feed_dict={self.placeholder_X: _X, self.placeholder_y: _y})
            predicted_ts = []
            actual_ts = []
            while (True):
                try:
                    pred_y, val_y = sess.run([self.pred_y, self.data_y])
                    predicted_ts.append(pred_y)
                    actual_ts.append(val_y)
                except tf.errors.OutOfRangeError:
                    break

            predicted_ts = np.vstack(predicted_ts).reshape(-1, self.nb_output_features)
            actual_ts = np.vstack(actual_ts).reshape(-1, self.nb_output_features)
            print('MSE')
            print(np.mean(np.square(predicted_ts - actual_ts)))

            if plot == True:
                self.plot_historical(predicted_ts, actual_ts, num_features_display)

            return predicted_ts, actual_ts


    def predict_periods(self,
                        forecast_data: ForecastTimeSeries):
        pass
