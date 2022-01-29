import os
import sys
import numpy as np
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import Input, Model
from constants import *
from signal_preprocessing import filtered_intervals_generator, r_Sxx,\
                                    filtered_intervals_generator_to_predict

                                    

class ResultGenerator:

    def __init__(self, input_dir, output_dir, batch_size=1):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH,custom_objects=None, compile=False, options=None)
        self.encoder = tf.keras.models.load_model(ENCODER_PATH, custom_objects=None, compile=False, options=None)
        self.decoder = tf.keras.models.load_model(DECOER_PATH, custom_objects=None, compile=False, options=None)
        self.spectrograms_dirname = "spectrograms"

    def __call__(self, random_seed=123, *args, **kwargs):
        self.preprocess_randomly_all_files(random_seed)

    def preprocess_randomly_all_files(self, random_seed):
        meas = [os.path.join(self.input_dir, name, 'measurements') for name in os.listdir(self.input_dir)]
        all_meas = []
        for m in meas:
            all_meas += [os.path.join(m, name) for name in os.listdir(m)]
        np.random.seed(random_seed)
        np.random.shuffle(all_meas)
        for idx, file_path in enumerate(all_meas):
            file_name = os.path.basename(file_path)
            date_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            self.process_one_file(date_dir, file_name)
            if idx % 100 == 0:
                print(f"{idx} files processed with success!")



    def process_one_file(self, date_dir, file_name):
        file_path = os.path.join(self.input_dir, date_dir, 'measurements', file_name)
        stft_generator = filtered_intervals_generator_to_predict(file_path, batch_size=self.batch_size)
        _ = next(stft_generator)
        if _ is None:
            return
        encoded_img = self.encoder.predict_generator(stft_generator)
        decoded_img = self.decoder.predict_generator(encoded_img)
        stft_generator = filtered_intervals_generator_to_predict(file_path, batch_size=self.batch_size)
        f, new_t = next(stft_generator)
        t = new_t[-1]
        sel_file_dir, rej_file_dir = self.create_dirs(date_dir, file_name)
        sel_spec_file_dir = os.path.join(sel_file_dir, self.spectrograms_dirname)
        rej_spec_file_dir = os.path.join(rej_file_dir, self.spectrograms_dirname)

        # metric_list = self.calc_metric_for_matrix(Sxx, decoded_img)
        file_to_save = ""
        # t_ids = np.where(metric_list > RECOGNIZION_ERROR)
        i = 0
        for one_Sxx in stft_generator:
            plt.pcolormesh(new_t, f, one_Sxx.reshape(IMAGE_SHAPE), vmax=0.5)
            plt.title('Spektrogram transformaty Gabora')
            plt.ylabel('Częstotliwość [Hz]')
            plt.xlabel('Czas [s]')
            metric = self.max_metric(one_Sxx, decoded_img[i])
            if metric > RECOGNIZION_ERROR:
                plt.savefig(os.path.join(sel_spec_file_dir, f"file{i}.png"))
                file_to_save += f"{t * i}, {t * (i + 1)}\n"
            else:
                plt.savefig(os.path.join(rej_spec_file_dir, f"file{i}.png"))
            plt.close()
            if file_to_save:
              with open(os.path.join(sel_file_dir, "time.txt"), "w+") as d:
                  d.write(file_to_save)
            i += 1
        #print(f"Processing file: {file_path} finished with success!")


    def create_dirs(self, date_dir, file_name):
        selected_files = os.path.join(self.output_dir, "selected_files")
        rejected_files = os.path.join(self.output_dir, "rejected_files")

        # threshold = np.percentile(rec_errors, 77.5)

        # os.makedirs(selected_files, exist_ok=True)
        # os.makedirs(rejected_files, exist_ok=True)
        file_dir = file_name.replace(".wfm", "")
        #TODO move to another method os.path.basename(file_path)
        # date_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path))) #TODO move to another method
        sel_file_dir = os.path.join(selected_files, date_dir, file_dir)
        rej_file_dir = os.path.join(rejected_files, date_dir, file_dir)
        sel_spec_file_dir = os.path.join(sel_file_dir, self.spectrograms_dirname)
        rej_spec_file_dir = os.path.join(rej_file_dir, self.spectrograms_dirname)
        if os.path.exists(sel_spec_file_dir):
            shutil.rmtree(sel_spec_file_dir)
        os.makedirs(sel_spec_file_dir)
        if os.path.exists(rej_spec_file_dir):
            shutil.rmtree(rej_spec_file_dir)
        os.makedirs(rej_spec_file_dir)
        return sel_file_dir, rej_file_dir



    def calc_metric_for_matrix(self, Sxx, decoded_img):
        calc_metric = lambda x, y: self.max_metric(x, y)
        return list(map(calc_metric, Sxx, decoded_img))


    @staticmethod
    def recognizion_error(X_test, X_pred):
        return np.power(X_test - X_pred, 2)

    @staticmethod
    def max_metric(X_test, X_pred):
        return np.max(ResultGenerator.recognizion_error(X_test, X_pred))
