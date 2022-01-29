import numpy as np

REC_ERRORS = np.load("/content/drive/MyDrive/source_code/errors.npy")

TIME_INTERVAL = 107

ENERGY_THRESHOLD = 0.01

HIGH_PASS_FREQUENCY_INDEX = 211

INPUT_SHAPE = 14124

IMAGE_SHAPE = (132, 107)

METRIC_THRESHOLD = 0.15

RECOGNIZION_ERROR = np.percentile(REC_ERRORS, 77.5)

# MODEL

AUTOENCODER_PATH = "/content/drive/MyDrive/tnew_autoencoder.h5"
ENCODER_PATH = "/content/drive/MyDrive/tnew_encoder.h5"
DECODER_PATH = "/content/drive/MyDrive/tnew_decoder.h5"