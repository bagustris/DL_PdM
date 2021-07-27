import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'		# disable warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import logging
from normal_models import build_SVR, build_RF, build_NN
from dl_models import build_BILSTM, build_LSTM, build_CNN
from dl_models import build_pre_normalAE, build_pre_denoiseAE
from data_loaders import load_data
from tensorflow.random import set_random_seed
from numpy.random import seed

# if running in GPU, uncommet the following two lines
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed(10132017)
set_random_seed(18071991)

log = "output.log"
logging.basicConfig(filename=log, level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


if __name__ == '__main__':
    for time in range(5):
        idx_test = 1
        logging.info("conducting exp.")
        t = load_data(True)
        X_train, y_train, X_test, y_test = t[0], t[1], t[2], t[3]
        data_dim = X_train.shape[1]

    	# liner svm
        linear_svr = build_SVR('linear')
        linear_svr.fit(X_train, y_train)
        y_pred = linear_svr.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   linear svr result: %f_%f" % (score, mae_score))
        rbf_svr = build_SVR('rbf')
        rbf_svr.fit(X_train, y_train)
        y_pred = rbf_svr.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   rbf svr result: %f_%f" % (score, mae_score))

    	# random forest
        NUM_ESTIMATOR = 50
        NUM_PREEPOCH = 30
        NUM_BPEPOCH = 30
        BATCH_SIZE = 24
        rf = build_RF(NUM_ESTIMATOR)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   randomforest  result: %f_%f" % (score, mae_score))

        # neural network
        nn_model = build_NN(data_dim)
        nn_model.fit(X_train, y_train, epochs=NUM_BPEPOCH,
                     batch_size=BATCH_SIZE)
        y_pred = nn_model.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   neural network result: %f_%f" % (score, mae_score))

        # AE
        normal_AE = build_pre_normalAE(
            data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140, 280])
        normal_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH,
                      batch_size=BATCH_SIZE)
        y_pred = normal_AE.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   normal ae result: %f_%f" % (score, mae_score))

        # denoise AE
        denois_AE = build_pre_denoiseAE(
            data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140, 280])
        denois_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH,
                      batch_size=BATCH_SIZE)
        y_pred = denois_AE.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   denoise ae result: %f_%f" % (score, mae_score))

        # Bi-directional LSTM
        t = load_data(False)
        X_train, y_train, X_test, y_test = t[0], t[1], t[2], t[3]
        data_dim = X_train.shape[2]
        timesteps = X_train.shape[1]
        biLSTM = build_BILSTM(timesteps, data_dim)
        biLSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATCH_SIZE)
        y_pred = biLSTM.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   bi-lstm result: %f_%f" % (score, mae_score))

        # LSTM
        LSTM = build_LSTM(timesteps, data_dim)
        LSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATCH_SIZE)
        y_pred = LSTM.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   lstm result: %f_%f" % (score, mae_score))

        # CNN
        CNN = build_CNN(timesteps, data_dim)
        CNN.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATCH_SIZE)
        y_pred = CNN.predict(X_test)
        score = mean_squared_error(y_pred, y_test)
        mae_score = mean_absolute_error(y_pred, y_test)
        logging.info("   cnn result: %f_%f" % (score, mae_score))
