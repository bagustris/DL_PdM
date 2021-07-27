# parse the log
import numpy as np
filepath = 'output.log'


def parse_line(str_res):
    results = str_res.split("result:")[1].strip()
    mse, mae = results.split('_')
    return float(mse), float(mae)


model_names = ['linear svr', 'rbf svr', 'randomforest',
               'neural network', 'normal ae', 'denoise ae', 'bi-lstm', 'lstm', 'cnn']
for model_name in model_names:
    print(model_name)
    mse_array = []
    mae_array = []
    with open(filepath) as fp:
        for line in fp:
            if model_name in line:
                mse, mae = parse_line(line)
                mse_array.append(mse)
                mae_array.append(mae)
    print('MSE MEAN: ', np.mean(mse_array))
    print('MSE STD: ', np.std(mse_array))
    print('MAE MEAN: ', np.mean(mae_array))
    print('MAE STD:', np.std(mae_array))
