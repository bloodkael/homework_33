import os
from os import walk
import pandas as pd
import dill
import json
from datetime import datetime

def get_data_from_file(filePath):
    with open('../data/test/' + filePath) as file_with_data:
        data_obj = pd.json_normalize(json.loads(file_with_data.read()))
    return data_obj


def predict():
    last_pipeline = os.listdir('../data/models')[-1]
    with open('../data/models/' + last_pipeline, 'rb') as file:
        model = dill.load(file)
    filenames = next(walk('../data/test/'), (None, None, []))[2]
    data_preds = []
    car_ids = []
    for fileName in filenames:
        data = get_data_from_file(fileName)
        data_preds.append(model.predict(data)[0])
        car_ids.append(data.id[0])
    pd.DataFrame({'pred': data_preds, 'car_id': car_ids})\
        .to_csv('../data/predictions/predict_' + datetime.now().strftime("%Y%m%d%H%M") + '.csv', index=False)
    pass


if __name__ == '__main__':
    predict()
