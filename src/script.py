import pandas as pd
from fbprophet import Prophet
import pickle
import numpy as np
import argparse
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # Input and output arguments.
    parser.add_argument('--input-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--output-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))


    args = parser.parse_args()
    
    print(args.input_dir)
    print(os.listdir(args.input_dir))

    # Find all *.txt files in the input directory.
    input_files = [ os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.lower().endswith('.csv') ]    
    model_name = os.path.splitext(os.path.basename(input_files[0]))[0]
    
#    print(input_files[0])
        
    data = pd.read_csv(input_files[0],parse_dates=['dt'], dayfirst = True)
    #data.head()
    for item_id in data.item_id.unique():
        
        data_item = data[(data['item_id'] == item_id)].sort_values(by=['dt'])
        df = data_item[['dt','item_ordered']]
        df.columns = ['ds','y']
        df = df[df['ds'] < '2019-05-01']

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=7)

        forecast = m.predict(future)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        
        actual = data_item[['dt','item_ordered']][data_item['dt'] >= '2019-05-01']
        predicted = forecast[['yhat']].tail(7).values
        #print(predicted)
        #print(actual['item_ordered'].fillna(0).values)

        mse = mean_squared_error(actual['item_ordered'].fillna(0).values, predicted)
        rmse = sqrt(mse)        
        
        print('RMSE: {}'.format(rmse))

        pkl_path = os.path.join(args.output_model_dir, "{}-{}.pkl".format(model_name, item_id))

        with open(pkl_path, "wb") as f:
            # Pickle the 'Prophet' model using the highest protocol available.
            pickle.dump(m, f)