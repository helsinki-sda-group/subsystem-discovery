#%%
from config import US_WEATHER_DATA_PATH
import pandas as pd

def get_us_weather_dataloader(args, wrap_to_future_predict_dataloader):
    df = pd.read_parquet(US_WEATHER_DATA_PATH)
    #print('cropping data to 50k samples 20 sensors for testing!')
    #df = df.iloc[:50_000, :50] # lets just take 200k samples

    data = df.to_numpy()

    print('us weather data shape', data.shape)
    # normalize per columns
    print('normalizing data over axis 0')

    include_num_elem = (1 - args.test_len * 2) * data.shape[0]
    train_data_mean = data[:int(include_num_elem)].mean(axis=0, keepdims=True)
    train_data_std = data[:int(include_num_elem)].std(axis=0, keepdims=True)

    data = (data - train_data_mean) / train_data_std
    print('done normalizing')

    train_loader, val_loader, test_loader, pred_loader = wrap_to_future_predict_dataloader(df.index, data, args)

    return train_loader, val_loader, test_loader, pred_loader, df.columns.to_list(), train_loader.dataset.data_x

if __name__ == '__main__':
    # _, _, _, _, columns, _ = get_us_weather_dataloader(config)
    # print('loader columns', columns)
    # print(len(columns))

    df = pd.read_parquet(US_WEATHER_DATA_PATH)
    print('col print', df.columns.to_list())
# %%
