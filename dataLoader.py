import os
import pandas
from pandas import read_csv, concat
from sklearn.preprocessing import LabelEncoder


class DataLoader():
    def __loadData_BLE(self, path='data/training/BLE/iBeacon_RSSI_Labeled.csv'):
        le_col = LabelEncoder()
        x = read_csv(path, index_col=None)
        x['loc'] = le_col.fit_transform(x['location'])
        df = x
        df = df.drop(['location', 'date'], axis=1)
        return df

    def __loadData_traffic(self, path='data/training/traffic/trafficData158324.csv'):
        x = read_csv(path, index_col=None, usecols=[5, 6])
        x['date'] = pandas.to_datetime(x['TIMESTAMP'])
        x['weekNo'] = x['date'].dt.week
        x['weekDay'] = x['date'].dt.weekday
        x['hour'] = x['date'].dt.hour
        x['minute'] = x['date'].dt.minute
        x['interval'] = x['minute'] // 30
        x = x.drop(['TIMESTAMP', 'date'], axis=1)
        df = x.groupby(['weekNo', 'weekDay', 'hour', 'interval']).mean().reset_index()
        df['minute'] = df['interval'] * 30
        df = df[['weekNo', 'weekDay', 'hour', 'minute', 'vehicleCount']]
        return df

    def __loadData_electricity(self, path='data/training/electricity'):
        x = []
        files = os.listdir(path)
        files.sort()
        for filename in files:
            ffn = os.path.join(path, filename)
            dfData = read_csv(ffn, index_col=None, header=0, usecols=[1, 2])
            dfData['date'] = pandas.to_datetime(dfData['SETTLEMENTDATE'])
            dfData['weekNo'] = dfData['date'].dt.week
            dfData['weekDay'] = dfData['date'].dt.weekday
            dfData['hour'] = dfData['date'].dt.hour
            dfData['minute'] = dfData['date'].dt.minute
            dfData = dfData.drop(['SETTLEMENTDATE', 'date'], axis=1)
            x.append(dfData)
        df = concat(x, ignore_index=True)
        df = df[['weekNo', 'weekDay', 'hour', 'minute', 'TOTALDEMAND']]

        df.columns = ['weekNo', 'weekDay', 'hour', 'minute', 'demand']

        return df

    def loadData(self, dataset='electricity'):
        if dataset == 'electricity':
            return self.__loadData_electricity()
        elif dataset == 'traffic':
            return self.__loadData_traffic()
        elif dataset == 'BLE':
            return self.__loadData_BLE()

    def loadModelsOutput(self, dataset='electricity'):
        folder = 'data/output/individual_models/predictions_output_%s.csv' % dataset
        df = read_csv(folder, index_col=None)
        return df
