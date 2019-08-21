import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def drop_outlier(df, outliers):
    for name, outlier in outliers.items():
        df = df.drop(df[df[name] > outlier].index)
    return df


def process_features(train, test):
    features = pd.concat([train, test])

    features = features.loc[:, train.isnull().mean() < 0.2]
    features.drop(columns=['Utilities', 'MoSold', 'YrSold'], inplace=True)

    converter_to_str = [
        'MSSubClass',
    ]
    fill_in_mode = [
        'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional',
        'SaleType',
    ]
    fill_in_none = [
        'MSZoning', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond',
        #'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature',
    ]
    fill_in_zero = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea',
    ]
    fill_in_mean = [
        'LotFrontage',
    ]

    for column in converter_to_str:
        features[column] = features[column].apply(str)
    
    for column in fill_in_mode:
        features[column].fillna(
            features[column].mode()[0],
            inplace=True)

    for column in fill_in_none:
        features[column].fillna(
            'None',
            inplace=True)

    for column in fill_in_zero:
        features[column].fillna(
            0,
            inplace=True)

    for column in fill_in_mean:
        features[column].fillna(
            features[column].mean(),
            inplace=True)

    for column in features.select_dtypes(exclude='object').columns:
        features[column]= boxcox1p(features[column], boxcox_normmax(features[column]+1))
        features[column + '_log'] = np.log(features[column] + 1)
        features[column + '_log_square'] = np.log(features[column] + 1) ** 2

    assert features.isnull().sum().sum() == 0
    return pd.get_dummies(features)


def process_dataframes(train, test):
    train_ = drop_outlier(
        train,
        {
            'LotFrontage': 300,
            'LotArea': 100000,
            'BsmtFinSF1': 5000,
            'TotalBsmtSF': 6000,
            '1stFlrSF': 4000,
            'GrLivArea': 4000,
        })

    features = process_features(
        train_.drop(columns=['Id', 'SalePrice']),
        test.drop(columns=['Id']))

    train_features = features.iloc[:train_.shape[0], :]
    train_target = np.log(train_['SalePrice'])
    test_features = features.iloc[train_.shape[0]:, :]
    
    return train_features, train_target, test_features


train_features, train_target, test_features = process_dataframes(train, test)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor


class Model:

    models = {
        ElasticNet: {'alpha':0.0005, 'l1_ratio':0.9},
        ##RidgeCV: {},
        Lasso: {'alpha':0.0005},
        #Lasso: {'alpha':0.001, 'max_iter':1000},
        XGBRegressor: {'max_depth':4, 'learning_rate':0.05, 'n_estimators':600, 'subsample':0.6, 'colsample_bytree':0.8},
        #XGBRegressor: {
        #    'colsample_bytree':0.4603, 'gamma':0.0468, 
        #    'learning_rat':0.05, 'max_depth':3, 
        #    'min_child_weight':1.7817, 'n_estimators':2200,
        #    'reg_alpha':0.4640, 'reg_lambda':0.8571,
        #    'subsample':0.5213, 'silent':1
        #},
        ##RandomForestRegressor: {'n_estimators':400, 'max_features':'sqrt', 'oob_score':True},
        ##GradientBoostingRegressor: {
        ##    'n_estimators':3000, 'learning_rate':0.05,
        ##    'max_depth':4, 'max_features':'sqrt',
        ##    'min_samples_leaf':15, 'min_samples_split':10, 
        ##    'loss':'huber', 'random_state':5
        ##},
    }
    
    def __init__(self):
        self._models = {model: [] for model in self.models.keys()}
        self._rmse = {model: [] for model in self.models.keys()}
    
    def fit(self, train_features, train_target):
        from sklearn.model_selection import train_test_split
        
        for cls, params in self.models.items():
            model = cls(**params)
            model.fit(train_features, train_target)
            self._models[cls].append(model)
        
        #n = 2
        #for index in range(n):
        #    features, target, features_, target_ = self._split(train_features, train_target, n, index)
        #    for cls, params in self.models.items():
        #        model = cls(**params)
        #        model.fit(features, target)
        #        predicted_target_ = model.predict(features_)
        #        self._models[cls].append(model)
        #        self._rmse[cls].append(self.rmse(target_, predicted_target_))
                
        #for cls in self.models.keys():
        #    print(cls)
        #    print(self._rmse[cls])
            
    def predict(self, features):
        result = []
        model_count = 0
        for cls, models in self._models.items():
            for model in models:
                result.append(model.predict(features))
                model_count += 1
        return np.array([
            sum / model_count
            for sum in map(sum, zip(*result))
        ])
        

    def rmse(self, target, predicted_target):
        return np.round(np.sqrt(mean_squared_error(target, predicted_target)), 3)
                
    def _split(self, train_features, train_target, n, i):
        block_size = train_features.shape[0] // n
        
        train_rows = [r for r in range(train_features.shape[0]) if r % n != i]
        test_rows = [r for r in range(train_features.shape[0]) if r % n == i]

        features = train_features.iloc[train_rows]
        target = train_target.iloc[train_rows]
        features_ = train_features.iloc[test_rows]
        target_ = train_target.iloc[test_rows]
        
        return features, target, features_, target_


model = Model()
model.fit(train_features, train_target)

predicted_train_target = model.predict(train_features)
print(model.rmse(train_target, predicted_train_target))

predicted_sale_price = np.exp(model.predict(test_features))
pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predicted_sale_price
}).to_csv('submission.csv', index=False)
