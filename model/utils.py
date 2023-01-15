from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = start + int(0.9 * (stop - start))
            yield indices[start: mid], indices[mid + margin: stop]

            
def grid_search_param(X, y, estimator, grid, k_fold, verbose=1):
    tscv = TimeSeriesSplit(n_splits=k_fold, gap=3, test_size=4)
    RSCV = GridSearchCV(estimator=estimator,
                        param_grid=grid,
                        cv=tscv,
                        n_jobs=-1,
                        scoring='neg_root_mean_squared_error',
                        refit=True,
                        verbose=verbose,
                        pre_dispatch=8,
                        error_score=-999,
                        return_train_score=True)
    RSCV.fit(X, y)
    return RSCV.best_params_, RSCV.best_score_, RSCV.best_estimator_


def optimal_params(X, y, estimator, grid, fold_range, verbose=1):
    best_score = -100000
    for k_fold in fold_range:
        param, score, model = grid_search_param(X,
                                         y,
                                         estimator,
                                         grid,
                                         k_fold,
                                         verbose)
        # scoring: neg_root_mean_squared_error
        if score > best_score:
            best_score = score
            best_param = param
            k = k_fold
            best_model = model

    return k, best_param, best_score, model


def fit_check(y, y_pred, plot=True):
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    if plot:
        plt.plot(y.index, y, label="true value")
        plt.legend()
        plt.xticks(y.index[::100])
        plt.xlabel("")
        plt.ylabel("bitcoin return")
        plt.plot(y.index, y_pred, label="prediction")
        plt.legend()
        plt.show()
    return rmse


def reg_feature_select(coef, features, level=0):
    identify = coef != 0
    model_features = features[identify]
    model_coef = coef[identify]
    model_importance = pd.DataFrame({'features': model_features, 
                                     'coef': model_coef}).sort_values('coef', key=abs)

    aggregate = model_importance.coef.apply(lambda x: abs(x)).sum()
    model_importance['importance_ratio'] = model_importance.coef.apply(lambda x: abs(x)/aggregate)
    index = np.where((model_importance['importance_ratio'] >= level).to_numpy() == True)
    model_importance = model_importance.iloc[index]
    plt.figure(figsize=(20,10))
    plt.barh(model_importance.features, model_importance.importance_ratio)
    plt.xlabel("importance")
    plt.ylabel("features")
    plt.show()
    return model_importance


def pca_analysis(X, y, model):
    scaler = StandardScaler()
    y_scaled = pd.DataFrame(scaler.fit_transform(y),
                            columns=y.columns)
    pca_range = range(2, 70)
    rmses = []

    for i in pca_range:
        pca = PCA(n_components=i).fit_transform(X)
        columns = [f"pca{i}" for i in range(pca.shape[1])]
        X_pca = pd.DataFrame(pca, columns=columns)
    
        model = model.fit(X_pca, y_scaled)
        y_pred = pd.DataFrame(model.predict(X_pca))
        y_pred = pd.DataFrame(scaler.inverse_transform(y_pred), columns=y.columns)
        rmses.append(fit_check(y, y_pred, plot=False))
    plt.plot(rmses)
    plt.xlabel("number of PCA")
    plt.ylabel("RMSE")
    plt.show()
    return rmses


def tree_feature_select(model, features, level):
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=features,
        columns=["importance"]).sort_values("importance", ascending=True
    )
    index = np.where((feature_importance.importance >= level).to_numpy() == True)
    df = feature_importance.iloc[index]
    plt.barh(df.index, df.importance)

    return feature_importance.iloc[index].index


def output_html(path, table):
    with open(path,'w') as f:
        f.write(table.as_html())