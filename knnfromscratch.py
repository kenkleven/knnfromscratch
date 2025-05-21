import numpy as np 

class KNNClassifier():
    def __init__(self, k=5):
        self.k = k
        
    def distance(self, Xi, Xj):
        return np.sqrt(np.sum((Xi - Xj) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        return predictions

    def accuracy(self, y_true, y_pred):
        correct = np.sum(np.array(y_true) == np.array(y_pred))
        return correct / len(y_true)
    
    def precision(self, y_true, y_pred, positive_class=1):
        tp = 0
        fp = 0 
        
        for yt, yp in zip(y_true, y_pred):
            if yp == positive_class:
                if yt == positive_class:
                    tp += 1
                else:
                    fp += 1
        
        if tp + fp == 0:
            return 0.0 
        return tp / (tp + fp)
    
    def recall(self, y_true, y_pred, positive_class=1):
        tp = 0
        fn = 0 
        
        for yt, yp in zip(y_true, y_pred):
            if yt == positive_class:
                if yp == positive_class:
                    tp += 1
                else:
                    fn += 1
        
        if tp + fn == 0:
            return 0.0 
        return tp / (tp + fn)
    
    def f1_score(self, y_true, y_pred, positive_class=1):
        prec = self.precision(y_true, y_pred, positive_class)
        rec = self.recall(y_true, y_pred, positive_class)
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)




class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def distance(self, Xi, Xj):
        return np.sqrt(np.sum((Xi - Xj) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_targets = [self.y_train[i] for i in k_indices]
            prediction = np.mean(k_nearest_targets)
            predictions.append(prediction)
        return np.array(predictions)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
