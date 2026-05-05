import numpy as np
from .preprocessing import leaveOneOutSplitting
from .models import Models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def evaluate(X, y, meta, modelName):
    predictions = np.zeros(len(y))
    valX = X.values
    nFolds = len(y)
    for fold, (trainIdx, testIdx) in enumerate(leaveOneOutSplitting(X, y), 1):
        xTrain = valX[trainIdx]
        xTest = valX[testIdx]
        yTrain = y[trainIdx]
        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled = scaler.transform(xTest)
        model = Models(modelName).initializeModel()
        model.fit(xTrainScaled, yTrain)
        predictions[testIdx] = np.ravel(model.predict(xTestScaled))

    mae = mean_absolute_error(y, predictions)
    r = np.corrcoef(y, predictions)[0, 1]
    r2 = r2_score(y, predictions)
    signAccuracy = np.mean((y > 0) == (predictions > 0))

    def toClass(z):
        if z >= 0.0:
            return 'healthy'
        elif z > -2:
            return 'lowStress'
        else:
            return 'highStress'

    trueClass = [toClass(z) for z in y]
    predClass = [toClass(z) for z in predictions]
    threeClassAccuracy = np.mean([t == p for t, p in zip(trueClass, predClass)])

    metrics = {
        'MAE': mae,
        'r': r,
        'r2': r2,
        'signAccuracy': signAccuracy,
        'threeClassAccuracy': threeClassAccuracy
    }
    return metrics, predictions


def trainFinal(X, y, meta, modelName, savePath):
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(X.values)
    model = Models(modelName).initializeModel()
    model.fit(xScaled, y)
    artifact = {
        'scaler': scaler,
        'model': model,
        'featureColumns': list(X.columns),
    }
    joblib.dump(artifact, savePath)
    return model, scaler







