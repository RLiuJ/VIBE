import numpy as np
import pandas as pd


def sampleEstimateSfAndWindowStep(testSignalDict, windowSeconds, stepSeconds):
    sf = {}
    windowSizes = []
    stepSizes = []
    for key, value in zip(testSignalDict.keys(), testSignalDict.values()):
        samplingFrequency = 1 / np.median(np.diff(value[:, 0]))
        sf = {
            f'sf_{key}': samplingFrequency
        }
        windowSizes.append(samplingFrequency * windowSeconds)
        stepSizes.append(samplingFrequency * stepSeconds)
    return sf, windowSizes, stepSizes


def predictOnTest(artifact, testSignalDict, featureExtractor, windowSeconds=45, stepSeconds=6):
    scaler = artifact['scaler']
    model = artifact['model']
    featureColumns = artifact['featureColumns']

    totalTime = testSignalDict['BloodPressure'].iloc[:, 0].values[-1]
    predictions = []

    tStart = 0
    while tStart + windowSeconds <= totalTime:
        tEnd = tStart + windowSeconds
        windowedSignals = {}
        for key in featureExtractor.keys:
            df = testSignalDict[key]
            time = df.iloc[:, 0].values
            vals = df.iloc[:, 1].values
            mask = (time >= tStart) & (time < tEnd)
            windowedSignals[key] = vals[mask]
        features = featureExtractor.extractFeatures(windowedSignals)
        featuresVec = np.array([features[col] for col in featureColumns]).reshape(1, -1)
        featureVecScaled = scaler.transform(featuresVec)
        zPred = np.ravel(model.predict(featureVecScaled))[0]
        predictions.append({
            'tStart': tStart,
            'tCenter': tStart + windowSeconds / 2,
            'tEnd': tEnd,
            'zPred': zPred
        })
        tStart += stepSeconds
    return predictions


def postProcess(predictions, clipLow=-np.inf, clipHigh=np.inf, medianWindow=3):
    zRaw = np.array([p['zPred'] for p in predictions])
    zClipped = np.clip(zRaw, clipLow, clipHigh)
    zSmoothed = pd.Series(zClipped).rolling(medianWindow, center=True, min_periods=1).median().values
    for i, p in enumerate(predictions):
        p['zClipped'] = zClipped[i]
        p['zSmoothed'] = zSmoothed[i]
        z = zSmoothed[i]
        p['class'] = 'healthy' if z > 0 else ('lowStress' if z > -2 else 'highStress')
    return predictions



