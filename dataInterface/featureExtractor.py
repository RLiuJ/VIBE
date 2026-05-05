import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class FeatureExtractor:
    def __init__(self):
        self.keys = ['BloodPressure', 'HeartRate', 'Glucose', 'BreathingPeriod']

    def extractFeatures(self, signal):
        features = {}
        for key in self.keys:
            values = signal[key]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0].values
            elif isinstance(values, pd.Series):
                values = values.values
            features[f'{key}_mean'] = np.mean(values)
            features[f'{key}_std'] = np.std(values)
            features[f'{key}_cv'] = np.std(values) / (np.mean(values) + 1e-8)
            features[f'{key}_min'] = np.min(values)
            features[f'{key}_max'] = np.max(values)
            features[f'{key}_range'] = np.max(values) - np.min(values)
            features[f'{key}_iqr'] = (np.percentile(values, 75) - np.percentile(values, 25) if len(values) >= 4 else np.ptp(values))
            features[f'{key}_median'] = np.median(values)
            features[f'{key}_skewness'] = skew(values) if len(values) >= 4 else 0
            features[f'{key}_kurtosis'] = kurtosis(values) if len(values) >= 4 else 0

        return features

    def buildFeatureMatrix(self, allTrainingData):
        featureList = []
        zScoreList = []
        metaList = []
        for signal in allTrainingData:
            featureList.append(self.extractFeatures(signal))
            zScoreList.append(signal['zScore'])
            metaList.append(signal['meta'])
        X = pd.DataFrame(featureList)
        y = np.array(zScoreList)
        meta = pd.DataFrame(metaList)
        return X, y, meta






