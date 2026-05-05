"""Main entry for stress analysis and visualization"""
from config import Config
from dataInterface.dataLoader import DataLoader
from dataInterface.featureExtractor import FeatureExtractor
from machineLearning.trainer import evaluate, trainFinal
from machineLearning.predictor import predictOnTest, postProcess
from machineLearning.models import Models
from plottingHelpers.plotter import plotAll, plotShap
import joblib
import pandas as pd


def main():
    cfg = Config
    dataLoader = DataLoader(cfg)
    allTrainingData = dataLoader.loadAllData()
    featureExtractor = FeatureExtractor()
    X, y, meta = featureExtractor.buildFeatureMatrix(allTrainingData)

    allResults = {}
    bestMetrics = None
    bestPredictions = None
    for modelName in cfg.modelNames:
        metrics, predictions = evaluate(X, y, meta, modelName)
        print(f'Model: {modelName}, MAE: {metrics['MAE']}, r: {metrics['r']}, r2: {metrics['r2']}, signAccuracy: {metrics['signAccuracy']}, threeClassAccuracy: {metrics['threeClassAccuracy']}')
        allResults[modelName] = metrics
        if modelName == cfg.bestModel:
            bestMetrics = metrics
            bestPredictions = predictions

    print(f'Training final model: {cfg.bestModel}')
    model, scaler = trainFinal(X, y, meta, cfg.bestModel, cfg.savePath)

    # testing
    testSignalDict = dataLoader.loadSingleTestData()
    artifact = joblib.load(cfg.savePath)
    testPredictions = predictOnTest(artifact, testSignalDict, featureExtractor, windowSeconds=cfg.slidingWindow, stepSeconds=cfg.step)
    testPredictions = postProcess(testPredictions)
    pd.DataFrame(testPredictions).to_excel('test_predictions.xlsx', index=False)
    plotAll(y, bestPredictions, bestMetrics, meta, testSignalDict, testPredictions, allResults, model, scaler, X, cfg.figurePath)
    modelDict = Models().returnModelRegistryForShap()
    plotShap(X, y, model, scaler, modelDict, cfg.figurePath)


if __name__ == '__main__':
    main()
