from pathlib import Path


class Config:
    masterRoot = Path(__file__).resolve().parent
    dataSumRoot = masterRoot / 'data'
    rawDataRoot = dataSumRoot / 'raw'
    testDataFile = dataSumRoot / 'test_data.xlsx'
    modelNames = ['Ridge', 'ElasticNet', 'SVRLinear', 'SVRRBF', 'Lasso', 'GBR']
    bestModel = 'Ridge'
    savePath = masterRoot / 'artifacts'
    figurePath = masterRoot / 'figures'
    slidingWindow = 45  # seconds
    step = 6



