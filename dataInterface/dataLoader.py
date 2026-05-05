from pathlib import Path
import pandas as pd


class DataLoader:
    def __init__(self, config):
        self.cfg = config
        self.rawDataPath = self.cfg.rawDataRoot
        self.testData = self.cfg.testDataFile

    def loadSingleMiceData(self, filePath):
        bloodPressure = pd.read_excel(filePath, sheet_name='Blood Pressure (mmHg)', header=None)
        heartRate = pd.read_excel(filePath, sheet_name='Heart Rate (BPM)', header=None)
        glucose = pd.read_excel(filePath, sheet_name='Glucose (mg-dL)', header=None)
        breathingPeriod = pd.read_excel(filePath, sheet_name='Breathing Period (s)', header=None)
        zScore = pd.read_excel(filePath, sheet_name='z-score', header=None).iloc[0,0]
        signalDict = {
            'BloodPressure': bloodPressure,  # int64
            'HeartRate': heartRate,  # int64
            'Glucose': glucose,  # int64
            'BreathingPeriod': breathingPeriod,  # float64
            'zScore': zScore  # float64
        }
        fileName = Path(filePath).stem
        parts = fileName.split('_')
        meta = {
            'date': parts[0],
            'ratId': parts[1],
            'binaryState': parts[2],
            'file': str(filePath)
        }
        signalDict['meta'] = meta
        return signalDict

    def loadAllData(self):
        allTrainingData = []
        for filePath in sorted(self.rawDataPath.glob('*.xlsx')):
            allTrainingData.append(self.loadSingleMiceData(filePath))
        print(f'Loaded {len(allTrainingData)} files')
        return allTrainingData

    def loadSingleTestData(self):
        # Data frame  (numPoints, numCols)
        bloodPressure = pd.read_excel(self.testData, sheet_name='Blood Pressure (mmHg)', header=0)
        heartRate = pd.read_excel(self.testData, sheet_name='Heart Rate (BPM)', header=0)
        glucose = pd.read_excel(self.testData, sheet_name='Glucose (mg-dL)', header=0)
        breathingPeriod = pd.read_excel(self.testData, sheet_name='Breathing Period (s)', header=0)

        signalDict = {
            'BloodPressure': bloodPressure,   # Data frame  (numPoints, numCols)
            'HeartRate': heartRate,  # Data frame  (numPoints, numCols)
            'Glucose': glucose,  # Data frame  (numPoints, numCols)
            'BreathingPeriod': breathingPeriod,  # Data frame  (numPoints, numCols)
        }
        return signalDict














