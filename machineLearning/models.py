from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


def _buildRegistry():
    return {
        'Ridge': Ridge(alpha=10),
        'ElasticNet': ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000),
        'SVRLinear': SVR(kernel='linear', C=1.0, epsilon=0.1),
        'SVRRBF': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'GBR': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42),
    }


class Models:
    def __init__(self, modelName=None):
        self.model = modelName

    def initializeModel(self):
        modelRegistry = _buildRegistry()
        if self.model in modelRegistry:
            return modelRegistry[self.model]
        else:
            raise NotImplementedError(f'Model {self.model} not in registry. Available: {list(modelRegistry.keys())}')

    def returnModelRegistryForShap(self):
        return _buildRegistry()