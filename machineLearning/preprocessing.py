from sklearn.model_selection import LeaveOneOut


def leaveOneOutSplitting(X, y):
    loo = LeaveOneOut()
    for trainIdx, testIdx, in loo.split(X):
        yield trainIdx, testIdx





