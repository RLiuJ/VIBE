import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from pathlib import Path
import shap
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.base import clone as _clone


PlottingStyle = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

Colors = {
    'healthy': '#2E86AB',
    'lowStress': '#E8963E',
    'highStress': '#C73E1D',
    'BP': '#2E86AB',
    'HR': '#C73E1D',
    'Gluc': '#2CA58D',
    'Br': '#E8963E',
    'pred_raw': '#9B99A9',
    'pred_smooth': '#5B2C8E',
    'scatter_healthy': '#2E86AB',
    'scatter_stressed': '#C73E1D',
}

SingleColInch = 3.50
DoubleColInch = 7.20


def _applyStyle():
    plt.rcParams.update(PlottingStyle)


def _addPanelLabel(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, va='top', ha='left')


def _toClass(z):
    if z >= 0:
        return 'healthy'
    elif z > -2:
        return 'lowStress'
    else:
        return 'highStress'


def _cleanName(name):
    return (name.replace('BloodPressure', 'BP')
                .replace('HeartRate', 'HR')
                .replace('BreathingPeriod', 'Breath')
                .replace('Glucose', 'Gluc')
                .replace('_', ' '))


def _getShapExplainer(model, X_scaled_df):
    if hasattr(model, 'coef_'):
        return shap.LinearExplainer(model, X_scaled_df)
    else:
        return shap.KernelExplainer(model.predict, X_scaled_df)


def _buildShapData(X, y, model, scaler):
    cleanNames = [_cleanName(c) for c in X.columns]
    X_scaled = scaler.transform(X.values)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cleanNames)
    explainer = _getShapExplainer(model, X_scaled_df)
    shapValues = explainer.shap_values(X_scaled_df)
    return X_scaled_df, shapValues, explainer, cleanNames


def plotTrainingPerformance(y, predictions, metrics, savePath):
    _applyStyle()
    fig, axes = plt.subplots(1, 2, figsize=(DoubleColInch, 3.0), gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axes[0]
    _addPanelLabel(ax, 'a')
    for i in range(len(y)):
        color = Colors['scatter_healthy'] if y[i] >= 0 else Colors['scatter_stressed']
        ax.scatter(y[i], predictions[i], c=color, s=25, edgecolors='none', linewidth=0, zorder=3)

    lims = [min(y.min(), predictions.min()) - 0.5, max(y.max(), predictions.max()) + 0.5]
    ax.plot(lims, lims, '--', color='#888888', linewidth=0.7, zorder=1)
    ax.axhline(0, color='#cccccc', linewidth=0.4, zorder=0)
    ax.axvline(0, color='#cccccc', linewidth=0.4, zorder=0)
    ax.set_xlabel('True z-score')
    ax.set_ylabel('Predicted z-score (LOOCV)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    textStr = (f"MAE = {metrics['MAE']:.2f}\n"
               f"r = {metrics['r']:.2f}\n"
               f"R\u00b2 = {metrics['r2']:.2f}\n"
               f"Sign acc. = {metrics['signAccuracy'] * 100:.0f}%")
    ax.text(0.05, 0.95, textStr, transform=ax.transAxes, fontsize=7, va='top', ha='left')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=Colors['scatter_healthy'], markersize=5, markeredgecolor='none', label='Healthy (z \u2265 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=Colors['scatter_stressed'], markersize=5, markeredgecolor='none', label='Stressed (z < 0)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False)
    ax2 = axes[1]
    _addPanelLabel(ax2, 'b')
    classLabels = ['healthy', 'lowStress', 'highStress']
    displayLabels = ['Healthy', 'Low\nstress', 'High\nstress']
    trueClasses = [_toClass(z) for z in y]
    predClasses = [_toClass(z) for z in predictions]
    labelMap = {c: i for i, c in enumerate(classLabels)}
    trueIdx = [labelMap[c] for c in trueClasses]
    predIdx = [labelMap[c] for c in predClasses]
    cm = confusion_matrix(trueIdx, predIdx, labels=[0, 1, 2])
    rowSums = cm.sum(axis=1, keepdims=True)
    rowSums[rowSums == 0] = 1
    cmFrac = cm / rowSums
    cmap = plt.cm.Purples
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(cmFrac[i, j] * 0.6), edgecolor='white', linewidth=0.5)
            ax2.add_patch(rect)
            textColor = 'white' if cmFrac[i, j] > 0.5 else 'black'
            ax2.text(j, i, f'{cmFrac[i, j]:.2f}', ha='center', va='center', fontsize=7, color=textColor)

    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(2.5, -0.5)
    ax2.set_aspect('equal')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(displayLabels, fontsize=7)
    ax2.set_yticklabels(displayLabels, fontsize=7)
    ax2.set_xlabel('Predicted class')
    ax2.set_ylabel('True class')
    ax2.set_title(f'3-class accuracy: {metrics["threeClassAccuracy"]:.2f}', fontsize=8, pad=8)
    plt.tight_layout(w_pad=2.0)
    plt.savefig(savePath / 'fig1_training_performance.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved fig1_training_performance')


def plotTestPrediction(testSignalDict, predictions, savePath):
    _applyStyle()
    totalTime = testSignalDict['BloodPressure'].iloc[:, 0].values[-1]
    predDf = pd.DataFrame(predictions)
    # 2x2 grid for biomarkers (square), full-width z-score + color bar below
    fig = plt.figure(figsize=(DoubleColInch, 9.0))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1.2, 0.25], hspace=0.35, wspace=0.3)

    biomarkers = [
        ('BloodPressure', 'BP', 'Blood pressure (mmHg)', 'a'),
        ('HeartRate', 'HR', 'Heart rate (BPM)', 'b'),
        ('Glucose', 'Gluc', 'Glucose (mg/dL)', 'c'),
        ('BreathingPeriod', 'Br', 'Breathing period (s)', 'd'),
    ]

    for idx, (key, shortKey, ylabel, panelLabel) in enumerate(biomarkers):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.set_box_aspect(1)
        _addPanelLabel(ax, panelLabel)
        df = testSignalDict[key]
        t = df.iloc[:, 0].values
        v = df.iloc[:, 1].values
        ax.plot(t, v, color=Colors[shortKey], alpha=0.15, linewidth=0.3)
        windowSize = max(len(v) // 30, 3)
        smoothed = pd.Series(v).rolling(windowSize, center=True).mean()
        ax.plot(t, smoothed, color=Colors[shortKey], linewidth=1.0)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlim(0, totalTime)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (s)', fontsize=7)

    axZ = fig.add_subplot(gs[2, :])
    axZ.set_box_aspect(1)
    _addPanelLabel(axZ, 'e')
    tCenters = predDf['tCenter'].values
    zClipped = predDf['zClipped'].values
    zSmoothed = predDf['zSmoothed'].values
    axZ.scatter(tCenters, zClipped, s=8, color=Colors['pred_raw'], alpha=0.5, edgecolors='none', zorder=2, label='Raw')
    axZ.plot(tCenters, zSmoothed, '-', color=Colors['pred_smooth'], linewidth=1.5, zorder=3, label='Median-smoothed')
    axZ.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.5)
    axZ.axhline(-2, color=Colors['highStress'], linewidth=0.5, linestyle=':', alpha=0.5)
    axZ.set_ylabel('Predicted z-score')
    allZ = np.concatenate([zClipped, zSmoothed])
    zMargin = (allZ.max() - allZ.min()) * 0.1
    axZ.set_ylim(allZ.min() - zMargin, allZ.max() + zMargin)
    axZ.set_xlim(0, totalTime)
    axZ.tick_params(labelbottom=False)
    axZ.legend(loc='upper right', frameon=False, ncol=2)

    axBar = fig.add_subplot(gs[3, :])
    _addPanelLabel(axBar, 'f', y=1.3)
    for _, row in predDf.iterrows():
        z = row['zSmoothed']
        if z >= 0:
            c = Colors['healthy']
        elif z > -2:
            c = Colors['lowStress']
        else:
            c = Colors['highStress']
        axBar.barh(0, row['tEnd'] - row['tStart'], left=row['tStart'], height=0.6, color=c, alpha=0.85, edgecolor='none', linewidth=0)
    axBar.set_xlim(0, totalTime)
    axBar.set_ylim(-0.5, 0.5)
    axBar.set_yticks([])
    axBar.set_xlabel('Time (s)')
    barLegend = [
        Patch(facecolor=Colors['healthy'], edgecolor='none', label='Healthy'),
        Patch(facecolor=Colors['lowStress'], edgecolor='none', label='Low stress'),
        Patch(facecolor=Colors['highStress'], edgecolor='none', label='High stress'),
    ]
    axBar.legend(handles=barLegend, loc='center left', fontsize=6, ncol=3, frameon=False)
    plt.savefig(savePath / 'fig2_test_prediction.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved fig2_test_prediction')


def plotFeatureImportance(X, y, model, scaler, savePath):
    _applyStyle()

    fig, axes = plt.subplots(1, 2, figsize=(DoubleColInch, 3.5), gridspec_kw={'width_ratios': [1, 1]})
    ax = axes[0]
    _addPanelLabel(ax, 'a')
    featureNames = list(X.columns)
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
    elif hasattr(model, 'feature_importances_'):
        coefficients = model.feature_importances_
    else:
        if hasattr(model, 'dual_coef_') and hasattr(model, 'support_vectors_'):
            coefficients = np.dot(model.dual_coef_, model.support_vectors_).flatten()
        else:
            coefficients = np.zeros(len(featureNames))
    topK = min(15, len(coefficients))
    idxSorted = np.argsort(np.abs(coefficients))[-topK:]
    barColors = [Colors['healthy'] if coefficients[i] > 0 else Colors['highStress'] for i in idxSorted]
    displayNames = [_cleanName(featureNames[i]) for i in idxSorted]
    ax.barh(range(topK), coefficients[idxSorted], color=barColors, edgecolor='none', linewidth=0, height=0.7)
    ax.set_yticks(range(topK))
    ax.set_yticklabels(displayNames, fontsize=6)
    ax.set_xlabel('Coefficient (scaled features)')
    ax.axvline(0, color='black', linewidth=0.4)
    ax.set_title('Feature weights', fontsize=8, pad=6)

    featureLegend = [
        Patch(facecolor=Colors['healthy'], edgecolor='none',
              label='Higher \u2192 healthier'),
        Patch(facecolor=Colors['highStress'], edgecolor='none',
              label='Higher \u2192 more stressed'),
    ]
    ax.legend(handles=featureLegend, loc='lower right', fontsize=6, frameon=False)

    ax2 = axes[1]
    _addPanelLabel(ax2, 'b')

    correlations = []
    for col in X.columns:
        vals = X[col].values.astype(float)
        if np.std(vals) == 0:
            r = 0.0
        else:
            r = np.corrcoef(vals, y)[0, 1]
        correlations.append((col, r))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    topCorr = correlations[:topK]

    names = []
    rVals = []
    for name, r in reversed(topCorr):
        names.append(_cleanName(name))
        rVals.append(r)

    barColors2 = [Colors['healthy'] if r > 0 else Colors['highStress'] for r in rVals]
    ax2.barh(range(len(names)), rVals, color=barColors2, edgecolor='none', linewidth=0, height=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel('Pearson r with z-score')
    ax2.axvline(0, color='black', linewidth=0.4)
    ax2.set_title('Univariate correlations', fontsize=8, pad=6)
    plt.tight_layout(w_pad=1.5)
    plt.savefig(savePath / 'fig3_feature_analysis.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('  Saved fig3_feature_analysis')


def plotModelComparison(allResults, savePath):
    _applyStyle()
    modelNames = list(allResults.keys())
    nModels = len(modelNames)
    fig, axes = plt.subplots(1, 4, figsize=(DoubleColInch, 2.5), sharey=True)
    metricKeys = ['MAE', 'r', 'r2', 'signAccuracy']
    metricLabels = ['MAE \u2193', 'Pearson r \u2191', 'R\u00b2 \u2191', 'Sign accuracy \u2191']
    metricColors = ['#5B2C8E', '#2E86AB', '#2CA58D', '#E8963E']

    for ax, metricKey, metricLabel, color in zip(axes, metricKeys, metricLabels, metricColors):
        values = [allResults[m][metricKey] for m in modelNames]

        if metricKey == 'MAE':
            bestIdx = np.argmin(values)
        else:
            bestIdx = np.argmax(values)

        barColors = [color if i == bestIdx else '#cccccc' for i in range(nModels)]

        bars = ax.bar(range(nModels), values, color=barColors, edgecolor='none', linewidth=0, width=0.7)
        ax.set_xlabel(metricLabel, fontsize=7)

        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    axes[0].set_xticks(range(nModels))
    axes[0].set_xticklabels(modelNames, rotation=45, ha='right', fontsize=6)
    for ax in axes[1:]:
        ax.set_xticks(range(nModels))
        ax.set_xticklabels(modelNames, rotation=45, ha='right', fontsize=6)
    plt.tight_layout(w_pad=0.5)
    plt.savefig(savePath / 'fig4_model_comparison.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved fig4_model_comparison')


def plotPerSampleAnalysis(y, predictions, meta, savePath):
    _applyStyle()
    fig, axes = plt.subplots(2, 1, figsize=(DoubleColInch, 4.0), gridspec_kw={'height_ratios': [1.2, 1]})
    ax = axes[0]
    _addPanelLabel(ax, 'a')
    nSamples = len(y)
    sortIdx = np.argsort(y)
    ySort = y[sortIdx]
    predSort = predictions[sortIdx]

    if meta is not None and 'file' in meta.columns:
        fileLabels = [Path(meta.iloc[i]['file']).stem for i in sortIdx]
        shortLabels = []
        for fl in fileLabels:
            parts = fl.split('_')
            rat = parts[1].replace('rat', 'r')
            cond = parts[2][0]
            shortLabels.append(f'{rat}_{cond}')
    else:
        shortLabels = [str(i) for i in range(nSamples)]

    x = np.arange(nSamples)
    ax.bar(x, ySort, width=0.35, align='edge', color=Colors['healthy'], alpha=0.6, edgecolor='none', linewidth=0, label='True z-score')
    ax.bar(x + 0.35, predSort, width=0.35, align='edge', color=Colors['pred_smooth'], alpha=0.6, edgecolor='none', linewidth=0, label='Predicted z-score')
    ax.axhline(0, color='black', linewidth=0.4)
    ax.set_xticks(x + 0.35)
    ax.set_xticklabels(shortLabels, rotation=90, fontsize=5)
    ax.set_ylabel('Z-score')
    ax.set_title('Per-sample LOOCV predictions (sorted by true z-score)', fontsize=8)
    ax.legend(fontsize=6, loc='upper left', frameon=False)

    ax2 = axes[1]
    _addPanelLabel(ax2, 'b')
    colors = [Colors['scatter_healthy'] if yi >= 0 else Colors['scatter_stressed'] for yi in y]
    ax2.scatter(y, predictions - y, c=colors, s=20, edgecolors='none', linewidth=0, zorder=3)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('True z-score')
    ax2.set_ylabel('Prediction error (pred \u2212 true)')
    ax2.set_title('Residuals vs. true z-score', fontsize=8)

    meanError = np.mean(predictions - y)
    ax2.text(0.95, 0.05, f'Mean bias: {meanError:+.2f}', transform=ax2.transAxes, fontsize=6, ha='right', va='bottom')
    plt.tight_layout(h_pad=1.5)
    plt.savefig(savePath / 'fig5_per_sample_analysis.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved fig5_per_sample_analysis')


def plotShapBeeswarmComparison(X, y, modelDict, savePath, maxDisplay=15):
    _applyStyle()
    nModels = len(modelDict)
    fig, axes = plt.subplots(1, nModels, figsize=(DoubleColInch + 3.0, 4.5), sharey=True)
    if nModels == 1:
        axes = [axes]

    cleanNames = [_cleanName(c) for c in X.columns]
    allMeanShap = np.zeros(len(cleanNames))

    modelShapData = {}
    for modelName, modelTemplate in modelDict.items():
        scaler = _SS()
        X_scaled = scaler.fit_transform(X.values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=cleanNames)
        model = _clone(modelTemplate)
        model.fit(X_scaled, y)
        explainer = _getShapExplainer(model, X_scaled_df)
        sv = explainer.shap_values(X_scaled_df)
        meanAbs = np.mean(np.abs(sv), axis=0)
        allMeanShap += meanAbs
        modelShapData[modelName] = {
            'shapValues': sv,
            'X_scaled_df': X_scaled_df,
            'meanAbs': meanAbs,
        }

    globalOrder = np.argsort(allMeanShap)[::-1][:maxDisplay][::-1]
    panelLabels = 'abcdefghij'
    for axIdx, (ax, (modelName, data)) in enumerate(zip(axes, modelShapData.items())):
        _addPanelLabel(ax, panelLabels[axIdx], x=-0.05 if axIdx > 0 else -0.15)
        sv = data['shapValues']
        X_df = data['X_scaled_df']
        meanAbs = data['meanAbs']
        for rank, featIdx in enumerate(globalOrder):
            shapCol = sv[:, featIdx]
            featVals = X_df.iloc[:, featIdx].values
            fMin, fMax = featVals.min(), featVals.max()
            normVals = (featVals - fMin) / (fMax - fMin + 1e-8)
            jitter = np.random.RandomState(42).normal(0, 0.15, size=len(shapCol))
            scatter = ax.scatter(shapCol, np.full_like(shapCol, rank) + jitter, c=normVals, cmap='coolwarm', s=10, alpha=0.75, edgecolors='none', vmin=0, vmax=1, zorder=3)
        for rank, featIdx in enumerate(globalOrder):
            barVal = meanAbs[featIdx]
            maxBar = meanAbs[globalOrder].max()
            barWidth = barVal / (maxBar + 1e-8) * 0.4
            ax.barh(rank, barWidth, left=ax.get_xlim()[0] if ax.get_xlim()[0] < 0 else 0, height=0.8, color='#f0f0f0', zorder=0, edgecolor='none')
        ax.axvline(0, color='black', linewidth=0.4)
        ax.set_title(modelName, fontsize=8, pad=6)
        ax.set_xlabel('SHAP value', fontsize=7)
        if axIdx == 0:
            ax.set_yticks(range(len(globalOrder)))
            ax.set_yticklabels([cleanNames[i] for i in globalOrder], fontsize=5.5)
        else:
            ax.tick_params(labelleft=False)
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.4, pad=0.02, aspect=30)
    cbar.set_label('Feature value', fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=6)
    cbar.ax.tick_params(width=0.3, length=2)
    plt.savefig(savePath / 'shap_beeswarm_comparison.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved shap_beeswarm_comparison')


def plotShapDecision(X, y, model, scaler, savePath):
    _applyStyle()
    X_scaled_df, shapValues, explainer, cleanNames = _buildShapData(X, y, model, scaler)
    fig, ax = plt.subplots(1, 1, figsize=(SingleColInch + 1.0, 5.0))
    meanAbs = np.mean(np.abs(shapValues), axis=0)
    featOrder = np.argsort(meanAbs)
    plt.sca(ax)
    shap.decision_plot(explainer.expected_value, shapValues, X_scaled_df, feature_names=cleanNames, feature_order=featOrder, show=False, color_bar=False)
    zNorm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    cmap = plt.cm.coolwarm
    lineObjs = [child for child in ax.get_children() if isinstance(child, plt.Line2D) and len(child.get_xdata()) > 1]
    for i, line in enumerate(lineObjs[:len(y)]):
        line.set_color(cmap(zNorm[i]))
        line.set_alpha(0.7)
        line.set_linewidth(0.8)
    ax.set_title('SHAP decision plot', fontsize=8, pad=8)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=y.min(), vmax=y.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label('True z-score', fontsize=6)
    cbar.ax.tick_params(labelsize=5, width=0.3, length=2)
    plt.tight_layout()
    plt.savefig(savePath / 'shap_decision.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved shap_decision')


def plotShap(X, y, model, scaler, modelDict, savePath):
    savePath = Path(savePath)
    savePath.mkdir(parents=True, exist_ok=True)
    print('Generating SHAP figures...')
    plotShapBeeswarmComparison(X, y, modelDict, savePath)
    plotShapDecision(X, y, model, scaler, savePath)
    print(f'All SHAP figures saved to {savePath}')


def plotAll(y, predictions, metrics, meta, testSignalDict, testPredictions, allResults, model, scaler, X, savePath):
    savePath = Path(savePath)
    savePath.mkdir(parents=True, exist_ok=True)
    print('Generating figures...')
    plotTrainingPerformance(y, predictions, metrics, savePath)
    plotTestPrediction(testSignalDict, testPredictions, savePath)
    plotFeatureImportance(X, y, model, scaler, savePath)
    plotModelComparison(allResults, savePath)
    plotPerSampleAnalysis(y, predictions, meta, savePath)
    print(f'All figures saved to {savePath}')