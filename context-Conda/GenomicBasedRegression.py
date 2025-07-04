# required librairies
## pip3.12 install --force-reinstall catboost==1.2.8
## pip3.12 install --force-reinstall pandas==2.2.2
## pip3.12 install --force-reinstall xgboost==2.1.3
## pip3.12 install --force-reinstall lightgbm==4.6.0
## pip3.12 install --force-reinstall boruta==0.4.3
## pip3.12 install --force-reinstall scipy==1.16.0
## pip3.12 install --force-reinstall scikit-learn==1.5.2
## pip3.12 install --force-reinstall numpy==1.26.4
## pip3.12 install --force-reinstall joblib==1.5.1
## pip3.12 install --force-reinstall tqdm==4.67.1
## pip3.12 install --force-reinstall tqdm-joblib==0.0.4
'''
# examples of commands with parameters
## for the BRI regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x BRI_FirstAnalysis -da random -sp 80 -q 10 -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/BRI_FirstAnalysis_features.obj -fe MyDirectory/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectory/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectory/BRI_FirstAnalysis_model.obj -o MyDirectory -x BRI_SecondAnalysis -w -de 20
## for the CB regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CB_FirstAnalysis -da manual -fs SKB -r CB -k 5 -pa tuning_parameters_CB.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/CB_FirstAnalysis_features.obj -fe MyDirectory/CB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/CB_FirstAnalysis_calibration_features.obj -ct MyDirectory/CB_FirstAnalysis_calibration_targets.obj -t MyDirectory/CB_FirstAnalysis_model.obj -o MyDirectory -x CB_SecondAnalysis -w -de 20
## for the DT regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs laSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/DT_FirstAnalysis_calibration_features.obj -ct MyDirectory/DT_FirstAnalysis_calibration_targets.obj -t MyDirectory/DT_FirstAnalysis_model.obj -o MyDirectory -x DT_SecondAnalysis -w -de 20
## for the EN regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs enSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/EN_FirstAnalysis_calibration_features.obj -ct MyDirectory/EN_FirstAnalysis_calibration_targets.obj -t MyDirectory/EN_FirstAnalysis_model.obj -o MyDirectory -x EN_SecondAnalysis -w -de 20
## for the GB regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs rfSFM -r GB -k 5 -pa tuning_parameters_GB.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/GB_FirstAnalysis_calibration_features.obj -ct MyDirectory/GB_FirstAnalysis_calibration_targets.obj -t MyDirectory/GB_FirstAnalysis_model.obj -o MyDirectory -x GB_SecondAnalysis -w -de 20
## for the HGB regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs BO -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/HGB_FirstAnalysis_model.obj -o MyDirectory -x HGB_SecondAnalysis -w -de 20
## for the HU regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HU_FirstAnalysis -da manual -fs BO -r HU -k 5 -pa tuning_parameters_HU.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HU_FirstAnalysis_features.obj -fe MyDirectory/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HU_FirstAnalysis_calibration_features.obj -ct MyDirectory/HU_FirstAnalysis_calibration_targets.obj -t MyDirectory/HU_FirstAnalysis_model.obj -o MyDirectory -x HU_SecondAnalysis -w -de 20
## for the KNN regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/KNN_FirstAnalysis_features.obj -fe MyDirectory/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectory/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectory/KNN_FirstAnalysis_model.obj -o MyDirectory -x KNN_SecondAnalysis -w -de 20
## for the LA regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LA_FirstAnalysis_features.obj -fe MyDirectory/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LA_FirstAnalysis_calibration_features.obj -ct MyDirectory/LA_FirstAnalysis_calibration_targets.obj -t MyDirectory/LA_FirstAnalysis_model.obj -o MyDirectory -x LA_SecondAnalysis -w -de 20
## for the LGBM regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectory/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectory/LGBM_FirstAnalysis_model.obj -o MyDirectory -x LGBM_SecondAnalysis -w -de 20
## for the MLP regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/MLP_FirstAnalysis_features.obj -fe MyDirectory/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectory/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectory/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectory/MLP_FirstAnalysis_model.obj -o MyDirectory -x MLP_SecondAnalysis -w -de 20
## for the NSV regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/NSV_FirstAnalysis_features.obj -fe MyDirectory/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectory/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectory/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectory/NSV_FirstAnalysis_model.obj -o MyDirectory -x NSV_SecondAnalysis -w -de 20
## for the PN regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x PN_FirstAnalysis -da manual -fs laSFM -r PN -k 5 -pa tuning_parameters_PN.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/PN_FirstAnalysis_features.obj -fe MyDirectory/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/PN_FirstAnalysis_calibration_features.obj -ct MyDirectory/PN_FirstAnalysis_calibration_targets.obj -t MyDirectory/PN_FirstAnalysis_model.obj -o MyDirectory -x PN_SecondAnalysis -w -de 20
## for the RI regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RI_FirstAnalysis_features.obj -fe MyDirectory/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RI_FirstAnalysis_calibration_features.obj -ct MyDirectory/RI_FirstAnalysis_calibration_targets.obj -t MyDirectory/RI_FirstAnalysis_model.obj -o MyDirectory -x RI_SecondAnalysis -w -de 20
## for the RF regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RF_FirstAnalysis_features.obj -fe MyDirectory/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RF_FirstAnalysis_calibration_features.obj -ct MyDirectory/RF_FirstAnalysis_calibration_targets.obj -t MyDirectory/RF_FirstAnalysis_model.obj -o MyDirectory -x RF_SecondAnalysis -w -de 20
## for the SVR regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/SVR_FirstAnalysis_features.obj -fe MyDirectory/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectory/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectory/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectory/SVR_FirstAnalysis_model.obj -o MyDirectory -x SVR_SecondAnalysis -w -de 20
## for the XGB regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi -w -de 20
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/XGB_FirstAnalysis_model.obj -o MyDirectory -x XGB_SecondAnalysis -w -de 20
'''
# import packages
## standard libraries
import sys as sys # no individual installation because is part of the Python Standard Library (no version)
import os as os # no individual installation because is part of the Python Standard Library (no version)
import datetime as dt # no individual installation because is part of the Python Standard Library (no version)
import argparse as ap # no individual installation because is part of the Python Standard Library (with version)
import pickle as pi # no individual installation because is part of the Python Standard Library (with version)
import warnings as wa # no individual installation because is part of the Python Standard Library (no version)
import re as re # no individual installation because is part of the Python Standard Library (with version)
import threading as th # no individual installation because is part of the Python Standard Library (no version)
import time as ti # no individual installation because is part of the Python Standard Library (no version)
import importlib.metadata as im # no individual installation because is part of the Python Standard Library (no version)
import functools as ft # no individual installation because is part of the Python Standard Library (no version)
## third-party libraries
import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sp
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import boruta as bo
import joblib as jl
import tqdm as tq
import tqdm.auto as tqa # no version because it corresponds a tqdm module
import tqdm_joblib as tqjl
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ParameterGrid
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, BayesianRidge, HuberRegressor, Lasso
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, SelectFromModel
from sklearn import set_config
from sklearn.inspection import permutation_importance

# set static metadata to keep outside the main function
## set workflow repositories
repositories = 'Please cite:\n GitHub (https://github.com/Nicolas-Radomski/GenomicBasedRegression),\n Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedregression),\n and/or Anaconda Hub (https://anaconda.org/nicolasradomski/genomicbasedregression).'
## set the workflow context
context = "The scikit-learn (sklearn)-based Python workflow independently supports both modeling (i.e., training and testing) and prediction (i.e., using a pre-built model), and implements 5 feature selection methods, 17 model regressors, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction interval estimation, execution monitoring via progress bars, and parallel processing."
## set the workflow reference
reference = "An article might potentially be published in the future."
## set the acknowledgement
acknowledgements = "Many thanks to Andrea De Ruvo, Adriano Di Pasquale and ChatGPT for the insightful discussions that helped improve the algorithm."
## set the version and release
__version__ = "1.1.0"
__release__ = "July 2025"

# set global sklearn config early
set_config(transform_output="pandas")

# define functions of interest

def smape(y_true, y_pred, threshold=1e-3):
	"""
	compute symmetric mean absolute percentage error (SMAPE) with thresholding for near-zero denominators
	parameters:
		y_true: array-like of true target values
		y_pred: array-like of predicted values
		threshold: float, default=1e-3, values with average magnitude below this are excluded to avoid instability, override for greater control in sensitive cases
	returns:
		float: SMAPE as a ratio (e.g., 0.0823), or np.nan if no valid values remain
	"""
	y_true_nda, y_pred_nda = np.array(y_true).ravel(), np.array(y_pred).ravel() # convert inputs to flattened ndarray to ensure compatibility
	denominator = (np.abs(y_true_nda) + np.abs(y_pred_nda)) / 2 # compute denominator: average of the absolute values of true and predicted values
	diff = np.abs(y_true_nda - y_pred_nda) # compute numerator: absolute difference between true and predicted values
	mask = denominator > threshold # create a boolean mask to select only valid values (denominator > threshold)
	if np.sum(mask) == 0: # return NaN if no valid values remain because SMAPE cannot be safely calculated
 		return np.nan
	return np.mean(diff[mask] / denominator[mask]) # calculate SMAPE on the filtered values where denominator is not zero, nor near-zero

def mape(y_true, y_pred, threshold=1e-3):
	"""
	compute mean absolute percentage error (MAPE), using mean_absolute_percentage_error and excluding near-zero targets to avoid inflation
	parameters:
		y_true: array-like of true target values
		y_pred: array-like of predicted values
		threshold: float, default=1e-3, values of |y_true| below this are excluded from the computation
	returns:
		float: MAPE as a ratio (e.g., 0.0872), or np.nan if no valid targets remain
	"""
	y_true_nda, y_pred_nda = np.array(y_true).ravel(), np.array(y_pred).ravel() # convert inputs to flattened ndarray to ensure compatibility
	mask = np.abs(y_true_nda) > threshold # create a boolean mask to exclude near-zero target values
	if np.sum(mask) == 0: # return NaN to indicate MAPE is undefined if no valid values remain after masking
		return np.nan
	return mean_absolute_percentage_error(y_true_nda[mask], y_pred_nda[mask]) # compute MAPE only on the valid (masked) subset of the data

def adjusted_r2(y_true, y_pred, n_features):
	"""
	compute adjusted R-squared (aR²), which adjusts the R² score based on the number of predictors used
	parameters:
		y_true: array-like of true target values
		y_pred: array-like of predicted values
		n_features: int, number of predictors (independent variables) used in the model
	returns:
		float: adjusted R-squared, or np.nan if computation is invalid
	"""
	y_true_nda, y_pred_nda = np.array(y_true).ravel(), np.array(y_pred).ravel() # convert inputs to flattened ndarray to ensure compatibility
	n = len(y_true_nda) # number of samples
	if n <= n_features + 1: # check for valid degrees of freedom
		return np.nan  # adjusted R² cannot be computed safely in this case
	r2 = r2_score(y_true_nda, y_pred_nda)  # compute the regular R-squared score
	# compute adjusted R² using the standard formula: adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
	adjusted = 1 - (1 - r2) * ((n - 1) / (n - n_features - 1))
	return adjusted  # return the adjusted R² value

def count_selected_features(pipeline: Pipeline, encoded_matrix):
	"""
	robust count of features the pipeline expects
	return the number of columns reaching the final estimator
	"""
	if 'feature_selection' in pipeline.named_steps:
		fs = pipeline.named_steps['feature_selection']
		if hasattr(fs, "support_"):
			return int(np.sum(fs.support_))          # fast & reliable
		else:                                       # very rare fallback
			return fs.transform(encoded_matrix[:1]).shape[1]
	# no explicit selector → try the estimator attribute
	est = pipeline.named_steps.get('model', pipeline)
	n_feat = getattr(est, "n_features_in_", None)
	if n_feat is None or n_feat == 0:               # CatBoost case
		n_feat = encoded_matrix.shape[1]            # use full width
	return int(n_feat)

def restricted_float_split(x: str) -> float:
	"""
	convert *x* to float and ensure 0 < x < 100
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as float or is not in (0, 100)
	"""
	try:
		x = float(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid float")
	if not (0.0 < x < 100.0):
		raise ap.ArgumentTypeError("split must be a float in the open interval (0, 100)")
	return x

def restricted_int_quantiles(x: str) -> int:
	"""
	convert *x* to int and ensure x >= 2
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 2
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 2:
		raise ap.ArgumentTypeError("quantiles must be an integer ≥ 2")
	return x

def restricted_int_limit(x: str) -> int:
	"""
	convert *x* to int and ensure x >= 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 1:
		raise ap.ArgumentTypeError("limit must be an integer ≥ 1")
	return x

def restricted_int_fold(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 2
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 2
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 2:
		raise ap.ArgumentTypeError("fold must be an integer ≥ 2 for cross-validation")
	return x

def restricted_int_jobs(x: str) -> int:
	"""
	convert *x* to int and ensure x == -1 or x ≥ 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is not -1 or ≥ 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x != -1 and x < 1:
		raise ap.ArgumentTypeError("jobs must be -1 (all CPUs) or an integer ≥ 1")
	return x

def restricted_int_nrepeats(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 1:
		raise ap.ArgumentTypeError("nrepeats must be an integer ≥ 1 for permutation importance")
	return x

def restricted_float_alpha(x: str) -> float:
	"""
	convert *x* to float and ensure 0 < x < 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as float or is not in (0, 1)
	"""
	try:
		x = float(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid float")
	if not (0.0 < x < 1.0):
		raise ap.ArgumentTypeError("alpha must be in the open interval (0, 1)")
	return x

# ---------- ResidualQuantileWrapper ----------
# import numpy as np # arrays
# from sklearn.base import BaseEstimator, RegressorMixin # Import base classes from Scikit-learn to ensure compatibility with its utilities (e.g., cross-validation, cloning, pipelines)
class ResidualQuantileWrapper(BaseEstimator, RegressorMixin):
	"""
	wrapper class to compute prediction intervals based on residual quantiles
	around a fitted regression estimator
	this is a custom implementation similar in spirit to MAPIE's ResidualQuantileWrapper
	parameters
	----------
	estimator : object
		a regression estimator implementing fit and predict methods

	alpha : float, default=0.05
		significance level for prediction intervals (e.g., 0.05 for 95% intervals)

	prefit : bool, default=False
		if True, assumes that the estimator has already been fitted externally
		and skips refitting during wrapper training.
	"""
	def __init__(self, estimator, alpha=0.05, prefit=False):
		self.estimator = estimator      # underlying regression model
		self.alpha = alpha              # confidence level (significance)
		self.prefit = prefit            # flag to avoid retraining an already fitted model
		self.lower_quantile = None      # to store residual lower quantile threshold
		self.upper_quantile = None      # to store residual upper quantile threshold
	def fit(self, X, y):
		"""
		fit the estimator and compute residual quantiles on training (or calibration) data
		parameters
		----------
		X : array-like of shape (n_samples, n_features)
			training or calibration features.
		y : array-like of shape (n_samples,) or (n_samples, 1)
			target values.
		returns
		-------
		self : object
			returns self for chaining.
		"""
		# check that there are enough samples to estimate quantiles
		if len(X) < 2:
			raise ValueError("ResidualQuantileWrapper requires at least 2 calibration samples to compute prediction intervals.")
		# fit the underlying regression model only if not already trained
		if not self.prefit:
			self.estimator.fit(X, y)
		# ensure target is 1D to align with predicted values
		y_1d = y.squeeze() if hasattr(y, "squeeze") else np.ravel(y)
		# calculate residuals as absolute differences
		residuals = np.abs(y_1d - self.estimator.predict(X))
		# compute residual quantiles to define interval width
		self.lower_quantile = np.quantile(residuals, self.alpha / 2)
		self.upper_quantile = np.quantile(residuals, 1 - self.alpha / 2)
		# optional print/log for debugging
		#print(f"[ResidualQuantileWrapper] Residual quantile bounds set to ±{self.upper_quantile:.4f} for α = {self.alpha}")
		return self
	def predict(self, X, return_prediction_interval=False):
		"""
		predict point estimates and optionally prediction intervals for new data.
		parameters
		----------
		X : array-like of shape (n_samples, n_features)
			input features
		return_prediction_interval : bool, default=False
			if True, also return prediction intervals (lower and upper bounds)
		returns
		-------
		y_pred : ndarray of shape (n_samples,)
			predicted target values
		y_pred_intervals : ndarray of shape (n_samples, 2), optional
			prediction intervals with columns [lower, upper], returned only if
			return_prediction_interval=True.
		"""
		# compute point predictions using the underlying model
		y_pred = self.estimator.predict(X)
		if return_prediction_interval:
			# construct prediction intervals using stored residual quantiles
			lower_bounds = y_pred - self.upper_quantile
			upper_bounds = y_pred + self.upper_quantile
			prediction_intervals = np.vstack((lower_bounds, upper_bounds)).T
			return y_pred, prediction_intervals
		return y_pred  # point predictions only
	def get_params(self, deep=True):
		"""
		get parameters for this estimator. Required for sklearn compatibility
		parameters
		----------
		deep : bool, default=True
			if True, will return the parameters for this estimator and contained subobjects
		returns
		-------
		params : dict
			parameter names mapped to their values.
		"""
		return {
			"estimator": self.estimator,
			"alpha": self.alpha,
			"prefit": self.prefit
		}
	def set_params(self, **params):
		"""
		set the parameters of this estimator. Required for sklearn compatibility.
		parameters
		----------
		**params : dict
			estimator parameters.
		returns
		-------
		self : object
			estimator instance.
		"""
		for key, value in params.items():
			setattr(self, key, value)
		return self

# ---------- BorutaSelectorDF ----------
# import boruta as bo # BorutaPy wrapper
# import numpy as np # arrays
# import pandas as pd # DataFrame support
# import threading as th # alias for threading
# import time as ti # alias for time
# import tqdm.auto as tqa # adaptive tqdm
# from sklearn.base import BaseEstimator, TransformerMixin # base classes for custom estimators and transformers compatible with sklearn pipelines
class BorutaSelectorDF(BaseEstimator, TransformerMixin):  # subclass for sklearn compatibility
	"""
	Boruta feature selector with nested tqdm bars that keep both inner and outer bars informative.
	compatible with sklearn Pipelines and GridSearchCV.
	"""
	""" -------------------- INIT -------------------- """
	def __init__(self, estimator, show_progress=True, **kwargs):  # constructor
		self.estimator = estimator  # base estimator used by Boruta
		self.kwargs = kwargs  # BorutaPy‑specific kwargs
		self.show_progress = show_progress  # toggle progress
		self.boruta = None  # will hold BorutaPy instance
		self.support_ = None  # mask of selected features after fit
		self.ranking_ = None  # ranking of all features after fit
		self.columns_ = None  # column names if X is a DataFrame
	""" ---------------- INTERNAL: Fit with tqdm ---------------- """
	def _fit_with_progress(self, X_np, y):  # helper: run Boruta with manual progress bar
		kwargs = self.kwargs.copy()  # copy kwargs to avoid side‑effects
		kwargs.pop("verbose", None)  # remove verbose to silence BorutaPy
		self.boruta = bo.BorutaPy(  # instantiate BorutaPy
			estimator=self.estimator,
			verbose=0,  # silence BorutaPy internal verbose
			**kwargs
		)
		total_iter = self.boruta.max_iter  # total iterations Boruta will run
		with tqa.tqdm(total=total_iter, desc="Boruta iterations",  # use tqdm.auto alias tqa
					position=1, leave=False, dynamic_ncols=True) as pbar:
			# Run Boruta fit in a separate thread to keep UI responsive
			def target():
				self.boruta.fit(X_np, y)  # actual fitting
			thread = th.Thread(target=target)  # create thread (alias th)
			thread.start()
			while thread.is_alive():  # update bar while running
				if pbar.n < total_iter:
					pbar.update(1)
				else:
					pbar.n = total_iter  # cap progress bar if exceeded
				pbar.refresh()
				ti.sleep(0.5)  # sleep (alias ti)
			thread.join()  # ensure completion
	""" -------------------- FIT -------------------- """
	def fit(self, X, y):  # standard sklearn fit
		if isinstance(X, pd.DataFrame):  # handle DataFrame input
			self.columns_ = X.columns  # save column names
			X_np = X.values  # convert to NumPy
		else:
			X_np = np.asarray(X)  # ensure NumPy

		if self.show_progress:  # choose verbose fit
			self._fit_with_progress(X_np, y)  # run with manual progress bar
		else:
			self.boruta = bo.BorutaPy(estimator=self.estimator, **self.kwargs)  # silent Boruta
			self.boruta.fit(X_np, y)  # fit without bars
		self.support_ = self.boruta.support_  # store mask
		self.ranking_ = self.boruta.ranking_  # store rankings
		return self  # return self for chaining
	""" ------------------ TRANSFORM ------------------ """
	def transform(self, X):  # reduce X to selected features
		if isinstance(X, pd.DataFrame):  # DataFrame input
			return X.loc[:, self.support_]  # mask columns
		return X[:, self.support_]  # mask NumPy array
	""" --------------- PARAMETER GETTER --------------- """
	def get_params(self, deep=True):  # expose params for grid search
		params = {"estimator": self.estimator, "show_progress": self.show_progress}  # base params
		if deep and hasattr(self.estimator, "get_params"):  # include nested estimator params
			for k, v in self.estimator.get_params().items():
				params[f"estimator__{k}"] = v  # flatten nested params
		params.update(self.kwargs)  # add BorutaPy kwargs
		return params  # return full param dict

# create a main function preventing the global scope from being unintentionally executed on import
def main():

	# step control
	step1_start = dt.datetime.now()

	# create the main parser
	parser = ap.ArgumentParser(
		prog="GenomicBasedRegression.py", 
		description="Perform regression-based modeling or prediction from binary (e.g., presence/absence of genes) or categorical (e.g., allele profiles) genomic data.",
		epilog=repositories
		)

	# create subparsers object
	subparsers = parser.add_subparsers(dest='subcommand')

	# create the parser for the "modeling" subcommand
	## get parser arguments
	parser_modeling = subparsers.add_parser('modeling', help='Help about the model building.')
	## define parser arguments
	parser_modeling.add_argument(
		'-m', '--mutations', 
		dest='inputpath_mutations', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_modeling.add_argument(
		'-ph', '--phenotypes', 
		dest='inputpath_phenotypes', 
		action='store', 
		required=True, 
		help="Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample identifiers identical to those in the input file of mutations (header: e.g., sample). Second column: categorical phenotype (header: e.g., phenotype). Third column: 'training' or 'testing' dataset (header: e.g., dataset). [MANDATORY]"
		)
	parser_modeling.add_argument(
		'-da', '--dataset', 
		dest='dataset', 
		type=str,
		action='store', 
		required=False, 
		choices=['random', 'manual'], 
		default='random', 
		help="Perform random (i.e., 'random') or manual (i.e., 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL, DEFAULT: 'random']"
		)
	parser_modeling.add_argument(
		'-sp', '--split', 
		dest='splitting', 
		type=restricted_float_split, # control (0, 100) open interval
		action='store', 
		required=False, 
		default=None, 
		help='Percentage of random splitting when preparing the training dataset using the holdout method. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-q', '--quantiles', 
		dest='quantiles', 
		type=restricted_int_quantiles, # control >= 2
		action='store', 
		required=False, 
		default=None, 
		help='Number of quantiles used to discretize the phenotype values when preparing the training dataset using the holdout method. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-l', '--limit', 
		dest='limit', 
		type=restricted_int_limit, # control >= 1
		action='store', 
		required=False, 
		default=10, 
		help='Recommended minimum number of samples in both the training and testing datasets to reliably estimate performance metrics. [OPTIONAL, DEFAULT: 10]'
		)
	parser_modeling.add_argument(
		'-fs', '--featureselection', 
		dest='featureselection', 
		type=str,
		action='store', 
		required=False, 
		default='None', 
		help='Acronym of the feature selection method to use: SelectKBest (SKB), SelectFromModel with Lasso (laSFM), SelectFromModel with ElasticNet (enSFM), SelectFromModel with Random Forest (rfSFM), or Boruta (BO). Listed in order of increasing computational demand, these methods were chosen for their efficiency, interpretability, and suitability for high-dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-r', '--regressor', 
		dest='regressor', 
		type=str,
		action='store', 
		required=False, 
		default='XGB', 
		help='Acronym of the regressor to use among bayesian bidge (BRI), cat boost (CB), decision tree (DT), elasticnet (EN), gradient boosting (GB), hist gradient boosting (HGB), huber (HU), k-nearest neighbors (KNN), lassa (LA), light gradient goosting machine (LGBM), multi-layer perceptron (MLP), nu support vector (NSV), polynomial (PN), ridge (RI), random forest (RF), support vector regressor (SVR) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]'
		)
	parser_modeling.add_argument(
		'-k', '--fold', 
		dest='fold', 
		type=restricted_int_fold, # control >= 2
		action='store', 
		required=False, 
		default=5, 
		help='Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-validation method. [OPTIONAL, DEFAULT: 5]'
		)
	parser_modeling.add_argument(
		'-pa', '--parameters', 
		dest='parameters', 
		action='store', 
		required=False, 
		help='Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument of the GridSearchCV function. (OPTIONAL)'
		)
	parser_modeling.add_argument(
		'-j', '--jobs', 
		dest='jobs', 
		type=restricted_int_jobs, # control -1 or >= 1
		action='store', 
		required=False, 
		default=-1, 
		help='Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]'
		)
	parser_modeling.add_argument(
		'-pi', '--permutationimportance', 
		dest='permutationimportance', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Compute permutation importance, which can be computationally expensive, especially with many features and/or high repetition counts. [OPTIONAL, DEFAULT: False]'
		)
	parser_modeling.add_argument(
		'-nr', '--nrepeats', 
		dest='nrepeats', 
		type=restricted_int_nrepeats, # control >= 1
		action='store', 
		required=False, 
		default=10, 
		help='Number of repetitions per feature for permutation importance (higher = more stable but slower). [OPTIONAL, DEFAULT: 10]'
		)
	parser_modeling.add_argument(
		'-a', '--alpha', 
		dest='alpha', 
		type=restricted_float_alpha, # control (0, 1) open interval
		action='store', 
		required=False, 
		default=0.05, 
		help='Significance level α (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − α) × 100]%% coverage. [OPTIONAL, DEFAULT: 0.05]'
		)
	parser_modeling.add_argument(
		'-o', '--output', 
		dest='outputpath', 
		action='store', 
		required=False, 
		default='.',
		help='Output path. [OPTIONAL, DEFAULT: .]'
		)
	parser_modeling.add_argument(
		'-x', '--prefix', 
		dest='prefix', 
		action='store', 
		required=False, 
		default='output',
		help='Prefix of output files. [OPTIONAL, DEFAULT: output]'
		)
	parser_modeling.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=int,
		action='store', 
		required=False, 
		default=0, 
		help='Traceback level when an error occurs. (DEFAULT: 0)'
		)
	parser_modeling.add_argument(
		'-w', '--warnings', 
		dest='warnings', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]'
		)
	parser_modeling.add_argument(
		'-nc', '--no-check', 
		dest='nocheck', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]'
		)

	# create the parser for the "prediction" subcommand
	## get parser arguments
	parser_prediction = subparsers.add_parser('prediction', help='Help about the model-based prediction.')
	## define parser arguments
	parser_prediction.add_argument(
		'-m', '--mutations', 
		dest='inputpath_mutations', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-f', '--features', 
		dest='inputpath_features', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including features from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-fe', '--featureencoder', 
		dest='inputpath_feature_encoder', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including encoder from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-cf', '--calibrationfeatures', 
		dest='inputpath_calibration_features', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including calibration features from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-ct', '--calibrationtargets', 
		dest='inputpath_calibration_targets', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including calibration targets from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-t', '--model', 
		dest='inputpath_model', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-a', '--alpha', 
		dest='alpha', 
		type=restricted_float_alpha, # control (0, 1) open interval
		action='store', 
		required=False, 
		default=0.05, 
		help='Significance level α (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − α) × 100]%% coverage. [OPTIONAL, DEFAULT: 0.05]'
		)
	parser_prediction.add_argument(
		'-o', '--output', 
		dest='outputpath', 
		action='store', 
		required=False, 
		default='.',
		help='Absolute or relative output path. [OPTIONAL, DEFAULT: .]'
		)
	parser_prediction.add_argument(
		'-x', '--prefix', 
		dest='prefix', 
		action='store', 
		required=False, 
		default='output',
		help='Prefix of output files. [OPTIONAL, DEFAULT: output_]'
		)
	parser_prediction.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=int,
		action='store', 
		required=False, 
		default=0, 
		help='Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]'
		)
	parser_prediction.add_argument(
		'-w', '--warnings', 
		dest='warnings', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]'
		)
	parser_prediction.add_argument(
		'-nc', '--no-check', 
		dest='nocheck', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]'
		)

	# print help if there are no arguments in the command
	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(1)

	# reshape arguments
	## parse the arguments
	args = parser.parse_args()
	## rename arguments
	if args.subcommand == 'modeling':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_PHENOTYPES=args.inputpath_phenotypes
		DATASET=args.dataset
		SPLITTING=args.splitting
		QUANTILES=args.quantiles
		LIMIT=args.limit
		FEATURESELECTION=args.featureselection
		REGRESSOR=args.regressor
		FOLD=args.fold
		PARAMETERS=args.parameters
		JOBS=args.jobs
		PERMUTATIONIMPORTANCE=args.permutationimportance
		NREPEATS=args.nrepeats
		ALPHA=args.alpha
		OUTPUTPATH=args.outputpath
		PREFIX=args.prefix
		DEBUG=args.debug
		WARNINGS=args.warnings
		NOCHECK=args.nocheck
	elif args.subcommand == 'prediction':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_FEATURES=args.inputpath_features
		INPUTPATH_FEATURE_ENCODER=args.inputpath_feature_encoder
		INPUTPATH_CALIBRATION_FEATURES=args.inputpath_calibration_features
		INPUTPATH_CALIBRATION_TARGETS=args.inputpath_calibration_targets
		INPUTPATH_MODEL=args.inputpath_model
		ALPHA=args.alpha
		OUTPUTPATH=args.outputpath
		PREFIX=args.prefix
		DEBUG=args.debug
		WARNINGS=args.warnings
		NOCHECK=args.nocheck

	# print a message about release
	message_release = "The GenomicBasedRegression script, version " + __version__ +  " (released in " + __release__ + ")," + " was launched"
	print(message_release)

	# set tracebacklimit
	sys.tracebacklimit = DEBUG
	message_traceback = "The traceback level was set to " + str(sys.tracebacklimit)
	print(message_traceback)

	# management of warnings
	if WARNINGS == True :
		wa.filterwarnings('default')
		message_warnings = "The warnings were not ignored"
		print(message_warnings)
	elif WARNINGS == False :
		wa.filterwarnings('ignore')
		message_warnings = "The warnings were ignored"
		print(message_warnings)

	# control versions
	if NOCHECK == False :
		## control Python version
		if sys.version_info[0] != 3 or sys.version_info[1] != 12 :
			raise Exception("Python 3.12 version is recommended")
		# control versions of packages
		if ap.__version__ != "1.1":
			raise Exception('argparse 1.1 (1.4.1) version is recommended')
		if im.version("boruta") != "0.4.3":
			raise Exception("boruta 0.4.3 version is recommended")
		if cb.__version__ != "1.2.8":
			raise Exception('catboost 1.2.8 version is recommended')
		if jl.__version__ != "1.5.1":
			raise Exception('joblib 1.5.1 version is recommended')
		if lgbm.__version__ != "4.6.0":
			raise Exception("lightgbm 4.6.0 version is recommended")
		if np.__version__ != "1.26.4":
			raise Exception("numpy 1.26.4 version is recommended")
		if pd.__version__ != "2.2.2":
			raise Exception('pandas 2.2.2 version is recommended')
		if pi.format_version != "4.0":
			raise Exception('pickle 4.0 version is recommended')
		if re.__version__ != "2.2.1":
			raise Exception('re 2.2.1 version is recommended')
		if sp.__version__ != "1.16.0":
			raise Exception("scipy 1.16.0 version is recommended")
		if sk.__version__ != "1.5.2":
			raise Exception('sklearn 1.5.2 version is recommended')
		if tq.__version__ != "4.67.1":
			raise Exception('tqdm 4.67.1 version is recommended')
		if im.version("tqdm-joblib") != "0.0.4":
			raise Exception("btqdm-joblib 0.0.4 version is recommended")
		if xgb.__version__ != "2.1.3":
			raise Exception("xgboost 2.1.3 version is recommended")
		message_versions = 'The recommended versions of Python and packages were properly controlled'
	else:
		message_versions = 'The recommended versions of Python and packages were not controlled'

	# print a message about version control
	print(message_versions)

	# set rounded digits
	digits = 6

	# check the subcommand and execute corresponding code
	if args.subcommand == 'modeling':

		# print a message about subcommand
		message_subcommand = "The modeling subcommand was used"
		print(message_subcommand)

		# manage minimal limits of samples
		if LIMIT < 10:
			message_limit = (
				"The provided sample limit per dataset (i.e., " + str(LIMIT) + ") was below the recommended minimum (i.e., 10) and may lead to unreliable performance metrics"
			)
			print(message_limit)
		else: 
			message_limit = (
				"The provided sample limit per dataset (i.e., " + str(LIMIT) + ") meets or exceeds the recommended minimum (i.e., 10), which is expected to support more reliable performance metrics"
			)
			print(message_limit)
		## define minimal limits of samples (i.e., 2 * LIMIT per dataset)
		limit_samples = 2 * LIMIT

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		## phenotypes
		df_phenotypes = pd.read_csv(INPUTPATH_PHENOTYPES, sep='\t', dtype=str)
		## convert the phenotype column (second column, index 1) to numeric after reading
		df_phenotypes.iloc[:, 1] = pd.to_numeric(df_phenotypes.iloc[:, 1], errors='raise')

		# check the input file of mutations
		## calculate the number of rows
		rows_mutations = len(df_mutations)
		## calculate the number of columns
		columns_mutations = len(df_mutations.columns)
		## check if more than limit_samples rows and 3 columns
		if (rows_mutations >= limit_samples) and (columns_mutations >= 3): 
			message_input_mutations = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., >= 3) in the input file of mutations were properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + ", respectively)"
			print(message_input_mutations)
		else: 
			message_input_mutations = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., >= 3) in the input file of mutations were not properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + ", respectively)"
			raise Exception(message_input_mutations)

		# check the input file of phenotypes
		## calculate the number of rows
		rows_phenotypes = len(df_phenotypes)
		## calculate the number of columns
		columns_phenotypes = len(df_phenotypes.columns)
		## check if more than limit_samples rows and 3 columns
		if (rows_phenotypes >= limit_samples) and (columns_phenotypes == 3): 
			message_input_phenotypes = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., = 3) in the input file of phenotypes were properly controlled (i.e., " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + ", respectively)"
			print(message_input_phenotypes)
		else: 
			message_input_phenotypes = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., = 3) in the input file of phenotypes were not properly controlled (i.e., " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + ", respectively)"
			raise Exception(message_input_phenotypes)
		## check the absence of missing data in the second column (i.e., phenotype)
		missing_phenotypes = pd.Series(df_phenotypes.iloc[:,1]).isnull().values.any()
		if missing_phenotypes == False: 
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was properly controlled (i.e., the second column)"
			print(message_missing_phenotypes)
		elif missing_phenotypes == True:
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was inproperly controlled (i.e., the second column)"
			raise Exception(message_missing_phenotypes)
		## check the absence of values other than 'training' or 'testing' in the third column (i.e., dataset)
		if (DATASET == "manual"):
			expected_datasets = all(df_phenotypes.iloc[:,2].isin(["training", "testing"]))
			if expected_datasets == True: 
				message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were properly controlled (i.e., the third column)"
				print (message_expected_datasets)
			elif expected_datasets == False:
				message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were inproperly controlled (i.e., the third column)"
				raise Exception(message_expected_datasets)
		elif (DATASET == "random"):
			message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were not controlled (i.e., the third column)"
			print(message_expected_datasets)

		# replace missing genomic data by a string
		df_mutations = df_mutations.fillna('missing')

		# rename variables of headers
		## mutations
		df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
		## phenotypes
		df_phenotypes.rename(columns={df_phenotypes.columns[0]: 'sample'}, inplace=True)
		df_phenotypes.rename(columns={df_phenotypes.columns[1]: 'phenotype'}, inplace=True)
		df_phenotypes.rename(columns={df_phenotypes.columns[2]: 'dataset'}, inplace=True)

		# sort by samples
		## mutations
		df_mutations = df_mutations.sort_values(by='sample')
		## phenotypes
		df_phenotypes = df_phenotypes.sort_values(by='sample')

		# check if lists of sorted samples are identical
		## convert DataFrame column as a list
		lst_mutations = df_mutations['sample'].tolist()
		lst_phenotypes = df_phenotypes['sample'].tolist()
		## compare lists
		if lst_mutations == lst_phenotypes: 
			message_sample_identifiers = "The sorted sample identifiers were confirmed as identical between the input files of mutations and phenotypes/datasets"
			print (message_sample_identifiers)
		else: 
			message_sample_identifiers = "The sorted sample identifiers were confirmed as not identical between the input files of mutations and phenotypes/datasets"
			raise Exception(message_sample_identifiers)

		# check compatibility between the dataset and splitting arguments
		if (DATASET == 'random') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and percentage of random splitting (i.e., " + str(SPLITTING) + "%) were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'random') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") required the percentage of random splitting (i.e., " + str(SPLITTING) + ")"
			raise Exception(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and percentage of random splitting (i.e., " + str(SPLITTING) + ") were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") did not require the percentage of random splitting (i.e., " + str(SPLITTING) + "%)"
			raise Exception(message_compatibility_dataset_slitting)
		
		# check compatibility between the dataset and quantiles arguments
		if (DATASET == 'random') and (QUANTILES != None):
			message_compatibility_dataset_quantiles = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and number of quantiles (i.e., " + str(QUANTILES) + ") were compatible"
			print(message_compatibility_dataset_quantiles)
		elif (DATASET == 'random') and (QUANTILES == None):
			message_compatibility_dataset_quantiles = "The provided selection of training/testing datasets (i.e., " + DATASET + ") required the number of quantiles (i.e., " + str(QUANTILES) + ")"
			raise Exception(message_compatibility_dataset_quantiles)
		elif (DATASET == 'manual') and (QUANTILES == None):
			message_compatibility_dataset_quantiles = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and number of quantiles (i.e., " + str(QUANTILES) + ") were compatible"
			print(message_compatibility_dataset_quantiles)
		elif (DATASET == 'manual') and (QUANTILES != None):
			message_compatibility_dataset_quantiles = "The provided selection of training/testing datasets (i.e., " + DATASET + ") did not require the number of quantiles (i.e., " + str(QUANTILES) + ")"
			raise Exception(message_compatibility_dataset_quantiles)

		# perform splitting of the training and testing datasets according to the settings
		if DATASET == 'random':
			message_dataset = "The training and testing datasets were constructed based on the 'random' setting"
			print(message_dataset)
			# trash dataset column
			df_phenotypes = df_phenotypes.drop("dataset", axis='columns')
			# index with sample identifiers the dataframes mutations (X) and phenotypes (y)
			X = df_mutations.set_index('sample')
			y = df_phenotypes.set_index('sample')
			# get the first column of y (i.e., phenotype) as a Pandas Series
			y_series = y.iloc[:, 0]
			# bin y into quantiles
			y_binned = pd.qcut(y_series, # quantile-based discretization
							q=QUANTILES, # number of quantiles
							labels=False, # return bin integer
							duplicates='drop') # reduce number of bin to avoid ValueError: Bin edges must be unique
			# perform a stratified split based on binned targets (Note: ensures balanced distributions even if the exact split ratio is not followed perfectly)
			split = StratifiedShuffleSplit(
				n_splits=1, 
				test_size=1 - SPLITTING/100,
            	random_state=None # make it random (random_state=42 for reproducibility)
			)
			for train_idx, test_idx in split.split(X, y_binned):
				X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
				y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
			# ensure y_train and y_test are 1D Series instead of (n,1) DataFrames
			y_train = y_train.iloc[:, 0] # extract phenotype column as Series to avoid DataConversionWarning (for random split)
			y_test = y_test.iloc[:, 0] # extract phenotype column as Series to avoid DataConversionWarning (for random split)
		elif DATASET == "manual":
			message_dataset = ("The training and testing datasets were constructed based on the 'manual' setting")
			print(message_dataset)
			# align phenotype and mutation dataframes by the common key ‘sample’ (validate='one_to_one' will raise if a sample exists twice in either file)
			# use merge (row is matched by sample name and mismatches raise errors) rather than concat(axis=1) (assume that dataframes have the same row order and rows correspond to the same samples index by index)
			df_phenotypes_mutations = (
				df_phenotypes
				.merge(df_mutations, on="sample", how="inner", validate="one_to_one")
			)
			# split rows back into training/testing, using the saved ‘dataset’ column
			df_training = df_phenotypes_mutations[
				df_phenotypes_mutations["dataset"] == "training"
			]
			df_testing = df_phenotypes_mutations[
				df_phenotypes_mutations["dataset"] == "testing"
			]
			# build y‑dataframes (phenotypes) indexing sample identifiers
			y_train = df_training[["sample", "phenotype"]].set_index("sample")
			y_test  = df_testing[["sample", "phenotype"]].set_index("sample")
			# ensure y_train and y_test are 1D Series instead of (n,1) DataFrames
			y_train = y_train["phenotype"] # extract phenotype column as Series to avoid DataConversionWarning (for manual split)
			y_test = y_test["phenotype"] # extract phenotype column as Series to avoid DataConversionWarning (for manual split)
			# build X‑dataframes (mutations), keeping one ‘sample’ column as index
			X_train = (
				df_training
				.drop(columns=["phenotype", "dataset"]) # drop only these two
				.set_index("sample")
			)
			X_test = (
				df_testing
				.drop(columns=["phenotype", "dataset"])
				.set_index("sample")
			)

		# check similarity between distributions of phenotypes from the training and testing datasets 
		## compute Kolmogorov–Smirnov statistic and p-values
		ks_stat, ks_p_value = sp.stats.ks_2samp(y_train, y_test)
		## convert as float
		ks_stat = ks_stat.item()
		ks_p_value = ks_p_value.item()
		## print a message
		message_differences_distributions = "The Kolmogorov–Smirnov statistic was computed to compare the distributions of phenotypes in the training and testing datasets: " + str(round(ks_stat, digits)) + " (p-value: " + str(round(ks_p_value, digits)) + ")"
		print(message_differences_distributions)

		# check number of samples per dataset (i.e., 'training' and 'testing')
		## count number of samples in the training dataset
		count_train_int = len(X_train)
		## count number of samples in the testing dataset
		count_test_int = len(X_test)
		## control minimal number of samples per dataset
		### detect small number of samples in the training dataset
		if (count_train_int >= LIMIT):
			detection_train = True
		else:
			detection_train = False
		### detect small number of samples in the testing dataset
		if (count_test_int >= LIMIT):
			detection_test = True
		else:
			detection_test = False
		### check the minimal quantity of samples per dataset
		if (detection_train == True) and (detection_test == True):
			message_count_samples = "The number of samples in the training (i.e., " + str(count_train_int) + ") and testing (i.e., " + str(count_test_int) + ") datasets was properly controlled to be higher than, or equal to, the set limit (i.e., " + str(LIMIT) + ")"
			print(message_count_samples)
		elif (detection_train == False) or (detection_test == False):
			message_count_samples = "The number of samples in the training (i.e., " + str(count_train_int) + ") and testing (i.e., " + str(count_test_int) + ") datasets was improperly controlled, making it lower than the set limit (i.e., " + str(LIMIT) + ")"
			raise Exception(message_count_samples)

		# encode categorical data into binary data using the one-hot encoder
		## save input feature names from training dataset
		features = X_train.columns
		## instantiate the encoder
		encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
		## fit encoder on training data (implicitly includes all training features)
		X_train_encoded = encoder.fit_transform(X_train.astype(str))
		## check missing features
		missing_features = set(features) - set(X_test.columns)
		if missing_features:
			message_missing_features = "The following training features expected by the one-hot encoder are missing in the input tested mutations: " + str(missing_features)
			raise Exception(message_missing_features)
		else: 
			message_missing_features = "The input tested mutations include all features required by the trained one-hot encoder"
			print (message_missing_features)
		## check extra features
		extra_features = set(X_test.columns) - set(features)
		if extra_features:
			message_extra_features = "The following unexpected features in the input tested mutations will be ignored for one-hot encoding: " + str(extra_features)
			print (message_extra_features)
		else: 
			message_extra_features = "The input tested mutations contain no unexpected features for one-hot encoding"
			print (message_extra_features)
		## ensure order and cast to str for encoding
		X_test_features_str = X_test[features].astype(str)
		## use the same encoder (already fitted) to transform test data
		X_test_encoded = encoder.transform(X_test_features_str)

		# count features
		## count the number of raw categorical features before one-hot encoding
		features_before_ohe_int = len(features)
		## count the number of binary features after one-hot encoding
		features_after_ohe_int = X_train_encoded.shape[1]
		## print a message
		message_ohe_features = "The " + str(features_before_ohe_int) + " provided features were one-hot encoded into " + str(features_after_ohe_int) + " encoded features"
		print(message_ohe_features)
		
		# prepare elements of the model
		## initialize the feature selection method (without tuning parameters: deterministic and repeatable)
		if FEATURESELECTION == 'None':
			message_feature_selection = "The provided feature selection method was properly recognized: None"
			print(message_feature_selection)
			selected_feature_selector = None
		elif FEATURESELECTION == 'SKB':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectKBest (SKB)"
			print(message_feature_selection)
			selected_feature_selector = SelectKBest(
				score_func=ft.partial( # partial allow reproducibility
					mutual_info_regression, # mutual_info_regression captures linear and non-linear dependancies, default f_regression was not proposed per default because it captures only linear dependancies, optional chi2 is not recommended for regression, all score_func can be used through tunining parameters
					random_state=42 # reproducibility
				), 
				k=10 # default top k features can be modified in the parameters file if needed
			) 
		elif FEATURESELECTION in ['laSFM', 'rfSFM', 'enSFM']:
			if FEATURESELECTION == 'laSFM':
				message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with lasso (laSFM)"
				selector_model = Lasso(random_state=42) # reproducibility
			elif FEATURESELECTION == 'rfSFM':
				message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with random forest (rfSFM)"
				selector_model = RandomForestRegressor(random_state=42) # reproducibility
			elif FEATURESELECTION == 'enSFM':
				message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with elasticnet (enSFM)"
				selector_model = ElasticNet(random_state=42) # reproducibility
			print(message_feature_selection)
			selected_feature_selector = SelectFromModel(
				estimator=selector_model,
				threshold=None # default threshold behavior based on model, user can specify max_features in the parameters file if needed together with 'threshold': [-float('inf')] to rank features by importance
			)
		elif FEATURESELECTION == 'BO':
			message_feature_selection = "The provided feature selection method was properly recognized: Boruta (BO)"
			print(message_feature_selection)
			# Boruta requires a regressor internally, for instance RandomForestRegressor with fast default or param overrides
			boruta_estimator = RandomForestRegressor(random_state=42) # reproducibility
			# create Boruta selector with fast default custom parameters, can be expanded later from parameters file
			boruta_selector = BorutaSelectorDF(
				estimator=boruta_estimator,
				n_estimators='auto', # use the same number of estimators as defined in the estimator (custom for speed, Boruta default: 100)
				max_iter=10, # few iterations to reduce runtime (custom for speed, Boruta default: 100)
				perc=85, # stricter cutoff can reduce number of features selected (custom for speed, Boruta default: 100)
				two_step=True, # can reduce the number of iterations (custom for speed, Boruta default: False)
				verbose=0, # avoid print in shell (custom, Boruta default: 1)
				random_state=42 # reproducibility
			)
			selected_feature_selector = boruta_selector
		else:
			message_feature_selection = "The provided feature selection method is not implemented yet"
			raise Exception(message_feature_selection)
		
		## initialize the regressor (without tuning parameters: deterministic and repeatable if possible)
		if REGRESSOR == 'BRI':
			message_regressor = "The provided regressor was properly recognized: bayesian ridge (BRI)"
			print(message_regressor)
			selected_regressor = BayesianRidge()
		elif REGRESSOR == 'CB':
			message_regressor = "The provided regressor was properly recognized: catboost (CB)"
			print(message_regressor)
			selected_regressor = cb.CatBoostRegressor(random_state=42, verbose=0)
		elif REGRESSOR == 'DT':
			message_regressor = "The provided regressor was properly recognized: decision tree (DT)"
			print(message_regressor)
			selected_regressor = DecisionTreeRegressor(random_state=42)
		elif REGRESSOR == 'EN':
			message_regressor = "The provided regressor was properly recognized: elasticNet (EN)"
			print(message_regressor)
			selected_regressor = ElasticNet(random_state=42, selection='random')
		elif REGRESSOR == 'GB':
			message_regressor = "The provided regressor was properly recognized: gradient boosting (GB)"
			print(message_regressor)
			selected_regressor = GradientBoostingRegressor(random_state=42)
		elif REGRESSOR == 'HGB':
			message_regressor = "The provided regressor was properly recognized: hist gradient boosting (HGB)"
			print(message_regressor)
			selected_regressor = HistGradientBoostingRegressor(random_state=42)
		elif REGRESSOR == 'HU':
			message_regressor = "The provided regressor was properly recognized: huber (HU)"
			print(message_regressor)
			selected_regressor = HuberRegressor()
		elif REGRESSOR == 'KNN':
			message_regressor = "The provided regressor was properly recognized: k-nearest neighbors (KNN)"
			print(message_regressor)
			selected_regressor = KNeighborsRegressor()
		elif REGRESSOR == 'LA':
			message_regressor = "The provided regressor was properly recognized: lasso (LA)"
			print(message_regressor)
			selected_regressor = Lasso(random_state=42)
		elif REGRESSOR == 'LGBM':
			message_regressor = "The provided regressor was properly recognized: Light Gradient Boosting Machine (LGBM)"
			print(message_regressor)
			selected_regressor = lgbm.LGBMRegressor(random_state=42, verbose=-1)
		elif REGRESSOR == 'MLP':
			message_regressor = "The provided regressor was properly recognized: multi-layer perceptron (MLP)"
			print(message_regressor)
			selected_regressor = MLPRegressor(random_state=42)
		elif REGRESSOR == 'NSV':
			message_regressor = "The provided regressor was properly recognized: nu support vector (NSV)"
			print(message_regressor)
			selected_regressor = NuSVR()
		elif REGRESSOR == 'PN':
			message_regressor = "The provided regressor was properly recognized: polynomial (PN)"
			print(message_regressor)
			selected_regressor = LinearRegression()
		elif REGRESSOR == 'RI':
			message_regressor = "The provided regressor was properly recognized: ridge (RI)"
			print(message_regressor)
			selected_regressor = Ridge()
		elif REGRESSOR == 'RF':
			message_regressor = "The provided regressor was properly recognized: random forest (RF)"
			print(message_regressor)
			selected_regressor = RandomForestRegressor(random_state=42)
		elif REGRESSOR == 'SVR':
			message_regressor = "The provided regressor was properly recognized: support vector regression (SVR)"
			print(message_regressor)
			selected_regressor = SVR()
		elif REGRESSOR == 'XGB':
			message_regressor = "The provided regressor was properly recognized: extreme gradient boosting (XGB)"
			print(message_regressor)
			selected_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
		else: 
			message_regressor = "The provided regressor is not implemented yet"
			raise Exception(message_regressor)

		## build the pipeline
		### create an empty list
		steps = []
		### add feature selection step if specified
		if FEATURESELECTION in ['SKB', 'laSFM', 'rfSFM', 'enSFM', 'BO']:
			steps.append(('feature_selection', selected_feature_selector))
		### add polynomial features if the regressor is polynomial
		if REGRESSOR == 'PN': # use aggressive feature selection (laSFM or rfSFM) before polynomial expansion to get stable predictions
			steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))  # # default polynomial regression of degree=2 (can be changed in the tuning parameter)
			steps.append(('scaler', StandardScaler()))  # avoid scale imbalance to minimize inflated predictions and prediction intervals
		### add the final model
		steps.append(('model', selected_regressor))
		### create the pipeline
		selected_pipeline = Pipeline(steps)
		### print a message
		message_pipeline = "The pipeline components were properly recognized: " + re.sub(r'\s+', ' ', str(selected_pipeline)).strip()
		print(message_pipeline)

		## initialize the parameters
		### if the tuning parameters are not provided by the user
		if PARAMETERS == None:
			parameters = [{}]
			message_parameters = "The tuning parameters were not provided by the user"
			print(message_parameters)
		### if the tuning parameters are provided by the user
		elif PARAMETERS != None:
			### read provided tuning parameters and convert string dictionary to Python dictionary keeping the convenience of writing real Python objects into the parameter file (e.g., chi2, -float('inf'), ....)
			with open(PARAMETERS, "r") as parameters_file:
				parameters = [eval(parameters_file.read())]
			### print a message
			message_parameters = "The provided tuning parameters were properly recognized: " + str(parameters)
			print(message_parameters)

		# build the model
		## prepare the grid search cross-validation (CV) first
		model = GridSearchCV(
			estimator=selected_pipeline,
			param_grid=parameters,
			cv=FOLD,
			scoring='neg_root_mean_squared_error', # average prediction error in the same unit as your target # the one with the lowest RMSE (i.e., highest neg-RMSE) was chosen)
			n_jobs=JOBS,
			verbose=0 # do not display any messages or logs
		)

		## compute metrics related to grid search CV
		### number of distinct parameter names (i.e., how many parameters are tuned)
		n_param_names = len({k for d in parameters for k in d})
		### number of parameter value options (i.e., how many values are tried in total)
		n_total_values = sum(len(v) for d in parameters for v in d.values())
		### number of parameter combinations (i.e., Cartesian product of all value options)
		param_combinations = len(list(ParameterGrid(parameters)))
		### number of fits during cross-validation (i.e., combinations × folds)
		gridsearchcv_fits = param_combinations * FOLD
		### print a message
		message_metrics_cv = "The cross-validation setting implied: " + str(n_param_names) + " distinct parameter names, " + str(n_total_values) + " parameter value options, " + str(param_combinations) + " parameter combinations, and " + str(gridsearchcv_fits) + " fits during cross-validation"
		print(message_metrics_cv)


		# ensure numeric compatibility (astype(np.float32)) with upstream encoding (sparse_output=False) and efficiency (float32 dtype), 
		# especially for tree-based regressors (e.g., HistGradientBoostingRegressor, XGBRegressor, LightGBM)
		X_train_encoded_float32 = X_train_encoded.astype(np.float32)
		# convert to float32 NumPy array (avoid object dtype, required by LightGBM and other regressors)
		y_train_float32 = y_train.astype(np.float32).to_numpy()

		## fit the model
		### use tqdm.auto rather than tqdm library because it automatically choose the best display format (terminal, notebook, etc.)
		### use a tqdm progress bar from the tqdm_joblib library (compatible with GridSearchCV)
		### use a tqdm progress bar immediately after the last print (position=0), disable the additional bar after completion (leave=False), and allow for dynamic resizing (dynamic_ncols=True)
		### force GridSearchCV to use the threading backend to avoid the DeprecationWarning from fork and ChildProcessError from the loky backend (default in joblib)
		### threading is slower than loky, but it allows using a progress bar with GridSearchCV and avoids the DeprecationWarning and ChildProcessError
		with tqa.tqdm(total=gridsearchcv_fits, desc="Model building progress", position=0, leave=False, dynamic_ncols=True) as progress_bar:
			with jl.parallel_backend('threading', n_jobs=JOBS):
				with tqjl.tqdm_joblib(progress_bar):
					model.fit(
						X_train_encoded_float32,
						y_train_float32
					)

		## print best parameters
		if PARAMETERS == None:
			message_best_parameters = "The best parameters during model cross-validation were not computed because they were not provided"
		elif PARAMETERS != None:
			message_best_parameters = "The best parameters during model cross-validation were: " + str(model.best_params_)
		print(message_best_parameters)
		## print best score
		message_best_score = "The best negative root mean squared error during model cross-validation was: " + str(round(model.best_score_, digits))
		print(message_best_score)

		# retrieve the combinations of tested parameters and corresponding scores
		## combinations of tested parameters
		allparameters_lst = model.cv_results_['params']
		## corresponding scores
		allscores_nda = model.cv_results_['mean_test_score']
		## transform the list of parameters into a dataframe
		allparameters_df = pd.DataFrame({'parameters': allparameters_lst})
		## transform the ndarray of scores into a dataframe
		allscores_df = pd.DataFrame({'scores': allscores_nda})
		## concatenate horizontally dataframes
		all_scores_parameters_df = pd.concat(
			[allscores_df, 
			allparameters_df], 
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)
		## remove unnecessary characters
		### replace each dictionary by string
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].apply(lambda x: str(x))
		### replace special characters { and } by nothing
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].replace(r'[\{\}]', '', regex=True)
		# sort the dataframe by scores in descending order and reset the index
		all_scores_parameters_df = all_scores_parameters_df.sort_values(by="scores", ascending=False).reset_index(drop=True)

		# select the best model
		best_model = model.best_estimator_
		
		# count features
		## count the number of features selected by feature selection actually used by the final regressor
		selected_features_int = count_selected_features(best_model, X_train_encoded)
		## print a message
		message_selected_features = (
			"The pipeline potentially selected and used "
			+ str(selected_features_int)
			+ " one-hot encoded features to train the model"
		)
		print(message_selected_features)

		# output a dataframe of features used by the final model with ranked importance scores
		# get the final estimator from the pipeline or directly if standalone
		final_estimator = best_model[-1] if hasattr(best_model, '__getitem__') else best_model
		# initialize list of feature names and selection mask
		feature_encoded_lst = None
		support_mask = None
		selector = None
		try:
			# check if the model is a pipeline
			if hasattr(best_model, 'named_steps'):
				# if it contains a polynomial transformer, extract feature names
				if 'poly' in best_model.named_steps and hasattr(best_model.named_steps['poly'], 'get_feature_names_out'):
					input_features = X_train_encoded.columns
					feature_encoded_lst = best_model.named_steps['poly'].get_feature_names_out(input_features=input_features)
				else:
					# otherwise use column names from the encoded training data
					feature_encoded_lst = X_train_encoded.columns
				# check for a feature selection step
				if 'feature_selection' in best_model.named_steps:
					selector = best_model.named_steps['feature_selection']
					# if get_support is defined, use it to filter selected features
					if hasattr(selector, 'get_support'):
						support_mask = selector.get_support()
						# align feature names with selected features
						feature_encoded_lst = np.array(feature_encoded_lst)[support_mask]
					# otherwise check for ranking_ attribute, used by boruta
					elif hasattr(selector, 'ranking_'):
						ranking = np.array(selector.ranking_)
						support_mask = ranking == 1
						# align feature names with selected features
						feature_encoded_lst = np.array(feature_encoded_lst)[support_mask]
					else:
						# fallback: assume all features were kept
						support_mask = np.ones(len(feature_encoded_lst), dtype=bool)
			# if the best_model (final estimator) has get_feature_names_out method (e.g., PolynomialFeatures)
			elif hasattr(best_model, 'get_feature_names_out'):
				# use model's method to get transformed feature names
				feature_encoded_lst = best_model.get_feature_names_out()
				# assume no feature selection applied (all True mask)
				support_mask = np.ones(len(feature_encoded_lst), dtype=bool)
			else:
				# fallback: use column names from encoded training data
				feature_encoded_lst = X_train_encoded.columns
				# assume no feature selection applied
				support_mask = np.ones(len(feature_encoded_lst), dtype=bool)
			message_importance_encoded_feature_names = "The full one-hot encoded feature names were recovered from the pipeline"
		except Exception:
			# fallback: on error, use encoded training data column names
			feature_encoded_lst = X_train_encoded.columns
			# assume no feature selection applied
			support_mask = np.ones(len(feature_encoded_lst), dtype=bool)
			message_importance_encoded_feature_names = "The full one-hot encoded feature names were not recovered from the pipeline"
		print(message_importance_encoded_feature_names)
		# ensure feature names are a list
		if hasattr(feature_encoded_lst, 'tolist'):
			feature_encoded_lst = feature_encoded_lst.tolist()
		# get importance_type parameter from model params if exists, else fallback to defaults
		importance_type = None
		try:
			if isinstance(final_estimator, cb.CatBoostRegressor):
				# CatBoost regressor: get feature importance using loss-based method
				importances = final_estimator.get_feature_importance(prettified=False)
				importance_type = "catboost's loss-based importance"
			elif isinstance(final_estimator, xgb.XGBRegressor):
				# XGBoost regressor: determine importance_type from parameters or fallback to 'weight'
				xgb_importance_type = None
				# check if params dict exists and contains importance_type (model__importance_type)
				if 'params' in locals():
					xgb_importance_type = params.get('model__importance_type', None)
				# else check model's own get_params for importance_type
				if xgb_importance_type is None and hasattr(final_estimator, 'get_params'):
					params_ = final_estimator.get_params()
					xgb_importance_type = params_.get('importance_type', None)
				# fallback to XGBoost default importance type 'weight' if none found
				if xgb_importance_type is None:
					xgb_importance_type = 'weight'
				# get booster and feature names
				booster = final_estimator.get_booster()
				xgb_feature_names = booster.feature_names
				# get importance scores dict for given importance_type
				importances_dict = booster.get_score(importance_type=xgb_importance_type)
				# create importance array aligned with booster feature names (fill missing with 0)
				importances = np.array([importances_dict.get(f, 0.0) for f in xgb_feature_names])
				# update feature names to booster feature names for correct alignment
				feature_encoded_lst = list(xgb_feature_names)
				importance_type = f"xgboost's {xgb_importance_type}-based importance"
			elif isinstance(final_estimator, lgbm.LGBMRegressor):
				# LightGBM regressor: determine importance_type from parameters or fallback to 'gain'
				lgbm_importance_type = None
				# check if params dict exists and contains importance_type (model__importance_type)
				if 'params' in locals():
					lgbm_importance_type = params.get('model__importance_type', None)
				# else check model's own get_params for importance_type
				if lgbm_importance_type is None and hasattr(final_estimator, 'get_params'):
					params_ = final_estimator.get_params()
					lgbm_importance_type = params_.get('importance_type', None)
				# fallback to LightGBM default importance type 'gain' if none found
				if lgbm_importance_type is None:
					lgbm_importance_type = 'gain'
				# get importance scores with chosen importance_type
				importances = final_estimator.booster_.feature_importance(importance_type=lgbm_importance_type)
				importance_type = f"lightgbm's {lgbm_importance_type}-based importance"
			elif hasattr(final_estimator, 'feature_importances_'):
				# fallback for tree-based models exposing feature_importances_
				importances = final_estimator.feature_importances_
				importance_type = "tree-based impurity reduction (feature_importances_)"
			elif hasattr(final_estimator, 'coef_'):
				# fallback for linear models with coef_ attribute
				importances = np.abs(final_estimator.coef_.ravel() if hasattr(final_estimator.coef_, 'ravel') else final_estimator.coef_)
				importance_type = "absolute coefficient magnitude (coef_)"
			else:
				# fallback: no importance available, fill with NaNs
				importances = np.array([np.nan] * len(feature_encoded_lst))
				importance_type = "NaN placeholder"
		except Exception as e:
			# fallback: error while extracting importance, fill with NaNs
			importances = np.array([np.nan] * len(feature_encoded_lst))
			importance_type = "NaN fallback due to importance extraction error: " + str(e)
		# check for size mismatch between importances and feature names, truncate to shortest length
		if len(importances) != len(feature_encoded_lst):
			min_len = min(len(importances), len(feature_encoded_lst))
			importances = importances[:min_len]
			feature_encoded_lst = feature_encoded_lst[:min_len]
		# print a message depending on regressor type
		if REGRESSOR in ('HGB', 'KNN', 'MLP', 'NSV', 'SVR'):
			message_importance_count = (
				"The selected model regressor did not expose feature importance natively ("
				+ importance_type + ")"
			)
		else:
			message_importance_count = (
				"The best model returned "
				+ str(len(importances))
				+ " importance values (" + importance_type + ") for "
				+ str(len(feature_encoded_lst))
				+ " one-hot encoded features (potentially selected and/or polynomially expanded)"
			)
		print(message_importance_count)
		# create dataframe of feature importances and round importance values to 6 decimal places
		feature_importance_df = pd.DataFrame({
			"feature": feature_encoded_lst,
			"importance": np.round(importances, digits)  # round before assigning
		})
		# sort by descending importance
		feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

		# compute permutation importance only if explicitly requested
		# use tqdm.auto rather than tqdm library because it automatically chooses the best display format (terminal, notebook, etc.)
		# use a tqdm progress bar from the tqdm_joblib library (compatible with permutation_importance using joblib parallelism)
		# use a tqdm progress bar immediately after the last print (position=0), disable the additional bar after completion (leave=False), and allow for dynamic resizing (dynamic_ncols=True)
		# force permutation_importance to use the threading backend to avoid the DeprecationWarning from fork and ChildProcessError from the loky backend (default in joblib)
		# threading is slower than loky, but it allows using a progress bar with joblib and avoids the DeprecationWarning and ChildProcessError
		if PERMUTATIONIMPORTANCE is True:
			try:
				# compute the total number of permutations to estimate progress: each feature will be shuffled 'n_repeats' times
				n_repeats = 10  # default number of permutations per feature
				n_features = X_train_encoded_float32.shape[1]
				permutation_total = n_features  # not n_features * n_repeats to make the bar correctly report based on actual joblib jobs
				with tqa.tqdm(total=permutation_total, desc="Permutation importance progress", position=0, leave=False, dynamic_ncols=True) as progress_bar:
					with jl.parallel_backend('threading', n_jobs=JOBS):
						with tqjl.tqdm_joblib(progress_bar):
							permutation_result = permutation_importance(
								best_model,                     # can be a full pipeline or a standalone estimator
								X_train_encoded_float32,       # encoded features used for training
								y_train_float32,               # target values
								n_repeats=NREPEATS,           # number of repetitions per feature
								random_state=42,               # ensure reproducibility
								scoring='neg_root_mean_squared_error',  # scoring consistent with RMSE used in CV
								n_jobs=JOBS                    # number of parallel jobs
							)
				# extract average permutation importance and its standard deviation
				perm_importances = np.round(permutation_result.importances_mean, digits)
				perm_std = np.round(permutation_result.importances_std, digits)
				# handle any shape mismatch between importance scores and recovered feature names
				if len(perm_importances) != len(feature_encoded_lst):
					min_len = min(len(perm_importances), len(feature_encoded_lst))
					perm_importances = perm_importances[:min_len]
					perm_std = perm_std[:min_len]
					feature_encoded_lst = feature_encoded_lst[:min_len]
				# construct the final DataFrame containing mean and std of permutation importance
				permutation_importance_df = pd.DataFrame({
					'feature': feature_encoded_lst,
					'permutation_importance_mean': perm_importances,
					'permutation_importance_std': perm_std
				}).sort_values(by="permutation_importance_mean", ascending=False).reset_index(drop=True)
				# message to confirm success
				message_permutation = (
					"The permutation importance was successfully computed "
					"using sklearn.inspection.permutation_importance"
				)
			except Exception as e:
				# fallback in case of failure: return empty DataFrame and report error
				permutation_importance_df = pd.DataFrame()
				message_permutation = (
					"An error occurred while computing permutation importance: " + str(e)
				)
		else:
			# if not requested, return empty DataFrame and skip computation
			permutation_importance_df = pd.DataFrame()
			message_permutation = "The permutation importance was not computed"
		# print a message
		print(message_permutation)

		# perform prediction
		## from the training dataset
		y_pred_train = best_model.predict(X_train_encoded)
		## from the testing dataset
		y_pred_test = best_model.predict(X_test_encoded)

		# evaluate model
		# root mean squared error (RMSE): square root of the mean of the squared differences between true and predicted values
		# → sensitive to large errors; useful when large deviations are particularly undesirable
		# mean squared error (MSE): average of the squared differences between true and predicted values
		# → penalizes larger errors more heavily; often used for optimization during model training
		# symmetric mean absolute percentage error (SMAPE): mean of the absolute differences between true and predicted values, divided by their average magnitude
		# → scale-independent and symmetric; useful when comparing performance across datasets with different scales
		# mean absolute percentage error (MAPE): mean of the absolute percentage errors between true and predicted values
		# → interpretable as average percentage error; can be misleading with near-zero targets
		# mean absolute error (MAE): average of the absolute differences between true and predicted values
		# → more robust to outliers than MSE; intuitive and scale-dependent
		# R-squared (R2): proportion of the variance in the dependent variable that is predictable from the independent variables
		# → indicates goodness of fit; higher is better, but does not penalize for model complexity
		# adjusted R-squared (aR2): R-squared adjusted for the number of predictors; it accounts for model complexity
		# → preferred when comparing models with different numbers of predictors

		## compute metrics
		### RMSE
		rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
		rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
		### MSE
		mse_train = mean_squared_error(y_train, y_pred_train)
		mse_test = mean_squared_error(y_test, y_pred_test)
		### SMAPE
		smape_train = smape(y_train, y_pred_train, threshold=1e-3)
		smape_test = smape(y_test, y_pred_test, threshold=1e-3)
		### MAPE
		mape_train = mape(y_train, y_pred_train, threshold=1e-3)
		mape_test = mape(y_test, y_pred_test, threshold=1e-3)
		### MAE
		mae_train = mean_absolute_error(y_train, y_pred_train)
		mae_test = mean_absolute_error(y_test, y_pred_test)	
		### R2
		r2_train = r2_score(y_train, y_pred_train)
		r2_test = r2_score(y_test, y_pred_test)
		### aR2
		ar2_train = adjusted_r2(y_train, y_pred_train, n_features=X_train.shape[1])
		ar2_test = adjusted_r2(y_test, y_pred_test, n_features=X_test.shape[1])

		## combine in dataframes
		## from the training dataset
		metrics_global_train_df = pd.DataFrame({
			'RMSE': [round(rmse_train, digits)], 
			'MSE': [round(mse_train, digits)], 
			'SMAPE': [round(smape_train, digits)], 
			'MAPE': [round(mape_train, digits)], 
			'MAE': [round(mae_train, digits)], 			
			'R2': [round(r2_train, digits)], 
			'aR2': [round(ar2_train, digits)],
			})
		## from the testing dataset
		metrics_global_test_df = pd.DataFrame({
			'RMSE': [round(rmse_test, digits)], 
			'MSE': [round(mse_test, digits)], 
			'SMAPE': [round(smape_test, digits)],
			'MAPE': [round(mape_test, digits)],
			'MAE': [round(mae_test, digits)], 
			'R2': [round(r2_test, digits)], 
			'aR2': [round(ar2_test, digits)], 
			})

		# combine expectation and prediction from the training
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pred_train_df = pd.DataFrame(y_pred_train)
		## retrieve the sample index in a column
		y_train_df = y_train.reset_index().rename(columns={"index":"sample"})
		## concatenate horizontally dataframes
		combined_train_df = pd.concat(
			[y_train_df.reset_index(drop=True), 
			y_pred_train_df.reset_index(drop=True)], 
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		) 
		## rename labels of headers
		combined_train_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
		combined_train_df.rename(columns={0: 'prediction'}, inplace=True)

		# combine expectation and prediction from the testing
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pred_test_df = pd.DataFrame(y_pred_test)
		## retrieve the sample index in a column
		y_test_df = y_test.reset_index().rename(columns={"index":"sample"})
		## concatenate horizontally dataframes
		combined_test_df = pd.concat(
			[y_test_df.reset_index(drop=True), 
			y_pred_test_df.reset_index(drop=True)], 
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		) 
		## rename labels of headers
		combined_test_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
		combined_test_df.rename(columns={0: 'prediction'}, inplace=True)

		# retrieve only prediction intervals using a custom ResidualQuantileWrapper independantly of mapie 0.9.2 to be able to manage only one sample
		## instantiate the residual quantile wrapper with the best trained model and the desired alpha level
		res_wrapper = ResidualQuantileWrapper(estimator=best_model, alpha=ALPHA)
		## fit the wrapper: trains the underlying model and calculates residual quantile for prediction intervals
		res_wrapper.fit(X_train_encoded, y_train_float32)
		## predict on training data, returning both point predictions and prediction intervals
		y_pred_train_res_wrapper, y_intervals_train = res_wrapper.predict(X_train_encoded, return_prediction_interval=True)
		## predict on testing data, returning both point predictions and prediction intervals
		y_pred_test_res_wrapper, y_intervals_test = res_wrapper.predict(X_test_encoded, return_prediction_interval=True)
		## convert the numpy array of prediction intervals on training data into a pandas DataFrame
		## columns are named 'lower' and 'upper' for interval bounds
		y_intervals_train_df = pd.DataFrame(y_intervals_train, columns=["lower", "upper"])
		## convert the numpy array of prediction intervals on testing data into a pandas DataFrame
		y_intervals_test_df = pd.DataFrame(y_intervals_test, columns=["lower", "upper"])
		## print a message
		message_alpha = (
			"The prediction intervals (i.e., "
			+ str(round((1 - ALPHA) * 100, 1))
			+ "%) were calculated using ResidualQuantileWrapper with α = "
			+ str(ALPHA)
		)
		print(message_alpha)
		## concatenate horizontally dataframes
		### from the training dataset
		combined_train_df = pd.concat(
			[combined_train_df.reset_index(drop=True), # avoids index misalignment during concatenation
			y_intervals_train_df.reset_index(drop=True)], # avoids index misalignment during concatenation
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)
		### from the testing dataset
		combined_test_df = pd.concat(
			[combined_test_df.reset_index(drop=True),  # avoids index misalignment during concatenation
			y_intervals_test_df.reset_index(drop=True)], # avoids index misalignment during concatenation
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)

		# round digits of a dataframe avoid SettingWithCopyWarning with .copy()
		## from the training dataset
		combined_train_df = combined_train_df.copy()
		numeric_cols_combined_train = combined_train_df.select_dtypes(include='number').columns
		combined_train_df[numeric_cols_combined_train] = combined_train_df[numeric_cols_combined_train].round(digits)
		## from the testing dataset
		combined_test_df = combined_test_df.copy()
		numeric_cols_combined_test = combined_test_df.select_dtypes(include='number').columns
		combined_test_df[numeric_cols_combined_test] = combined_test_df[numeric_cols_combined_test].round(digits)

		# build a clean phenotype/dataset file to reuse later
		## keep only sample identifiers and the true phenotype value
		simplified_train_df = combined_train_df[["sample", "expectation"]].copy()
		simplified_test_df  = combined_test_df[["sample", "expectation"]].copy()
		## annotate dataset origin
		simplified_train_df["dataset"] = "training"
		simplified_test_df["dataset"]  = "testing"
		## concatenate vertically dataframes
		simplified_train_test_df = pd.concat(
			[simplified_train_df, 
			simplified_test_df],
			ignore_index=True,
			axis=0, join="inner" # safeguards against accidental column misalignment down the line
		)
		## rename 'expectation' to 'phenotype' for clarity
		simplified_train_test_df = simplified_train_test_df.rename(
			columns={"expectation": "phenotype"}
		).copy()
		## make sure sample identifiers are unique and sorted, then reset index
		simplified_train_test_df = (
			simplified_train_test_df
			.sort_values("sample")
			.reset_index(drop=True)
		)

		# check if the output directory does not exists and make it
		if not os.path.exists(OUTPUTPATH):
			os.makedirs(OUTPUTPATH)
			message_output_directory = "The output directory was created successfully"
			print(message_output_directory)
		else:
			message_output_directory = "The output directory already existed"
			print(message_output_directory)

		# step control
		step1_end = dt.datetime.now()
		step1_diff = step1_end - step1_start

		# output results
		## output path
		outpath_features = OUTPUTPATH + '/' + PREFIX + '_features' + '.obj'
		outpath_feature_encoder = OUTPUTPATH + '/' + PREFIX + '_feature_encoder' + '.obj'
		outpath_calibration_features =  OUTPUTPATH + '/' + PREFIX + '_calibration_features' + '.obj'
		outpath_calibration_targets =  OUTPUTPATH + '/' + PREFIX + '_calibration_targets' + '.obj'
		outpath_model = OUTPUTPATH + '/' + PREFIX + '_model' + '.obj'
		outpath_scores_parameters = OUTPUTPATH + '/' + PREFIX + '_scores_parameters' + '.tsv'
		outpath_feature_importance = OUTPUTPATH + '/' + PREFIX + '_feature_importances' + '.tsv'
		if PERMUTATIONIMPORTANCE is True:
			outpath_permutation_importance = OUTPUTPATH + '/' + PREFIX + '_permutation_importances' + '.tsv'
		outpath_metrics_global_train = OUTPUTPATH + '/' + PREFIX + '_metrics_global_training' + '.tsv'
		outpath_metrics_global_test = OUTPUTPATH + '/' + PREFIX + '_metrics_global_testing' + '.tsv'
		outpath_train = OUTPUTPATH + '/' + PREFIX + '_prediction_training' + '.tsv'
		outpath_test = OUTPUTPATH + '/' + PREFIX + '_prediction_testing' + '.tsv'
		outpath_phenotype_dataset = OUTPUTPATH + '/' + PREFIX + '_phenotype_dataset' + '.tsv'
		outpath_log = OUTPUTPATH + '/' + PREFIX + '_modeling_log' + '.txt'
		## write output in a tsv file
		all_scores_parameters_df.to_csv(outpath_scores_parameters, sep="\t", index=False, header=True)
		feature_importance_df.to_csv(outpath_feature_importance, sep="\t", index=False, header=True)
		if PERMUTATIONIMPORTANCE is True:
			permutation_importance_df.to_csv(outpath_permutation_importance, sep="\t", index=False, header=True)
		metrics_global_train_df.to_csv(outpath_metrics_global_train, sep="\t", index=False, header=True)
		metrics_global_test_df.to_csv(outpath_metrics_global_test, sep="\t", index=False, header=True)
		combined_train_df.to_csv(outpath_train, sep="\t", index=False, header=True)
		combined_test_df.to_csv(outpath_test, sep="\t", index=False, header=True)
		simplified_train_test_df.to_csv(outpath_phenotype_dataset, sep="\t", index=False, header=True)
		## save the training features
		with open(outpath_features, 'wb') as file:
			pi.dump(features, file)
		## save the fitted encoder
		with open(outpath_feature_encoder, 'wb') as file:
			pi.dump(encoder, file)
		## save the calibration features
		with open(outpath_calibration_features, 'wb') as file:
			pi.dump(X_train_encoded, file)
		## save the calibration targets
		with open(outpath_calibration_targets, 'wb') as file:
			pi.dump(y_train, file)
		## save the model
		with open(outpath_model, 'wb') as file:
			pi.dump(best_model, file)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["########################\n####### context  #######\n########################\n"])
		print(context, file=log_file)
		log_file.writelines(["########################\n###### reference  ######\n########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["########################\n##### repositories #####\n########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["########################\n### acknowledgements ###\n########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["########################\n####### versions #######\n########################\n"])
		log_file.writelines("GenomicBasedRegression: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("boruta: " + str(im.version("boruta")) + "\n")
		log_file.writelines("catboost: " + str(cb.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("re: " + str(re.__version__) + "\n")
		log_file.writelines("scipy: " + str(sp.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(im.version("tqdm-joblib")) + "\n")	
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines(["########################\n####### arguments ######\n########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["########################\n######## checks ########\n########################\n"])
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_limit + "\n")
		log_file.writelines(message_input_mutations + "\n")
		log_file.writelines(message_input_phenotypes + "\n")
		log_file.writelines(message_missing_phenotypes + "\n")
		log_file.writelines(message_expected_datasets + "\n")
		log_file.writelines(message_sample_identifiers + "\n")
		log_file.writelines(message_compatibility_dataset_slitting + "\n")
		log_file.writelines(message_compatibility_dataset_quantiles + "\n")
		log_file.writelines(message_dataset + "\n")
		log_file.writelines(message_differences_distributions + "\n")
		log_file.writelines(message_count_samples + "\n")
		log_file.writelines(message_missing_features + "\n")
		log_file.writelines(message_extra_features + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_feature_selection + "\n")
		log_file.writelines(message_regressor + "\n")
		log_file.writelines(message_pipeline + "\n")
		log_file.writelines(message_parameters + "\n")
		log_file.writelines(message_metrics_cv + "\n")
		log_file.writelines(message_best_parameters + "\n")
		log_file.writelines(message_best_score + "\n")
		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_importance_encoded_feature_names + "\n")
		log_file.writelines(message_importance_count + "\n")
		log_file.writelines(message_alpha + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["########################\n###### execution #######\n########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		total_secs = step1_diff.total_seconds() # store total seconds before modification
		secs = total_secs # use a working copy for breakdown
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration = 'The script lasted {} days, {} hrs, {} mins and {} secs (i.e., {} secs in total)'.format(
			int(days), int(hrs), int(mins), secs, round(total_secs, 2)
		)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["########################\n##### output files #####\n########################\n"])
		log_file.writelines(outpath_features + "\n")
		log_file.writelines(outpath_feature_encoder + "\n")
		log_file.writelines(outpath_calibration_features + "\n")
		log_file.writelines(outpath_calibration_targets + "\n")
		log_file.writelines(outpath_model + "\n")
		log_file.writelines(outpath_scores_parameters + "\n")
		log_file.writelines(outpath_feature_importance + "\n")
		if PERMUTATIONIMPORTANCE is True:
			log_file.writelines(outpath_permutation_importance + "\n")
		log_file.writelines(outpath_metrics_global_train + "\n")
		log_file.writelines(outpath_metrics_global_test + "\n")
		log_file.writelines(outpath_train + "\n")
		log_file.writelines( outpath_test+ "\n")
		log_file.writelines(outpath_phenotype_dataset + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["########################\n# performance  metrics #\n########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_global_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_global_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"Note: RMSE stands for root mean squared error. \n")
		log_file.writelines(f"Note: MSE stands for mean square error. \n")	
		log_file.writelines(f"Note: MAPE stands for mean absolute percentage error. \n")
		log_file.writelines(f"Note: MAE stands for mean absolute error. \n")
		log_file.writelines(f"Note: R2 stands for R-squared. \n")
		log_file.writelines(["########################\n### training dataset ###\n########################\n"])
		print(combined_train_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: Lower and upper correspond to the range of the prediction intervals. \n")
		log_file.writelines(["########################\n### testing  dataset ###\n########################\n"])
		print(combined_test_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: Lower and upper correspond to the range of the prediction intervals. \n")
		log_file.close()

	elif args.subcommand == 'prediction':

		# print a message about subcommand
		message_subcommand = "The prediction subcommand was used"
		print(message_subcommand)

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		### check the input file of mutations
		#### calculate the number of rows
		rows_mutations = len(df_mutations)
		#### calculate the number of columns
		columns_mutations = len(df_mutations.columns)
		#### check if at least one sample and 3 columns
		if (rows_mutations >= 1) and (columns_mutations >= 3): 
			message_input_mutations = "The minimum required number of samples in the dataset (i.e., >= 1) and the expected number of columns (i.e., >= 3) in the input file of mutations were properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + ", respectively)"
			print(message_input_mutations)
		else: 
			message_input_mutations = "The minimum required number of samples in the dataset (i.e., 1) and the expected number of columns (i.e., >= 3) in the input file of mutations were not properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + ", respectively)"
			raise Exception(message_input_mutations)
		## training features
		with open(INPUTPATH_FEATURES, 'rb') as file:
			features = pi.load(file)
		## feature encoder
		with open(INPUTPATH_FEATURE_ENCODER, 'rb') as file:
			feature_encoder = pi.load(file)
		## calibration features
		with open(INPUTPATH_CALIBRATION_FEATURES, 'rb') as file:
			X_calib = pi.load(file)
		## calibration targets
		with open(INPUTPATH_CALIBRATION_TARGETS, 'rb') as file:
			y_calib = pi.load(file)
		## model
		with open(INPUTPATH_MODEL, 'rb') as file:
			loaded_model = pi.load(file)
		
		# prepare data
		## replace missing genomic data by a string
		df_mutations = df_mutations.fillna('missing')
		## rename labels of headers
		df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
		## sort by samples
		df_mutations = df_mutations.sort_values(by='sample')
		## prepare mutations indexing the sample columns
		X_mutations = df_mutations.set_index('sample')

		# encode categorical data into binary data using the one-hot encoder from the modeling subcommand
		## check missing features
		missing_features = set(features) - set(X_mutations.columns)
		if missing_features:
			message_missing_features = "The following training features expected by the one-hot encoder are missing in the input tested mutations: " + str(sorted(missing_features))
			raise Exception(message_missing_features)
		else: 
			message_missing_features = "The input tested mutations include all features required by the trained one-hot encoder"
			print (message_missing_features)
		## check extra features
		extra_features = set(X_mutations.columns) - set(features)
		if extra_features:
			message_extra_features = "The following unexpected features in the input tested mutations will be ignored for one-hot encoding: " + str(sorted(extra_features))
			print (message_extra_features)
		else: 
			message_extra_features = "The input tested mutations contain no unexpected features for one hot encoding"
			print (message_extra_features)
		## ensure feature column order and cast to str
		X_features_str = X_mutations[features].astype(str)
		## apply the trained OneHotEncoder to the selected input features
		X_mutations_encoded = feature_encoder.transform(X_features_str)

		# count features for diagnostics
		## count the number of raw categorical features before one-hot encoding
		features_before_ohe_int = len(features)
		## count the number of binary features after one-hot encoding
		features_after_ohe_int = X_mutations_encoded.shape[1]
		## print a message
		message_ohe_features = "The " + str(features_before_ohe_int) + " provided features were one-hot encoded into " + str(features_after_ohe_int) + " encoded features"
		print(message_ohe_features)
		## count the number of features used by the model
		selected_features_int = count_selected_features(loaded_model, X_mutations_encoded)
		## print a message
		message_selected_features = "The pipeline expected " + str(selected_features_int) + " one-hot encoded features to perform prediction"
		print(message_selected_features)

		# detect the loaded model and perform prediction
		## detect the loaded model
		detected_model = loaded_model.__class__.__name__
		## print a message
		message_detected_model = "The pipeline components of the provided best model were properly recognized: " + re.sub(r'\s+', ' ', str(loaded_model)).strip()
		print(message_detected_model)
		## perform prediction
		y_pred_mutations = loaded_model.predict(X_mutations_encoded)

		# prepare output results
		## transform prediction array to DataFrame
		y_pred_mutations_df = pd.DataFrame(y_pred_mutations)
		## retrieve sample identifiers and rename the column
		y_samples_df = pd.DataFrame(
			X_mutations_encoded.reset_index().iloc[:, 0]
		).rename(columns={X_mutations_encoded.reset_index().columns[0]: "sample"})
		## concatenate horizontally dataframes
		combined_mutations_df = pd.concat(
			[y_samples_df.reset_index(drop=True), 
			y_pred_mutations_df.reset_index(drop=True)],
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)
		## rename prediction column
		combined_mutations_df.rename(columns={0: 'prediction'}, inplace=True)

		# retrieve only prediction intervals using a custom ResidualQuantileWrapper independantly of mapie 0.9.2 to be able to manage only one sample
		## fit the wrapper with loaded model and calibration data
		res_wrapper = ResidualQuantileWrapper(estimator=loaded_model, alpha=ALPHA, prefit=True)
		res_wrapper.fit(X_calib, y_calib)
		## predict with prediction intervals
		y_pred_mutations_res_wrapper, y_intervals_mutations = res_wrapper.predict(
			X_mutations_encoded,
			return_prediction_interval=True
		)
		## convert prediction intervals to DataFrame
		y_intervals_mutations_df = pd.DataFrame(
			y_intervals_mutations,
			columns=["lower", "upper"]
		)
		## print a message
		message_alpha = "The prediction intervals (i.e., " + str(((1-ALPHA)*100)) + "%) were calculated using a significance level of α = " + str(ALPHA)
		print(message_alpha)
		## concatenate intervals with predictions
		combined_mutations_df = pd.concat(
			[combined_mutations_df.reset_index(drop=True),
			y_intervals_mutations_df.reset_index(drop=True)],
			axis=1, join="inner"
		)

		# round digits of a dataframe avoid SettingWithCopyWarning with .copy()
		combined_mutations_df = combined_mutations_df.copy()
		numeric_cols_combined_mutations = combined_mutations_df.select_dtypes(include='number').columns
		combined_mutations_df[numeric_cols_combined_mutations] = combined_mutations_df[numeric_cols_combined_mutations].round(digits)

		# check if the output directory does not exists and make it
		if not os.path.exists(OUTPUTPATH):
			os.makedirs(OUTPUTPATH)
			message_output_directory = "The output directory was created successfully"
			print(message_output_directory)
		else:
			message_output_directory = "The output directory already existed"
			print(message_output_directory)

		# step control
		step1_end = dt.datetime.now()
		step1_diff = step1_end - step1_start

		# output results
		## output path
		outpath_prediction = OUTPUTPATH + '/' + PREFIX + '_prediction' + '.tsv'
		outpath_log = OUTPUTPATH + '/' + PREFIX + '_prediction_log' + '.txt'
		## write output in a tsv file
		combined_mutations_df.to_csv(outpath_prediction, sep="\t", index=False, header=True)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["########################\n####### context  #######\n########################\n"])
		print(context, file=log_file)
		log_file.writelines(["########################\n###### reference  ######\n########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["########################\n##### repositories #####\n########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["########################\n### acknowledgements ###\n########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["########################\n####### versions #######\n########################\n"])
		log_file.writelines("GenomicBasedRegression: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("boruta: " + str(im.version("boruta")) + "\n")
		log_file.writelines("catboost: " + str(cb.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("re: " + str(re.__version__) + "\n")
		log_file.writelines("scipy: " + str(sp.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(im.version("tqdm-joblib")) + "\n")	
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines(["########################\n####### arguments ######\n########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["########################\n######## checks ########\n########################\n"])
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_input_mutations + "\n")
		log_file.writelines(message_missing_features + "\n")
		log_file.writelines(message_extra_features + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_detected_model + "\n")
		log_file.writelines(message_alpha + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["########################\n###### execution #######\n########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		total_secs = step1_diff.total_seconds() # store total seconds before modification
		secs = total_secs # use a working copy for breakdown
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration = 'The script lasted {} days, {} hrs, {} mins and {} secs (i.e., {} secs in total)'.format(
			int(days), int(hrs), int(mins), secs, round(total_secs, 2)
		)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["########################\n##### output files #####\n########################\n"])
		log_file.writelines(outpath_prediction + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["########################\n## prediction dataset ##\n########################\n"])
		print(combined_mutations_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Lower and upper correspond to the range of the prediction intervals. \n")
		log_file.close()

	# print final message
	print(message_duration)
	print("The results are ready: " + OUTPUTPATH)
	print(parser.epilog)

# identify the block which will only be run when the script is executed directly
if __name__ == "__main__":
	main()
