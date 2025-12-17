# required librairies
## pip3.12 install --force-reinstall pandas==2.2.2
## pip3.12 install --force-reinstall scipy==1.16.0
## pip3.12 install --force-reinstall scikit-learn==1.5.2
## pip3.12 install --force-reinstall catboost==1.2.8
## pip3.12 install --force-reinstall lightgbm==4.6.0
## pip3.12 install --force-reinstall xgboost==2.1.3
## pip3.12 install --force-reinstall numpy==1.26.4
## pip3.12 install --force-reinstall joblib==1.5.1
## pip3.12 install --force-reinstall tqdm==4.67.1
## pip3.12 install --force-reinstall tqdm-joblib==0.0.4
'''
# examples of commands with parameters
## without feature selection and with the ADA model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x ADA_FirstAnalysis -da random -sp 80 -q 10 -r ADA -k 5 -pa tuning_parameters_ADA.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/ADA_FirstAnalysis_features.obj -fe MyDirectory/ADA_FirstAnalysis_feature_encoder.obj -cf MyDirectory/ADA_FirstAnalysis_calibration_features.obj -ct MyDirectory/ADA_FirstAnalysis_calibration_targets.obj -t MyDirectory/ADA_FirstAnalysis_model.obj -o MyDirectory -x ADA_SecondAnalysis
## with the SKB feature selection and the BRI model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x BRI_FirstAnalysis -da manual -fs SKB -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/BRI_FirstAnalysis_features.obj -fe MyDirectory/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectory/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectory/BRI_FirstAnalysis_model.obj -o MyDirectory -x BRI_SecondAnalysis
## with the laSFM feature selection and the CAT model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CAT_FirstAnalysis -da manual -fs laSFM -r CAT -k 5 -pa tuning_parameters_CAT.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/CAT_FirstAnalysis_features.obj -fe MyDirectory/CAT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/CAT_FirstAnalysis_calibration_features.obj -ct MyDirectory/CAT_FirstAnalysis_calibration_targets.obj -t MyDirectory/CAT_FirstAnalysis_model.obj -o MyDirectory -x CAT_SecondAnalysis
## with the enSFM feature selection and the DT model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs enSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/DT_FirstAnalysis_calibration_features.obj -ct MyDirectory/DT_FirstAnalysis_calibration_targets.obj -t MyDirectory/DT_FirstAnalysis_model.obj -o MyDirectory -x DT_SecondAnalysis
## with the riSFM feature selection and the EN model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs riSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/EN_FirstAnalysis_calibration_features.obj -ct MyDirectory/EN_FirstAnalysis_calibration_targets.obj -t MyDirectory/EN_FirstAnalysis_model.obj -o MyDirectory -x EN_SecondAnalysis
## with the rfSFM feature selection and the ET model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x ET_FirstAnalysis -da manual -fs rfSFM -r ET -k 5 -pa tuning_parameters_ET.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/ET_FirstAnalysis_features.obj -fe MyDirectory/ET_FirstAnalysis_feature_encoder.obj -cf MyDirectory/ET_FirstAnalysis_calibration_features.obj -ct MyDirectory/ET_FirstAnalysis_calibration_targets.obj -t MyDirectory/ET_FirstAnalysis_model.obj -o MyDirectory -x ET_SecondAnalysis
## with the SKB feature selection and the GB model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs SKB -r GB -k 5 -pa tuning_parameters_GB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/GB_FirstAnalysis_calibration_features.obj -ct MyDirectory/GB_FirstAnalysis_calibration_targets.obj -t MyDirectory/GB_FirstAnalysis_model.obj -o MyDirectory -x GB_SecondAnalysis
## with the SKB feature selection and the HGB model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs SKB -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/HGB_FirstAnalysis_model.obj -o MyDirectory -x HGB_SecondAnalysis
## with the SKB feature selection and the HU model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HU_FirstAnalysis -da manual -fs SKB -r HU -k 5 -pa tuning_parameters_HU.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HU_FirstAnalysis_features.obj -fe MyDirectory/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HU_FirstAnalysis_calibration_features.obj -ct MyDirectory/HU_FirstAnalysis_calibration_targets.obj -t MyDirectory/HU_FirstAnalysis_model.obj -o MyDirectory -x HU_SecondAnalysis
## with the SKB feature selection and the KNN model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/KNN_FirstAnalysis_features.obj -fe MyDirectory/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectory/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectory/KNN_FirstAnalysis_model.obj -o MyDirectory -x KNN_SecondAnalysis
## with the SKB feature selection and the LA model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LA_FirstAnalysis_features.obj -fe MyDirectory/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LA_FirstAnalysis_calibration_features.obj -ct MyDirectory/LA_FirstAnalysis_calibration_targets.obj -t MyDirectory/LA_FirstAnalysis_model.obj -o MyDirectory -x LA_SecondAnalysis
## with the SKB feature selection and the LGBM model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectory/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectory/LGBM_FirstAnalysis_model.obj -o MyDirectory -x LGBM_SecondAnalysis
## with the SKB feature selection and the MLP model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/MLP_FirstAnalysis_features.obj -fe MyDirectory/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectory/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectory/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectory/MLP_FirstAnalysis_model.obj -o MyDirectory -x MLP_SecondAnalysis
## with the SKB feature selection and the NSV model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/NSV_FirstAnalysis_features.obj -fe MyDirectory/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectory/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectory/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectory/NSV_FirstAnalysis_model.obj -o MyDirectory -x NSV_SecondAnalysis
## with the SKB feature selection and the PN model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x PN_FirstAnalysis -da manual -fs SKB -r PN -k 5 -pa tuning_parameters_PN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/PN_FirstAnalysis_features.obj -fe MyDirectory/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/PN_FirstAnalysis_calibration_features.obj -ct MyDirectory/PN_FirstAnalysis_calibration_targets.obj -t MyDirectory/PN_FirstAnalysis_model.obj -o MyDirectory -x PN_SecondAnalysis
## with the SKB feature selection and the RF model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RF_FirstAnalysis_features.obj -fe MyDirectory/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RF_FirstAnalysis_calibration_features.obj -ct MyDirectory/RF_FirstAnalysis_calibration_targets.obj -t MyDirectory/RF_FirstAnalysis_model.obj -o MyDirectory -x RF_SecondAnalysis
## with the SKB feature selection and the RI model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RI_FirstAnalysis_features.obj -fe MyDirectory/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RI_FirstAnalysis_calibration_features.obj -ct MyDirectory/RI_FirstAnalysis_calibration_targets.obj -t MyDirectory/RI_FirstAnalysis_model.obj -o MyDirectory -x RI_SecondAnalysis
## with the SKB feature selection and the SVR model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/SVR_FirstAnalysis_features.obj -fe MyDirectory/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectory/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectory/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectory/SVR_FirstAnalysis_model.obj -o MyDirectory -x SVR_SecondAnalysis
## with the SKB feature selection and the XGB model regressor
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/XGB_FirstAnalysis_model.obj -o MyDirectory -x XGB_SecondAnalysis
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
import contextlib as ctl # no individual installation because is part of the Python Standard Library (no version)
import io as io # no individual installation because is part of the Python Standard Library (no version)
import threadpoolctl as tpc # no individual installation because is part of the Python Standard Library (no version)
## third-party libraries
import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sp
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, SelectFromModel
from sklearn import set_config
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor, Pool

# compatibility patch: prevent GridSearchCV from injecting random_state into CatBoost
class SafeCatBoostRegressor(CatBoostRegressor):
    """a subclass of CatBoostRegressor that safely ignores sklearn random_state parameter."""
    def set_params(self, **params):
		# Drop sklearn’s automatic random_state injection to avoid CatBoostError
        params.pop("random_state", None)
        return super().set_params(**params)

# set static metadata to keep outside the main function
## set workflow repositories
repositories = 'Please cite:\n GitHub (https://github.com/Nicolas-Radomski/GenomicBasedRegression),\n Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedregression),\n and/or Anaconda Hub (https://anaconda.org/nicolasradomski/genomicbasedregression).'
## set the workflow context
context = "The scikit-learn (sklearn)-based Python workflow independently supports both modeling (i.e., training and testing) and prediction (i.e., using a pre-built model), and implements 5 feature selection methods, 19 model regressors, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction interval estimation, execution monitoring via progress bars, and parallel processing."
## set the workflow reference
reference = "An article might potentially be published in the future."
## set the acknowledgement
acknowledgements = "Many thanks to Andrea De Ruvo, Adriano Di Pasquale and ChatGPT for the insightful discussions that helped improve the algorithm."
## set the version and release
__version__ = "1.3.0"
__release__ = "December 2025"

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

def count_selected_features(pipeline, encoded_matrix):
	"""
	robust count of features the pipeline expects
	returns the number of columns reaching the final estimator
	handles both Pipeline objects and direct estimators
	"""
	# ensure the object is a pipeline; wrap standalone estimators
	if not hasattr(pipeline, "named_steps"):
		pipeline = Pipeline([("model", pipeline)])
	# ensure encoded_matrix preserves feature names
	if not hasattr(encoded_matrix, "columns"):
		raise ValueError(
			"encoded_matrix must be a pandas DataFrame with feature names "
			"to safely count selected features"
		)
	# check if a feature selection step exists
	if "feature_selection" in pipeline.named_steps:
		fs = pipeline.named_steps["feature_selection"]
		# support_ is the most reliable and warning-free indicator
		if hasattr(fs, "get_support"):
			try:
				support = fs.get_support()
				return int(np.sum(support))
			except Exception:
				pass
		# fallback: selector exists but does not expose support_
		# assume no dimensionality reduction occurred
		return int(encoded_matrix.shape[1])
	# no explicit selector → check the estimator directly
	est = pipeline.named_steps.get("model", pipeline)
	# sklearn 1.3+ compatibility
	n_feat = getattr(est, "n_features_in_", None)
	if n_feat is None and hasattr(est, "feature_names_in_"):
		n_feat = len(est.feature_names_in_)
	# CatBoost, XGB, HGB often hide n_features_in_
	if n_feat is None or n_feat == 0:
		n_feat = encoded_matrix.shape[1]
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
		#print(f"[ResidualQuantileWrapper] Residual quantile bounds set to ±{self.upper_quantile:.4f} for alpha = {self.alpha}")
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

def restricted_int_digits(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 0
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is negative.
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 0:
		raise ap.ArgumentTypeError("digits must be an integer ≥ 0")
	return x

def restricted_debug_level(x: str) -> int:
	"""
	convert *x* to int and ensure x >= 0.
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is negative.
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 0:
		raise ap.ArgumentTypeError("debug must be zero or a positive integer (0, 1, 2, ...)")
	return x

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
		help='Acronym of the regression-compatible feature selection method to use: SelectKBest (SKB), SelectFromModel with lasso (laSFM), SelectFromModel with elasticnet (enSFM), or SelectFromModel with ridge (riSFM), or SelectFromModel with random forest (rfSFM). These methods are suitable for high-dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-r', '--regressor', 
		dest='regressor', 
		type=str,
		action='store', 
		required=False, 
		default='XGB', 
		help='Acronym of the regressor to use among adaboost (ADA), bayesian ridge (BRI), catboost (CAT), decision tree (DT), elasticnet (EN), extra trees (ET), gradient boosting (GB), histogram-based gradient boosting (HGB), huber (HU), k-nearest neighbors (KNN), lassa (LA), light gradient boosting machine (LGBM), multi-layer perceptron (MLP), nu support vector (NSV), polynomial (PN), ridge (RI), random forest (RF), support vector regressor (SVR) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]'
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
		help='Number of repetitions per feature for permutation importance; higher values provide more stable estimates but increase runtime. [OPTIONAL, DEFAULT: 10]'
		)
	parser_modeling.add_argument(
		'-a', '--alpha', 
		dest='alpha', 
		type=restricted_float_alpha, # control (0, 1) open interval
		action='store', 
		required=False, 
		default=0.05, 
		help='Significance level alpha (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − alpha) × 100]%% coverage. [OPTIONAL, DEFAULT: 0.05]'
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
		'-di', '--digits', 
		dest='digits', 
		type=restricted_int_digits, # control >= 0
		action='store', 
		required=False, 
		default=6, 
		help='Number of decimal digits to round numerical results (e.g., root mean squared error, importance, metrics). [OPTIONAL, DEFAULT: 6]'
		)
	parser_modeling.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=restricted_debug_level, # control >= 0
		action='store', 
		required=False, 
		default=0, 
		help='Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]'
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
		help='Significance level alpha (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − alpha) × 100]%% coverage. [OPTIONAL, DEFAULT: 0.05]'
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
		'-di', '--digits', 
		dest='digits', 
		type=restricted_int_digits, # control >= 0
		action='store', 
		required=False, 
		default=6, 
		help='Number of decimal digits to round numerical results (e.g., root mean squared error, importance, metrics). [OPTIONAL, DEFAULT: 6]'
		)
	parser_prediction.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=restricted_debug_level, # control >= 0
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
		DIGITS=args.digits
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
		DIGITS=args.digits
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
		if sp.__version__ != "1.16.0":
			raise Exception("scipy 1.16.0 version is recommended")
		if pd.__version__ != "2.2.2":
			raise Exception('pandas 2.2.2 version is recommended')
		if sk.__version__ != "1.5.2":
			raise Exception('sklearn 1.5.2 version is recommended')
		if pi.format_version != "4.0":
			raise Exception('pickle 4.0 version is recommended')
		if cb.__version__ != "1.2.8":
			raise Exception('catboost 1.2.8 version is recommended')
		if lgbm.__version__ != "4.6.0":
			raise Exception("lightgbm 4.6.0 version is recommended")
		if xgb.__version__ != "2.1.3":
			raise Exception("xgboost 2.1.3 version is recommended")
		if np.__version__ != "1.26.4":
			raise Exception("numpy 1.26.4 version is recommended")
		if jl.__version__ != "1.5.1":
			raise Exception('joblib 1.5.1 version is recommended')
		if tq.__version__ != "4.67.1":
			raise Exception('tqdm 4.67.1 version is recommended')
		if im.version("tqdm-joblib") != "0.0.4":
			raise Exception("tqdm-joblib 0.0.4 version is recommended")
		message_versions = 'The recommended versions of Python and packages were properly controlled'
	else:
		message_versions = 'The recommended versions of Python and packages were not controlled'

	# print a message about version control
	print(message_versions)

	# set rounded digits
	digits = DIGITS

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

		# define minimal limits of samples (i.e., 2 * LIMIT per dataset)
		limit_samples = 2 * LIMIT

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		## phenotypes
		df_phenotypes = pd.read_csv(INPUTPATH_PHENOTYPES, sep='\t', dtype=str)

		# transform the phenotype as numeric
		## make sure the phenotype column exists (i.e., the second column)
		if df_phenotypes.shape[1] < 2:
			message_phenotype_numeric = ("The presence of phenotype in the input file of phenotypes was improperly controlled (i.e., the second column is missing)")
			raise Exception(message_phenotype_numeric)
		## extract the phenotype column
		phenotype_col = df_phenotypes.iloc[:, 1]
		## make sure that the phenotype column can be transformed as numeric
		elif_invalid = pd.to_numeric(phenotype_col, errors="coerce").isna().any()
		if elif_invalid:
			message_phenotype_numeric = ("The phenotype in the input file of phenotypes cannot be transformed as numeric (i.e., the second column contains non-numeric values)")
			raise Exception(message_phenotype_numeric)
		else:
			# convert the phenotype column to numeric
			df_phenotypes.iloc[:, 1] = pd.to_numeric(phenotype_col)
			message_phenotype_numeric = ("The phenotype in the input file of phenotypes was properly transformed as numeric (i.e., the second column)")

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

		# perform splitting of the training and testing datasets according to the setting
		if DATASET == 'random':
			message_dataset = "The training and testing datasets were constructed based on the 'random' setting"
			print(message_dataset)
			# drop dataset column (since it is irrelevant in random mode)
			df_phenotypes = df_phenotypes.drop("dataset", axis='columns')
			# deterministic merge of phenotypes and mutations on sample identifier
			# ensures perfect alignment and prevents any mis-ordered sample mixing
			df_all = (
				pd.merge(df_phenotypes, df_mutations, on="sample", how="inner", validate="one_to_one")
				.sort_values(by="sample") # enforce deterministic ordering
				.reset_index(drop=True)
			)
			# build X (mutations) and y (phenotypes) with sample index preserved
			X = df_all.drop(columns=["phenotype"]).set_index("sample")
			y = df_all[["sample", "phenotype"]].set_index("sample")["phenotype"]
			# quantile binning of continuous phenotypes to enable stratified splitting
			y_binned = pd.qcut(
				y,
				q=QUANTILES,
				labels=False,
				duplicates='drop' # avoids ValueError when bins overlap
			)
			# deterministic stratified split
			# random_state=None allows true randomness; use 42 for reproducibility if desired
			splitter = StratifiedShuffleSplit(
				n_splits=1,
				train_size=SPLITTING / 100,
				random_state=None # no reproducibility by design
			)
			for train_idx, test_idx in splitter.split(X, y_binned):
				X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
				y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
			# y_train and y_test are already 1D Series (phenotype only)
		elif DATASET == 'manual':
			message_dataset = "The training and testing datasets were constructed based on the 'manual' setting"
			print(message_dataset)
			# merge phenotypes and mutations deterministically
			df_all = (
				pd.merge(df_phenotypes, df_mutations, on="sample", how="inner", validate="one_to_one")
				.sort_values(by="sample")
				.reset_index(drop=True)
			)
			# normalize only here (since dataset column exists)
			df_all["dataset"] = df_all["dataset"].astype(str).str.strip().str.lower()
			# split according to dataset column
			df_training = df_all[df_all["dataset"] == "training"]
			df_testing  = df_all[df_all["dataset"] == "testing"]
			# build X and y dataframes for training/testing
			## extract numerical genomic features and set sample identifiers as index
			X_train = df_training.drop(columns=["phenotype", "dataset"]).set_index("sample")
			y_train = df_training[["sample", "phenotype"]].set_index("sample")["phenotype"]
			## extract phenotype as a clean 1-D Series indexed by sample
			X_test  = df_testing .drop(columns=["phenotype", "dataset"]).set_index("sample")
			y_test  = df_testing [ ["sample", "phenotype"]].set_index("sample")["phenotype"]

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
		## assert identical encoded features between training and testing datasets
		train_encoded_features = list(X_train_encoded.columns)
		test_encoded_features = list(X_test_encoded.columns)
		if train_encoded_features != test_encoded_features:
			message_assert_encoded_features = "The encoded features between training and testing datasets do not match"
			raise AssertionError(message_assert_encoded_features)
		else:
			message_assert_encoded_features = "The encoded features between training and testing datasets were confirmed as identical"
			print(message_assert_encoded_features)

		# enforce consistent one-hot encoded column order between train/test and across runs
		X_train_encoded = X_train_encoded.reindex(sorted(X_train_encoded.columns), axis=1)
		X_test_encoded  = X_test_encoded.reindex(sorted(X_train_encoded.columns), axis=1, fill_value=0)
		message_column_order = ("The one-hot encoded column order was harmonized across training and testing datasets to ensure deterministic feature alignment for feature selection and modeling")
		print(message_column_order)

		# count features
		## count the number of raw categorical features before one-hot encoding
		features_before_ohe_int = len(features)
		## count the number of binary features after one-hot encoding
		features_after_ohe_int = X_train_encoded.shape[1]
		## print a message
		message_ohe_features = "The " + str(features_before_ohe_int) + " provided features were one-hot encoded into " + str(features_after_ohe_int) + " encoded features"
		print(message_ohe_features)
		
		# prepare elements of the model
		## initialize the feature selection method (without tuning parameters: deterministic randomness (random_state=42) without repeatability for parallel nondeterminism (n_jobs=1 or thread_count=1))
		if FEATURESELECTION == 'None':
			message_feature_selection = "The provided feature selection method was properly recognized: None"
			print(message_feature_selection)
			selected_feature_selector = None
		elif FEATURESELECTION == 'SKB':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectKBest (SKB)"
			print(message_feature_selection)
			selected_feature_selector = SelectKBest(
				score_func=ft.partial(  # partial ensures deterministic behavior
					mutual_info_regression, # captures linear + nonlinear dependencies; preferred over f_regression (linear only)
					random_state=42 # deterministic
				),
				k=10 # default top k features; user may override in the parameter file
			)
		elif FEATURESELECTION == 'laSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with Lasso (laSFM)"
			print(message_feature_selection)
			selector_model = Lasso(random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'enSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with ElasticNet (enSFM)"
			print(message_feature_selection)
			selector_model = ElasticNet(random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'riSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with ridge (riSFM)"
			print(message_feature_selection) 
			# ridge regression does not produce exact zero coefficients; feature selection is therefore based on coefficient magnitude and should be interpreted as a soft, ranking-based filter
			selector_model = Ridge(random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'rfSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with Random Forest (rfSFM)"
			print(message_feature_selection)
			selector_model = RandomForestRegressor(
				random_state=42, # deterministic
				bootstrap=False # disable bootstrapping to reduce random variability in feature importance computation
			)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		else:
			message_feature_selection = "The provided feature selection method is not implemented yet"
			raise Exception(message_feature_selection)
		## initialize the regressor (without tuning parameters: deterministic randomness (random_state=42) without repeatability for parallel nondeterminism (n_jobs=1 or thread_count=1))
		if REGRESSOR == 'ADA':
			message_regressor = "The provided regressor was properly recognized: AdaBoost (ADA)"
			print(message_regressor)
			selected_regressor = AdaBoostRegressor(random_state=42) # adaboost (ADA)
		elif REGRESSOR == 'BRI':
			message_regressor = "The provided regressor was properly recognized: bayesian ridge (BRI)"
			print(message_regressor)
			selected_regressor = BayesianRidge() # bayesian ridge (BRI)
		elif REGRESSOR == 'CAT':
			message_regressor = "The provided regressor was properly recognized: CatBoost (CAT)"
			print(message_regressor)
			# deterministic configuration for CatBoost
			# - use random_seed (not random_state) to prevent synonym conflicts
			# - allow_writing_files=False disables CatBoost automatic file outputs
			# - bootstrap_type='Bayesian' + random_strength=0 ensure reproducible splits
			# - verbose=False keeps logs clean and avoids stdout clutter
			selected_regressor = SafeCatBoostRegressor(
				loss_function="RMSE", # default loss for regression
				random_seed=42, # deterministic
				verbose=False, # no stdout spam
				allow_writing_files=False, # prevent auto logging files
				bootstrap_type="Bayesian", # deterministic bootstrap
				random_strength=0 # must be 0 for reproducibility
			) # catboost (CAT)
		elif REGRESSOR == 'DT':
			message_regressor = "The provided regressor was properly recognized: decision tree (DT)"
			print(message_regressor)
			selected_regressor = DecisionTreeRegressor(random_state=42) # decision tree (DT)
		elif REGRESSOR == 'EN':
			message_regressor = "The provided regressor was properly recognized: elasticNet (EN)"
			print(message_regressor)
			selected_regressor = ElasticNet(random_state=42, selection='random') # elasticnet (EN)
		elif REGRESSOR == 'ET':
			message_regressor = "The provided regressor was properly recognized: extra trees (ET)"
			print(message_regressor)
			selected_regressor = ExtraTreesRegressor(random_state=42) # extra trees (ET)
		elif REGRESSOR == 'GB':
			message_regressor = "The provided regressor was properly recognized: gradient boosting (GB)"
			print(message_regressor)
			selected_regressor = GradientBoostingRegressor(random_state=42) # gradient boosting (GB)
		elif REGRESSOR == 'HGB':
			message_regressor = "The provided regressor was properly recognized: histogram-based gradient boosting (HGB)"
			print(message_regressor)
			selected_regressor = HistGradientBoostingRegressor(random_state=42) # histogram-based gradient boosting (HGB)
		elif REGRESSOR == 'HU':
			message_regressor = "The provided regressor was properly recognized: huber (HU)"
			print(message_regressor)
			selected_regressor = HuberRegressor() # huber (HU)
		elif REGRESSOR == 'KNN':
			message_regressor = "The provided regressor was properly recognized: k-nearest neighbors (KNN)"
			print(message_regressor)
			selected_regressor = KNeighborsRegressor() # k-nearest neighbors (KNN)
		elif REGRESSOR == 'LA':
			message_regressor = "The provided regressor was properly recognized: lasso (LA)"
			print(message_regressor)
			selected_regressor = Lasso(random_state=42) # lasso (LA)
		elif REGRESSOR == 'LGBM':
			message_regressor = "The provided regressor was properly recognized: light gradient boosting machine (LGBM)"
			print(message_regressor)
			selected_regressor = lgbm.LGBMRegressor(random_state=42, verbose=-1) # light gradient boosting machine (LGBM)
		elif REGRESSOR == 'MLP':
			message_regressor = "The provided regressor was properly recognized: multi-layer perceptron (MLP)"
			print(message_regressor)
			selected_regressor = MLPRegressor(random_state=42) # multi-layer perceptron (MLP)
		elif REGRESSOR == 'NSV':
			message_regressor = "The provided regressor was properly recognized: nu support vector (NSV)"
			print(message_regressor)
			selected_regressor = NuSVR() # nu support vector (NSV)
		elif REGRESSOR == 'PN':
			message_regressor = "The provided regressor was properly recognized: polynomial (PN)"
			print(message_regressor)
			selected_regressor = LinearRegression() # polynomial (PN)
		elif REGRESSOR == 'RI':
			message_regressor = "The provided regressor was properly recognized: ridge (RI)"
			print(message_regressor)
			selected_regressor = Ridge() # ridge (RI)
		elif REGRESSOR == 'RF':
			message_regressor = "The provided regressor was properly recognized: random forest (RF)"
			print(message_regressor)
			selected_regressor = RandomForestRegressor(random_state=42) # random forest (RF)
		elif REGRESSOR == 'SVR':
			message_regressor = "The provided regressor was properly recognized: support vector regression (SVR)"
			print(message_regressor)
			selected_regressor = SVR() # support vector regression (SVR)
		elif REGRESSOR == 'XGB':
			message_regressor = "The provided regressor was properly recognized: extreme gradient boosting (XGB)"
			print(message_regressor)
			selected_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) # extreme gradient boosting (XGB)
		else: 
			message_regressor = "The provided regressor is not implemented yet"
			raise Exception(message_regressor)

		## build the pipeline
		### create an empty list
		steps = []
		### add feature selection step if specified
		if FEATURESELECTION in ['SKB', 'laSFM', 'enSFM', 'riSFM', 'rfSFM']:
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
		# especially for tree-based regressors (e.g., DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, HistGradientBoostingRegressor, LightGBM)
		X_train_encoded_float32 = X_train_encoded.astype(np.float32)
		X_test_encoded_float32  = X_test_encoded.astype(np.float32)
		# preserve sample IDs
		train_sample_ids = y_train.index
		test_sample_ids  = y_test.index
		# note: explicit astype(np.float32) casting of the target vector is required for LightGBM *regression* only,
		# because LightGBMRegressor strictly requires numeric labels (int/float/bool) and rejects pandas object dtypes;
		# this additional casting is not required in the classification workflow, where LightGBMClassifier tolerates
		# non-numeric or object-typed class labels
		# reconstruct clean 1-D numeric Series required by LightGBM, CatBoost and sklearn
		y_train_series = pd.Series(np.ravel(y_train.values).astype(np.float32), index=train_sample_ids)
		y_test_series = pd.Series(np.ravel(y_test.values).astype(np.float32), index=test_sample_ids)

		# check parallelization and print a message
		if JOBS == 1:
			message_parallelization = "The tqdm_joblib progress bars are deactivated when using one job"
			print(message_parallelization)
		else:
			message_parallelization = "The tqdm_joblib progress bars are activated when using two or more jobs"
			print(message_parallelization)

		## fit the model
		### use tqdm.auto rather than tqdm library because it automatically chooses the best display format (terminal, notebook, etc.)
		### use a tqdm progress bar from the tqdm_joblib library (compatible with GridSearchCV)
		### use a tqdm progress bar immediately after the last print (position=0), disable the additional bar after completion (leave=False), and allow for dynamic resizing (dynamic_ncols=True)
		### force GridSearchCV to use the threading backend to avoid the DeprecationWarning from fork and ChildProcessError from the loky backend (default in joblib)
		### threading is slower than loky, but it allows using a progress bar with GridSearchCV and avoids the DeprecationWarning and ChildProcessError
		if JOBS == 1:
			# when using a single thread, tqdm_joblib does not display intermediate progress updates
			# in this case, we run GridSearchCV normally without the tqdm_joblib wrapper
			with jl.parallel_backend('threading', n_jobs=JOBS):
				model.fit(
					X_train_encoded_float32,
					y_train_series
				)
		else:
			# when using multiple threads, tqdm_joblib correctly hooks into joblib and displays progress updates
			with tqa.tqdm(
				total=gridsearchcv_fits, 
				desc="Model building progress", 
				position=0, 
				leave=False, 
				dynamic_ncols=True
			) as progress_bar:
				with jl.parallel_backend('threading', n_jobs=JOBS):
					with tqjl.tqdm_joblib(progress_bar):
						model.fit(
							X_train_encoded_float32,
							y_train_series
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

		# re-fit only the final estimator (not the entire pipeline)
		# this is necessary because re-fitting the whole pipeline would overwrite the
		# feature selection mask learned during GridSearchCV
		# keeping the original mask ensures consistent selected features between
		# modeling and prediction subcommands
		## check if a feature selection step exists inside the pipeline
		if hasattr(best_model, "named_steps") and ("feature_selection" in best_model.named_steps):
			# retrieve feature selector and final estimator
			selector = best_model.named_steps["feature_selection"]
			final_estimator = best_model.named_steps["model"]
			# apply the selector to the harmonized one-hot encoded training matrix
			# keep a pandas DataFrame to preserve feature names and avoid sklearn warnings
			X_train_selected = selector.transform(
				X_train_encoded_float32
				if hasattr(X_train_encoded_float32, "columns")
				else X_train_encoded_float32.astype(np.float32)
			)
			# re-fit only the final estimator on selected features
			final_estimator.fit(X_train_selected, y_train_series)
			# rebuild the pipeline WITHOUT re-fitting the selector
			best_model = Pipeline([
				("feature_selection", selector),
				("model", final_estimator)
			])
		else:
			# no feature selection method was used
			# we directly retrieve the final estimator
			if hasattr(best_model, "named_steps"):
				final_estimator = best_model.named_steps["model"]
			else:
				final_estimator = best_model
			# re-fit the final estimator on the full harmonized one-hot encoded training matrix
			# (no feature selection → the estimator expects full encoded matrices)
			final_estimator.fit(X_train_encoded_float32, y_train_series)
			# rebuild the pipeline to maintain a consistent structure
			best_model = Pipeline([
				("model", final_estimator)
			])
		## refresh final estimator reference (required before feature importances and permutation importance)
		final_estimator = (
			best_model.named_steps["model"]
			if hasattr(best_model, "named_steps")
			else best_model
		)

		# count features
		## count the number of features selected by feature selection actually used by the final regressor
		selected_features_int = count_selected_features(best_model, X_train_encoded_float32)
		## print a message
		message_selected_features = (
			"The pipeline potentially selected and used "
			+ str(selected_features_int)
			+ " one-hot encoded features to train the model"
		)
		print(message_selected_features)

		# output a dataframe of features used by the final model with ranked importance scores
		## recover encoded feature names before feature selection
		if hasattr(best_model, "named_steps"):
			# polynomial features take priority
			if "poly" in best_model.named_steps and hasattr(
				best_model.named_steps["poly"], "get_feature_names_out"
			):
				input_features = X_train_encoded_float32.columns
				encoded_names = best_model.named_steps["poly"].get_feature_names_out(
					input_features=input_features
				)
			# otherwise recover from encoder
			elif "encoder" in best_model.named_steps and hasattr(
				best_model.named_steps["encoder"], "get_feature_names_out"
			):
				encoded_names = best_model.named_steps["encoder"].get_feature_names_out()
			else:
				encoded_names = X_train_encoded_float32.columns
		else:
			encoded_names = X_train_encoded_float32.columns
		## always convert to a clean Python list
		if hasattr(encoded_names, "tolist"):
			feature_encoded_lst_full = encoded_names.tolist()
		else:
			feature_encoded_lst_full = list(encoded_names)
		## determine feature selection mask
		if hasattr(best_model, "named_steps") and ("feature_selection" in best_model.named_steps):
			selector = best_model.named_steps["feature_selection"]
			if hasattr(selector, "get_support"):
				support_mask = selector.get_support()
			else:
				support_mask = np.ones(len(feature_encoded_lst_full), dtype=bool)
		else:
			support_mask = np.ones(len(feature_encoded_lst_full), dtype=bool)

		## filtered list for feature importance reporting (not for permutation importance)
		feature_encoded_lst_filtered = np.array(feature_encoded_lst_full)[support_mask].tolist()
		## print a message
		message_importance_encoded_feature_names = (
			"The one-hot encoded feature names were recovered before feature selection"
		)
		print(message_importance_encoded_feature_names)
		## extract feature importance depending on regressor type
		try:
			# catboost must come first (otherwise feature_importances_ steals the branch)
			if isinstance(final_estimator, cb.CatBoostRegressor):
				try:
					train_pool = cb.Pool(
						X_train_encoded_float32,
						y_train_series,
						feature_names=feature_encoded_lst_full
					)
					importances = final_estimator.get_feature_importance(
						train_pool,
						type="PredictionValuesChange"
					)
					importance_type = "catboost's loss-based importance (PredictionValuesChange)"
				except Exception:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = "NaN placeholder (CatBoost importance extraction failed)"
			# XGBoost regressor
			elif isinstance(final_estimator, xgb.XGBRegressor):
				booster = final_estimator.get_booster()
				booster_feature_names = booster.feature_names
				# enforce a valid importance_type (XGBoost requires it explicitly)
				xgb_importance_type = final_estimator.get_params().get("importance_type")
				if xgb_importance_type is None:
					xgb_importance_type = "gain"
				importance_dict = booster.get_score(importance_type=xgb_importance_type)
				importances = np.array([importance_dict.get(f, 0.0) for f in booster_feature_names])
				feature_encoded_lst_filtered = list(booster_feature_names)
				importance_type = f"xgboost's {xgb_importance_type}-based importance"
			# LightGBM regressor
			elif isinstance(final_estimator, lgbm.LGBMRegressor):
				try:
					lgbm_importance_type = final_estimator.get_params().get(
						"importance_type", "gain"
					)
					importances = final_estimator.booster_.feature_importance(
						importance_type=lgbm_importance_type
					)
					importance_type = f"lightgbm's {lgbm_importance_type}-based importance"
				except Exception:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = "NaN placeholder (LightGBM importance extraction failed)"
			# histogram-based GBDT
			elif isinstance(final_estimator, HistGradientBoostingRegressor):
				try:
					from sklearn.ensemble._hist_gradient_boosting.utils import get_feature_importances
					importances = get_feature_importances(final_estimator, normalize=True)
				except Exception:
					importances = np.array([])
				if importances is None or len(importances) == 0 or np.all(importances == 0):
					n_features = X_train_encoded_float32.shape[1]
					importances = np.zeros(n_features, dtype=float)
					for predictors in getattr(final_estimator, "_predictors", []):
						if predictors is None:
							continue
						for tree in np.atleast_1d(predictors):
							if hasattr(tree, "split_features_") and hasattr(tree, "split_gains_"):
								for feat, gain in zip(tree.split_features_, tree.split_gains_):
									if feat >= 0:
										importances[int(feat)] += gain
					total_gain = np.sum(importances)
					if total_gain > 0:
						importances /= total_gain
				importance_type = (
					"histogram-based mean impurity reduction"
					" (auto fallback to internal split gains)"
				)
			# support vector regression (SVR / NuSVR)
			elif isinstance(final_estimator, (SVR, NuSVR)):
				kernel_type = getattr(final_estimator, "kernel", "unknown")
				if kernel_type == "linear" and hasattr(final_estimator, "coef_"):
					importances = np.abs(final_estimator.coef_.ravel())
					importance_type = (
						"absolute coefficient magnitude (linear "
						+ final_estimator.__class__.__name__ + " coef_)"
					)
				else:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = (
						"NaN placeholder (no native importance for "
						+ kernel_type + " kernel "
						+ final_estimator.__class__.__name__ + ")"
					)
			# tree-based regressors (e.g., ADA, DT, ET, GB, RF)
			elif hasattr(final_estimator, "feature_importances_"):
				importances = final_estimator.feature_importances_
				importance_type = "tree-based impurity reduction (feature_importances_)"
			# linear models (e.g, BRI, EN, HU, LA, PN, RI)
			elif hasattr(final_estimator, "coef_"):
				coef = final_estimator.coef_
				coef = coef.ravel() if hasattr(coef, "ravel") else coef
				importances = np.abs(coef)
				importance_type = "absolute coefficient magnitude (coef_)"
			# default
			else:
				importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
				importance_type = "NaN placeholder (no native importance)"
		except Exception as e:
			importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
			importance_type = "NaN fallback due to extraction error: " + str(e)
		## handle mismatch between feature names and importances
		if len(importances) != len(feature_encoded_lst_filtered):
			min_len = min(len(importances), len(feature_encoded_lst_filtered))
			importances = importances[:min_len]
			feature_encoded_lst_filtered = feature_encoded_lst_filtered[:min_len]
		## print message about extracted importances
		if importance_type.startswith("NaN"):
			message_importance_count = (
				"The selected regressor did not expose feature importances natively ("
				+ importance_type + ")"
			)
		else:
			message_importance_count = (
				"The best model returned " + str(len(importances)) + 
				" importance values (" + importance_type + ") for " + 
				str(len(feature_encoded_lst_filtered)) + " one-hot encoded features"
			)
		print(message_importance_count)
		## ensure importances array is valid
		if importances is None or np.all(np.isnan(importances)):
			importances = np.full(len(feature_encoded_lst_filtered), np.nan)
		## create feature importance dataframe
		feature_importance_df = pd.DataFrame({
			"feature": feature_encoded_lst_filtered,
			"importance": np.round(importances, digits)
		}).sort_values(by="importance", ascending=False).reset_index(drop=True)

		# check compatibility between permutation importance and the number of repetitions
		set_nrepeats = ('-nr' in sys.argv) or ('--nrepeats' in sys.argv)
		if (PERMUTATIONIMPORTANCE is True) and (set_nrepeats is True):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") and the number of repetitions was explicitly set (i.e., " + str(set_nrepeats) +
				") to a specific value (i.e., " + str(NREPEATS) + ")"
			)
		elif (PERMUTATIONIMPORTANCE is True) and (set_nrepeats is False):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") but the number of repetitions was not set (i.e., " + str(set_nrepeats) +
				"); the default value is therefore used (i.e., " + str(NREPEATS) + ")"
			)
		elif (PERMUTATIONIMPORTANCE is False) and (set_nrepeats is True):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was not requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") but the number of repetitions was set (i.e., " + str(set_nrepeats) +
				"); this setting is consequently ignored (i.e., " + str(NREPEATS) + ")"
			)
		else:
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was not requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") and the number of repetitions was not set, as expected (i.e., " +
				str(set_nrepeats) + ")"
			)
		print(message_compatibility_permutation_nrepeat)

		# fix nested parallelism issues for random forest and extra trees so tqdm_joblib stays accurate
		if REGRESSOR in ['RF', 'ET']:
			try:
				best_model.set_params(model__n_jobs=1)
			except Exception:
				if hasattr(best_model, "n_jobs"):
					best_model.set_params(n_jobs=1)

		# compute permutation importance only if explicitly requested
		# use tqdm.auto rather than tqdm library because it automatically chooses the best display format (terminal, notebook, etc.)
		# use a tqdm progress bar from the tqdm_joblib library (compatible with permutation_importance using joblib parallelism)
		# use threading backend to avoid DeprecationWarning from fork and ChildProcessError from the loky backend
		# threading is slightly slower but ensures smooth tqdm display and avoids nested multiprocessing issues
		if PERMUTATIONIMPORTANCE is True:
			# determine which features to use for permutation importance (selected features only)
			try:
				if hasattr(best_model, "named_steps") and 'feature_selection' in best_model.named_steps:
					selector = best_model.named_steps['feature_selection']
					if hasattr(selector, "get_support"):
						# retrieve boolean mask of selected features
						support_mask_perm = selector.get_support()
						# apply mask directly on float32 encoded data
						X_train_perm_df = X_train_encoded_float32.loc[:, support_mask_perm]
						X_test_perm_df  = X_test_encoded_float32.loc[:, support_mask_perm]
						feature_encoded_perm_lst = (
							np.array(feature_encoded_lst_full)[support_mask_perm].tolist()
						)
						message_perm_selection = (
							"The permutation importance was restricted to the features selected upstream (i.e., "
							+ str(len(feature_encoded_perm_lst))
							+ ") by the specified feature selection method"
						)
					else:
						# infer selection from transformed dimension
						X_train_transformed = selector.transform(X_train_encoded_float32)
						n_in = X_train_encoded_float32.shape[1]
						n_out = X_train_transformed.shape[1]
						if n_out < n_in:
							support_mask_perm = np.zeros(n_in, dtype=bool)
							support_mask_perm[:n_out] = True
							# use transformed matrices for permutation importance
							X_train_perm_df = pd.DataFrame(
								X_train_transformed,
								columns=np.array(feature_encoded_lst_full)[support_mask_perm].tolist()
							)
							X_test_transformed = selector.transform(X_test_encoded_float32)
							X_test_perm_df = pd.DataFrame(
								X_test_transformed,
								columns=np.array(feature_encoded_lst_full)[support_mask_perm].tolist()
							)
							feature_encoded_perm_lst = (
								np.array(feature_encoded_lst_full)[support_mask_perm].tolist()
							)
							message_perm_selection = (
								"The permutation importance was restricted to the features selected upstream (i.e., "
								+ str(len(feature_encoded_perm_lst))
								+ ") using an inferred selection mask because support_ was not available"
							)
						else:
							# no dimensionality reduction performed
							X_train_perm_df = X_train_encoded_float32.copy()
							X_test_perm_df  = X_test_encoded_float32.copy()
							feature_encoded_perm_lst = feature_encoded_lst_full.copy()
							message_perm_selection = (
								"The permutation importance was computed on all one-hot encoded features (i.e., "
								+ str(len(feature_encoded_perm_lst))
								+ ") because no dimensionality reduction was detected"
							)
				else:
					# no feature selection method was applied
					X_train_perm_df = X_train_encoded_float32.copy()
					X_test_perm_df  = X_test_encoded_float32.copy()
					feature_encoded_perm_lst = feature_encoded_lst_full.copy()
					message_perm_selection = (
						"The permutation importance was computed on all one-hot encoded features (i.e., "
						+ str(len(feature_encoded_perm_lst))
						+ ") because no feature selection method was used"
					)
			except Exception:
				# fallback if anything unexpected happens
				X_train_perm_df = X_train_encoded_float32.copy()
				X_test_perm_df  = X_test_encoded_float32.copy()
				feature_encoded_perm_lst = feature_encoded_lst_full.copy()
				message_perm_selection = (
					"The permutation importance defaulted to all one-hot encoded features (i.e., "
					+ str(len(feature_encoded_perm_lst))
					+ ") because selected features could not be recovered from the model"
				)
			# print a message
			print(message_perm_selection)
			# compute permutation importance
			try:
				# compute total number of permutations to estimate progress: one job per feature
				n_features = X_train_perm_df.shape[1]
				permutation_total = n_features
				# single-thread execution
				if JOBS == 1:
					# when using a single thread, tqdm_joblib does not display intermediate updates
					# in this case, permutation_importance is executed normally without a progress bar
					permutation_train = permutation_importance(
						final_estimator,
						X_train_perm_df,
						y_train_series,
						n_repeats=NREPEATS,
						random_state=42,
						scoring="neg_root_mean_squared_error",
						n_jobs=1
					)
					permutation_test = permutation_importance(
						final_estimator,
						X_test_perm_df,
						y_test_series,
						n_repeats=NREPEATS,
						random_state=42,
						scoring="neg_root_mean_squared_error",
						n_jobs=1
					)
				else:
					# when using multiple threads, tqdm_joblib correctly displays the progress bar
					# permutation importance on training dataset
					with tqa.tqdm(
						total=permutation_total,
						desc="Permutation importance on the training dataset",
						position=0,
						leave=True,
						dynamic_ncols=True,
						mininterval=0.2
					) as progress_bar:
						with jl.parallel_backend("threading", n_jobs=JOBS):
							with tqjl.tqdm_joblib(progress_bar):
								with tpc.threadpool_limits(limits=1):
									with ctl.redirect_stdout(io.StringIO()), ctl.redirect_stderr(io.StringIO()):
										permutation_train = permutation_importance(
											final_estimator,
											X_train_perm_df,
											y_train_series,
											n_repeats=NREPEATS,
											random_state=42,
											scoring="neg_root_mean_squared_error",
											n_jobs=JOBS
										)
					# permutation importance on testing dataset
					with tqa.tqdm(
						total=permutation_total,
						desc="Permutation importance on the testing dataset",
						position=0,
						leave=True,
						dynamic_ncols=True,
						mininterval=0.2
					) as progress_bar:
						with jl.parallel_backend("threading", n_jobs=JOBS):
							with tqjl.tqdm_joblib(progress_bar):
								with tpc.threadpool_limits(limits=1):
									with ctl.redirect_stdout(io.StringIO()), ctl.redirect_stderr(io.StringIO()):
										permutation_test = permutation_importance(
											final_estimator,
											X_test_perm_df,
											y_test_series,
											n_repeats=NREPEATS,
											random_state=42,
											scoring="neg_root_mean_squared_error",
											n_jobs=JOBS
										)
				# extract average permutation importance and its standard deviation (train)
				perm_train_mean = np.round(permutation_train.importances_mean, digits)
				perm_train_std  = np.round(permutation_train.importances_std, digits)
				# extract average permutation importance and its standard deviation (test)
				perm_test_mean = np.round(permutation_test.importances_mean, digits)
				perm_test_std  = np.round(permutation_test.importances_std, digits)
				# handle shape mismatch between names and scores
				min_len = min(len(feature_encoded_perm_lst), len(perm_train_mean))
				feature_encoded_perm_lst = feature_encoded_perm_lst[:min_len]
				perm_train_mean = perm_train_mean[:min_len]
				perm_train_std  = perm_train_std[:min_len]
				perm_test_mean  = perm_test_mean[:min_len]
				perm_test_std   = perm_test_std[:min_len]
				# combine permutation importances from training and testing
				permutation_importance_df = pd.DataFrame({
					"feature": feature_encoded_perm_lst,
					"train_mean": perm_train_mean,
					"train_std":  perm_train_std,
					"test_mean":  perm_test_mean,
					"test_std":   perm_test_std
				}).sort_values(by="train_mean", ascending=False).reset_index(drop=True)
				# message to confirm success
				message_permutation = (
					"The permutation importance was successfully computed on both training and testing datasets"
				)
			except Exception as e:
				# fallback in case of failure: return empty DataFrame and report error
				permutation_importance_df = pd.DataFrame()
				message_permutation = (
					"An error occurred while computing permutation importance: " + str(e)
				)
		else:
			# if not requested, return empty DataFrame
			permutation_importance_df = pd.DataFrame()
			message_permutation = "The permutation importance was not computed"
		# print a message
		print(message_permutation)

		# perform prediction
		## from the training dataset
		y_pred_train = best_model.predict(X_train_encoded_float32)
		## from the testing dataset
		y_pred_test = best_model.predict(X_test_encoded_float32)

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
		rmse_train = np.sqrt(mean_squared_error(y_train_series, y_pred_train))
		rmse_test = np.sqrt(mean_squared_error(y_test_series, y_pred_test))
		### MSE
		mse_train = mean_squared_error(y_train_series, y_pred_train)
		mse_test = mean_squared_error(y_test_series, y_pred_test)
		### SMAPE
		smape_train = smape(y_train_series, y_pred_train, threshold=1e-3)
		smape_test = smape(y_test_series, y_pred_test, threshold=1e-3)
		### MAPE
		mape_train = mape(y_train_series, y_pred_train, threshold=1e-3)
		mape_test = mape(y_test_series, y_pred_test, threshold=1e-3)
		### MAE
		mae_train = mean_absolute_error(y_train_series, y_pred_train)
		mae_test = mean_absolute_error(y_test_series, y_pred_test)	
		### R2
		r2_train = r2_score(y_train_series, y_pred_train)
		r2_test = r2_score(y_test_series, y_pred_test)
		### aR2
		ar2_train = adjusted_r2(y_train_series, y_pred_train, n_features=X_train.shape[1])
		ar2_test = adjusted_r2(y_test_series, y_pred_test, n_features=X_test.shape[1])

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

		# combine expectation and prediction from the training dataset
		## convert numpy.ndarray into a dataframe with explicit column name
		y_pred_train_df = pd.DataFrame(y_pred_train, columns=["prediction"])
		## rebuild a clean dataframe for expectations using preserved sample IDs
		y_train_df = pd.DataFrame({
			"sample": train_sample_ids, # explicit sample identifiers
			"expectation": y_train_series.values # always 1-dimensional
		})
		## concatenate horizontally with index resets
		combined_train_df = pd.concat(
			[y_train_df.reset_index(drop=True), # avoids index misalignment
			y_pred_train_df.reset_index(drop=True)], # avoids index misalignment
			axis=1, 
			join="inner" # safeguards against accidental row misalignment
		)

		# combine expectation and prediction from the testing dataset
		## convert numpy.ndarray into a dataframe with explicit column name
		y_pred_test_df = pd.DataFrame(y_pred_test, columns=["prediction"])
		## rebuild a clean dataframe for expectations using preserved sample IDs
		y_test_df = pd.DataFrame({
			"sample": test_sample_ids, # explicit sample identifiers
			"expectation": y_test_series.values # always 1-dimensional
		})
		## concatenate horizontally with index resets
		combined_test_df = pd.concat(
			[y_test_df.reset_index(drop=True), # avoids index misalignment
			y_pred_test_df.reset_index(drop=True)], # avoids index misalignment
			axis=1, 
			join="inner" # safeguards against accidental row misalignment
		)

		# retrieve only prediction intervals using a custom ResidualQuantileWrapper independantly of mapie 0.9.2 to be able to manage only one sample
		## instantiate the residual quantile wrapper with the best trained model and the desired alpha level
		res_wrapper = ResidualQuantileWrapper(estimator=best_model, alpha=ALPHA)
		## fit the wrapper: trains the underlying model and calculates residual quantile for prediction intervals
		res_wrapper.fit(X_train_encoded_float32, y_train_series)
		## predict on training data, returning both point predictions and prediction intervals
		y_pred_train_res_wrapper, y_intervals_train = res_wrapper.predict(X_train_encoded_float32, return_prediction_interval=True)
		## predict on testing data, returning both point predictions and prediction intervals
		y_pred_test_res_wrapper, y_intervals_test = res_wrapper.predict(X_test_encoded_float32, return_prediction_interval=True)
		## convert the numpy array of prediction intervals on training data into a pandas DataFrame
		## columns are named 'lower' and 'upper' for interval bounds
		y_intervals_train_df = pd.DataFrame(y_intervals_train, columns=["lower", "upper"])
		## convert the numpy array of prediction intervals on testing data into a pandas DataFrame
		y_intervals_test_df = pd.DataFrame(y_intervals_test, columns=["lower", "upper"])
		## print a message
		message_alpha = (
			"The prediction intervals (i.e., "
			+ str(round((1 - ALPHA) * 100, 1))
			+ "%) were calculated using ResidualQuantileWrapper with alpha = "
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

		# round digits of all numeric columns (use .copy() to avoid SettingWithCopyWarning)
		## from the training dataset
		combined_train_df = combined_train_df.copy()
		numeric_cols = combined_train_df.select_dtypes(include='number').columns
		combined_train_df[numeric_cols] = combined_train_df[numeric_cols].astype(float).round(digits)
		## from the testing dataset
		combined_test_df = combined_test_df.copy()
		numeric_cols = combined_test_df.select_dtypes(include='number').columns
		combined_test_df[numeric_cols] = combined_test_df[numeric_cols].astype(float).round(digits)

		# ensure samples are sorted in alphanumerical order
		## from the training dataset
		combined_train_df = (
			combined_train_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
			.reset_index(drop=True)
		)
		## from the testing dataset
		combined_test_df = (
			combined_test_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
			.reset_index(drop=True)
		)

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
			pi.dump(X_train_encoded_float32, file)
		## save the calibration targets
		with open(outpath_calibration_targets, 'wb') as file:
			pi.dump(y_train_series, file)
		## save the model
		with open(outpath_model, 'wb') as file:
			pi.dump(best_model, file)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["###########################\n######### context #########\n###########################\n"])
		print(context, file=log_file)
		log_file.writelines(["###########################\n######## reference ########\n###########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["###########################\n###### repositories  ######\n###########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["###########################\n#### acknowledgements  ####\n###########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["###########################\n######## versions  ########\n###########################\n"])
		log_file.writelines("GenomicBasedRegression: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("scipy: " + str(sp.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")	
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")		
		log_file.writelines("catboost: " + str(cb.__version__) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(im.version("tqdm-joblib")) + "\n")	
		log_file.writelines(["###########################\n######## arguments  #######\n###########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["###########################\n######### checks  #########\n###########################\n"])
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_limit + "\n")
		log_file.writelines(message_phenotype_numeric + "\n")
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
		log_file.writelines(message_assert_encoded_features + "\n")
		log_file.writelines(message_column_order + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_feature_selection + "\n")
		log_file.writelines(message_regressor + "\n")
		log_file.writelines(message_pipeline + "\n")
		log_file.writelines(message_parameters + "\n")
		log_file.writelines(message_metrics_cv + "\n")
		log_file.writelines(message_parallelization + "\n")
		log_file.writelines(message_best_parameters + "\n")
		log_file.writelines(message_best_score + "\n")
		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_importance_encoded_feature_names + "\n")
		log_file.writelines(message_importance_count + "\n")
		log_file.writelines(message_compatibility_permutation_nrepeat + "\n")
		if PERMUTATIONIMPORTANCE is True:
			log_file.writelines(message_perm_selection + "\n")
		log_file.writelines(message_permutation + "\n")
		log_file.writelines(message_alpha + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["###########################\n####### execution  ########\n###########################\n"])
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
		log_file.writelines(["###########################\n###### output  files ######\n###########################\n"])
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
		log_file.writelines(["###########################\n### feature  importance ###\n###########################\n"])
		print(feature_importance_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: NaN placeholder in case no native or detectable feature importance is available. \n")
		log_file.writelines(f"Note: Boosting models, especially histogram-based gradient boosting (HGB), may yield all-zero feature importances when no meaningful split gains are computed—typically due to strong regularization, shallow trees, or low feature variability. \n")
		log_file.writelines(["###########################\n# permutation  importance #\n###########################\n"])
		if PERMUTATIONIMPORTANCE is True:
			print(permutation_importance_df.head(20).to_string(index=False), file=log_file)
			log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
			log_file.writelines(f"Note: Positive permutation importance values indicate features that contribute positively to the model’s performance, while negative values suggest features that degrade performance when included. \n")
		else:
			log_file.writelines(f"Note: Permutation importance was not requested. \n")
		log_file.writelines(["###########################\n### performance metrics ###\n###########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_global_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_global_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"Note: RMSE stands for root mean squared error. \n")
		log_file.writelines(f"Note: MSE stands for mean square error. \n")	
		log_file.writelines(f"Note: MAPE stands for mean absolute percentage error. \n")
		log_file.writelines(f"Note: MAE stands for mean absolute error. \n")
		log_file.writelines(f"Note: R2 stands for R-squared. \n")
		log_file.writelines(["###########################\n#### training  dataset ####\n###########################\n"])
		print(combined_train_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: Lower and upper correspond to the range of the prediction intervals. \n")
		log_file.writelines(["###########################\n##### testing dataset #####\n###########################\n"])
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
		## assert identical encoded features between training and prediction datasets
		training_encoded_features = list(feature_encoder.get_feature_names_out())
		prediction_encoded_features = list(X_mutations_encoded.columns)
		if training_encoded_features != prediction_encoded_features:
			message_assert_encoded_features = "The encoded features between training and prediction datasets do not match"
			raise AssertionError(message_assert_encoded_features)
		else:
			message_assert_encoded_features = "The encoded features between training and prediction datasets were confirmed as identical"
			print(message_assert_encoded_features)

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

		# detect regressor type even if wrapped in a Pipeline
		if hasattr(loaded_model, 'named_steps') and 'model' in loaded_model.named_steps:
			final_estimator = loaded_model.named_steps['model']
		else:
			final_estimator = loaded_model
		detected_model = final_estimator.__class__.__name__

		# print a message about pipeline components
		message_detected_model = (
			"The pipeline components of the provided best model were properly recognized: " + re.sub(r'\s+', ' ', str(loaded_model)).strip())
		print(message_detected_model)

		# determine expected feature order from the trained pipeline
		if hasattr(loaded_model, "feature_names_in_"):
			# the pipeline itself remembers training feature names (since sklearn 1.0+)
			expected_features = list(loaded_model.feature_names_in_)
		elif hasattr(loaded_model, "named_steps") and "feature_selection" in loaded_model.named_steps:
			# use the input feature names stored in the selector (without validation)
			expected_features = getattr(
				loaded_model.named_steps["feature_selection"], "feature_names_in_", X_mutations_encoded.columns
			)
		else:
			# fallback to whatever is available
			expected_features = X_mutations_encoded.columns
		# align prediction matrix to expected feature names and order
		X_mutations_encoded = X_mutations_encoded.reindex(columns=expected_features, fill_value=0)
		message_prediction_alignment = ("The one-hot encoded prediction matrix was reindexed and aligned to match the exact feature names and order expected by the trained pipeline")
		print(message_prediction_alignment)

		# perform prediction
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
		message_alpha = "The prediction intervals (i.e., " + str(((1-ALPHA)*100)) + "%) were calculated using a significance level of alpha = " + str(ALPHA)
		print(message_alpha)
		## concatenate intervals with predictions
		combined_mutations_df = pd.concat(
			[combined_mutations_df.reset_index(drop=True),
			y_intervals_mutations_df.reset_index(drop=True)],
			axis=1, join="inner"
		)

		# round digits of all numeric columns (use .copy() to avoid SettingWithCopyWarning)
		combined_mutations_df = combined_mutations_df.copy()
		numeric_cols = combined_mutations_df.select_dtypes(include='number').columns
		combined_mutations_df[numeric_cols] = combined_mutations_df[numeric_cols].astype(float).round(digits)

		# ensure samples are sorted in alphanumerical order
		combined_mutations_df = (
			combined_mutations_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
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
		outpath_prediction = OUTPUTPATH + '/' + PREFIX + '_prediction' + '.tsv'
		outpath_log = OUTPUTPATH + '/' + PREFIX + '_prediction_log' + '.txt'
		## write output in a tsv file
		combined_mutations_df.to_csv(outpath_prediction, sep="\t", index=False, header=True)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["###########################\n######### context #########\n###########################\n"])
		print(context, file=log_file)
		log_file.writelines(["###########################\n######## reference ########\n###########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["###########################\n###### repositories  ######\n###########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["###########################\n#### acknowledgements  ####\n###########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["###########################\n######## versions  ########\n###########################\n"])
		log_file.writelines("GenomicBasedRegression: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("scipy: " + str(sp.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")	
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")		
		log_file.writelines("catboost: " + str(cb.__version__) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(im.version("tqdm-joblib")) + "\n")
		log_file.writelines(["###########################\n######## arguments  #######\n###########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["###########################\n######### checks  #########\n###########################\n"])
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_input_mutations + "\n")
		log_file.writelines(message_missing_features + "\n")
		log_file.writelines(message_extra_features + "\n")
		log_file.writelines(message_assert_encoded_features + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_detected_model + "\n")
		log_file.writelines(message_prediction_alignment + "\n")
		log_file.writelines(message_alpha + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["###########################\n####### execution  ########\n###########################\n"])
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
		log_file.writelines(["###########################\n###### output  files ######\n###########################\n"])
		log_file.writelines(outpath_prediction + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["###########################\n### prediction  dataset ###\n###########################\n"])
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
