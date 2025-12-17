# Usage
The repository GenomicBasedRegression provides a Python (recommended version 3.12) script called GenomicBasedRegression.py to perform regression-based modeling or prediction from binary (e.g., presence/absence of genes) or categorical (e.g., allele profiles) genomic data.
# Context
The scikit-learn (sklearn)-based Python workflow independently supports both modeling (i.e., training and testing) and prediction (i.e., using a pre-built model), and implements 5 feature selection methods, 19 model regressors, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction interval estimation, execution monitoring via progress bars, and parallel processing.
# Version (release)
1.3.0 (December 2025)
# Dependencies
The Python script GenomicBasedRegression.py was prepared and tested with the Python version 3.12 and Ubuntu 20.04 LTS Focal Fossa.
- pandas==2.2.2
- scipy==1.16.0
- scikit-learn==1.5.2
- catboost==1.2.8
- lightgbm==4.6.0
- xgboost==2.1.3
- numpy==1.26.4
- joblib==1.5.1
- tqdm==4.67.1
- tqdm-joblib==0.0.4
# Implemented feature selection methods
- SelectKBest (SKB): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- SelectFromModel with Lasso (laSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with ElasticNet (enSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with ridge (riSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with Random Forest (rfSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
# Implemented model regressors
- adaboost (ADA): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
- bayesian bidge (BRI): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
- catboost (CAT): https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
- decision tree (DT): https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
- elasticnet (EN): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
- extra trees (ET): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
- gradient boosting (GB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- histogram-based gradient boosting (HGB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- huber (HU): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
- k-nearest neighbors (KNN): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
- lassa (LA): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- light gradient goosting machine (LGBM): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
- multi-layer perceptron (MLP): https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
- nu support vector (NSV): https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
- polynomial (PN): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- random forest (RF): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- ridge (RI): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- support vector regressor (SVR): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
- extreme gradient boosting (XGB): https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFRegressor
# Recommended environments
## install Python libraries with pip
```
pip3.12 install pandas==2.2.2
pip3.12 install scipy==1.15.3
pip3.12 install scikit-learn==1.5.2
pip3.12 install catboost==1.2.8
pip3.12 install lightgbm==4.6.0
pip3.12 install xgboost==2.1.3
pip3.12 install numpy==1.26.4
pip3.12 install joblib==1.5.1
pip3.12 install tqdm==4.67.1
pip3.12 install tqdm-joblib==0.0.4
```
## or install a Docker image
```
docker pull nicolasradomski/genomicbasedregression:1.3.0
```
## or install a Conda environment
```
conda update --all
conda --version # conda 25.7.0
conda create --name env_conda_GenomicBasedRegression_1.3.0 python=3.12
conda activate env_conda_GenomicBasedRegression_1.3.0
python --version # Python 3.12.12
conda install -c conda-forge mamba=2.0.5
mamba install -c conda-forge pandas==2.2.2
mamba install -c conda-forge scipy==1.16.0
mamba install -c conda-forge scikit-learn==1.5.2
mamba install -c conda-forge catboost==1.2.8
mamba install -c conda-forge lightgbm==4.6.0
mamba install -c conda-forge xgboost==2.1.3
mamba install -c conda-forge numpy==1.26.4
mamba install -c conda-forge joblib==1.5.1
mamba install -c conda-forge tqdm==4.67.1
mamba install -c nicolasradomski tqdm-joblib=0.0.4
conda list -n env_conda_GenomicBasedRegression_1.3.0
conda deactivate # after usage
```
## or install a Conda package
```
conda update --all
conda create -n env_anaconda_GenomicBasedRegression_1.3.0 -c nicolasradomski -c conda-forge -c defaults genomicbasedregression=1.3.0
conda activate env_anaconda_GenomicBasedRegression_1.3.0
conda deactivate # after usage
```
# Helps
## modeling
```
usage: GenomicBasedRegression.py modeling [-h] -m INPUTPATH_MUTATIONS -ph INPUTPATH_PHENOTYPES [-da {random,manual}] [-sp SPLITTING] 
                                          [-q QUANTILES] [-l LIMIT] [-fs FEATURESELECTION] [-r REGRESSOR] [-k FOLD] [-pa PARAMETERS] 
                                          [-j JOBS] [-pi] [-nr NREPEATS] [-a ALPHA] [-o OUTPUTPATH] [-x PREFIX] [-di DIGITS] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical
                        to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of
                        mutations). [MANDATORY]
  -ph INPUTPATH_PHENOTYPES, --phenotypes INPUTPATH_PHENOTYPES
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample
                        identifiers identical to those in the input file of mutations (header: e.g., sample). Second column: categorical phenotype (header: e.g.,
                        phenotype). Third column: 'training' or 'testing' dataset (header: e.g., dataset). [MANDATORY]
  -da {random,manual}, --dataset {random,manual}
                        Perform random (i.e., 'random') or manual (i.e., 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL,
                        DEFAULT: 'random']
  -sp SPLITTING, --split SPLITTING
                        Percentage of random splitting when preparing the training dataset using the holdout method. [OPTIONAL, DEFAULT: None]
  -q QUANTILES, --quantiles QUANTILES
                        Number of quantiles used to discretize the phenotype values when preparing the training dataset using the holdout method. [OPTIONAL, DEFAULT:
                        None]
  -l LIMIT, --limit LIMIT
                        Recommended minimum number of samples in both the training and testing datasets to reliably estimate performance metrics. [OPTIONAL, DEFAULT:
                        10]
  -fs FEATURESELECTION, --featureselection FEATURESELECTION
                        Acronym of the regression-compatible feature selection method to use: SelectKBest (SKB), SelectFromModel with lasso (laSFM), SelectFromModel
                        with elasticnet (enSFM), or SelectFromModel with ridge (riSFM), or SelectFromModel with random forest (rfSFM). These methods are suitable for
                        high-dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]
  -r REGRESSOR, --regressor REGRESSOR
                        Acronym of the regressor to use among adaboost (ADA), bayesian ridge (BRI), catboost (CAT), decision tree (DT), elasticnet (EN), extra trees
                        (ET), gradient boosting (GB), histogram-based gradient boosting (HGB), huber (HU), k-nearest neighbors (KNN), lassa (LA), light gradient
                        boosting machine (LGBM), multi-layer perceptron (MLP), nu support vector (NSV), polynomial (PN), ridge (RI), random forest (RF), support
                        vector regressor (SVR) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]
  -k FOLD, --fold FOLD  Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-validation method.
                        [OPTIONAL, DEFAULT: 5]
  -pa PARAMETERS, --parameters PARAMETERS
                        Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument of the GridSearchCV
                        function. (OPTIONAL)
  -j JOBS, --jobs JOBS  Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]
  -pi, --permutationimportance
                        Compute permutation importance, which can be computationally expensive, especially with many features and/or high repetition counts.
                        [OPTIONAL, DEFAULT: False]
  -nr NREPEATS, --nrepeats NREPEATS
                        Number of repetitions per feature for permutation importance; higher values provide more stable estimates but increase runtime. [OPTIONAL,
                        DEFAULT: 10]
  -a ALPHA, --alpha ALPHA
                        Significance level alpha (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − alpha) × 100]% coverage.
                        [OPTIONAL, DEFAULT: 0.05]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output]
  -di DIGITS, --digits DIGITS
                        Number of decimal digits to round numerical results (e.g., root mean squared error, importance, metrics). [OPTIONAL, DEFAULT: 6]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
## prediction
```
usage: GenomicBasedRegression.py prediction [-h] -m INPUTPATH_MUTATIONS -f INPUTPATH_FEATURES -fe INPUTPATH_FEATURE_ENCODER 
                                            -cf INPUTPATH_CALIBRATION_FEATURES -ct INPUTPATH_CALIBRATION_TARGETS -t INPUTPATH_MODEL 
                                            [-a ALPHA] [-o OUTPUTPATH] [-x PREFIX] [-di DIGITS] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers
                        identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels
                        of mutations). [MANDATORY]
  -f INPUTPATH_FEATURES, --features INPUTPATH_FEATURES
                        Absolute or relative input path of an object (obj) file including features from the training dataset (i.e., mutations). [MANDATORY]
  -fe INPUTPATH_FEATURE_ENCODER, --featureencoder INPUTPATH_FEATURE_ENCODER
                        Absolute or relative input path of an object (obj) file including encoder from the training dataset (i.e., mutations). [MANDATORY]
  -cf INPUTPATH_CALIBRATION_FEATURES, --calibrationfeatures INPUTPATH_CALIBRATION_FEATURES
                        Absolute or relative input path of an object (obj) file including calibration features from the training dataset (i.e., mutations).
                        [MANDATORY]
  -ct INPUTPATH_CALIBRATION_TARGETS, --calibrationtargets INPUTPATH_CALIBRATION_TARGETS
                        Absolute or relative input path of an object (obj) file including calibration targets from the training dataset (i.e., mutations). [MANDATORY]
  -t INPUTPATH_MODEL, --model INPUTPATH_MODEL
                        Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]
  -a ALPHA, --alpha ALPHA
                        Significance level alpha (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − alpha) × 100]% coverage.
                        [OPTIONAL, DEFAULT: 0.05]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Absolute or relative output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output_]
  -di DIGITS, --digits DIGITS
                        Number of decimal digits to round numerical results (e.g., root mean squared error, importance, metrics). [OPTIONAL, DEFAULT: 6]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
# Expected input files
## phenotypes and datasets for modeling (e.g., phenotype_dataset.tsv)
```
sample    phenotype	dataset
S0.1.01   20	training
S0.1.02   20	training
S0.1.03   15	training
S0.1.04   48	training
S0.1.05   14	testing
S0.1.06   16	training
S0.1.07   47	training
S0.1.08   17	training
S0.1.09   58	training
S0.1.10   46	testing
```
## genomic data for modeling (e.g., genomic_profiles_for_modeling.tsv). "A" and "L" stand for alleles and locus, respectively
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10
S0.1.01 	A3	A2	A3	A4	A5	A6	A7	A3	A4	A10
S0.1.02 	A8	A5	A3	A4	A5	A6	A7	A3	A4	A10
S0.1.03 	A6	A7	A6	A2	A17	A5	A6	A7	A8	A18
S0.1.04 	A12	A44	A8	A5	A16	A4	A5	A6	A12	A17
S0.1.05 	A6	A7	A15	A16	A3	A14	A6	A7	A8	A18
S0.1.06 	A6	A7	A15	A16	A8	A5	A6	A7	A8	A18
S0.1.07 	A7		A9	A10	A11	A14	A3	A2	A10	A16
S0.1.08 	A6	A7	A15	A16	A17	A5	A7	A5	A8	A18
S0.1.09 	A12	A13	A14	A15	A16	A4	A5	A6	A3	A2
S0.1.10 	A12	A13	A14	A15	A16	A4	A5	A6	A8	A8
```
## tuning parameters for modeling
### example with the SKB feature selection method and the XGB model regressor (tuning_parameters_XGB.txt)
```
{
# --- Feature selection (SelectKBest) ---
# used only to modify selected SKB behavior
'feature_selection__k': [25, 50], # number of top features to keep
'feature_selection__score_func': [mutual_info_regression], # score function (avoid chi2 in a context of regression and f_regression it captures only linear dependancies)
# --- Model tuning (XGBRegressor) ---
# used only to modify XGBRegressor behavior
# simplified tuning used together with SKB
'model__max_depth': [3, 4, 5], # shallow to moderate tree depth for speed and flexibility
'model__learning_rate': [0.05, 0.1, 0.2], # slower to moderate learning rates for stable training
'model__subsample': [0.7, 0.8], # moderate data subsampling to reduce overfitting
'model__colsample_bytree': [0.7], # balanced feature subsampling
'model__n_estimators': [50], # fewer boosting rounds for speed
'model__gamma': [0], # no complexity penalty by default
'model__importance_type': ['gain'] # importance extraction
}
```
### other examples depending on the selected feature selection method and model regressor
- tuning_parameters_ADA.txt
- tuning_parameters_BRI.txt
- tuning_parameters_CAT.txt
- tuning_parameters_DT.txt
- tuning_parameters_EN.txt
- tuning_parameters_ET.txt
- tuning_parameters_GB.txt
- tuning_parameters_HGB.txt
- tuning_parameters_HU.txt
- tuning_parameters_KNN.txt
- tuning_parameters_LA.txt
- tuning_parameters_LGBM.txt
- tuning_parameters_MLP.txt
- tuning_parameters_NSV.txt
- tuning_parameters_PN.txt
- tuning_parameters_RF.txt
- tuning_parameters_RI.txt
- tuning_parameters_SVR.txt
- tuning_parameters_XGB.txt
## genomic profils for prediction (e.g., genomic_profiles_for_prediction.tsv). "A" and "L" stand for alleles and locus, respectively
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10	L_11
S2.1.01 	A3	A2	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.02 	A8	A5	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.03 	A6	A7	A6	A2	A17	A5	A6	A7	A8	A1	A18
S2.1.04 	 	A13	A8	A5	A16	A4	A5	A6	A12	A1	A17
S2.1.05 	A6	A24	A15	A16	A3	A14	A6	A7	A8	A1	A18
S2.1.06 	A6	A7	A15	A16	A8	A5	A6	A7	A8	A1	A18
S2.1.07 	A7	A8	A9	A10	A11	A14	A3	A2	A88	A1	A16
S2.1.08 	A6	A7	A15	A16	A17	A5	A7	A5	A8	A1	A18
S2.1.09 	A12	A13	A14	A25	A16	A4	A5		A3	A1	A2
S2.1.10 	A12	A13	A14	A15	A16	A4	A5	A6	A8	A1	A8
```
# Examples of commands
## import the GitHub repository
```
git clone --branch v1.3.0 --single-branch https://github.com/Nicolas-Radomski/GenomicBasedRegression.git
cd GenomicBasedRegression
```
## using Python libraries from pip
### without feature selection and with the ADA model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x ADA_FirstAnalysis -da random -sp 80 -q 10 -r ADA -k 5 -pa tuning_parameters_ADA.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/ADA_FirstAnalysis_features.obj -fe MyDirectory/ADA_FirstAnalysis_feature_encoder.obj -cf MyDirectory/ADA_FirstAnalysis_calibration_features.obj -ct MyDirectory/ADA_FirstAnalysis_calibration_targets.obj -t MyDirectory/ADA_FirstAnalysis_model.obj -o MyDirectory -x ADA_SecondAnalysis
```
### with the SKB feature selection and the BRI model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x BRI_FirstAnalysis -da manual -fs SKB -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/BRI_FirstAnalysis_features.obj -fe MyDirectory/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectory/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectory/BRI_FirstAnalysis_model.obj -o MyDirectory -x BRI_SecondAnalysis
```
### with the laSFM feature selection and the CAT model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CAT_FirstAnalysis -da manual -fs laSFM -r CAT -k 5 -pa tuning_parameters_CAT.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/CAT_FirstAnalysis_features.obj -fe MyDirectory/CAT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/CAT_FirstAnalysis_calibration_features.obj -ct MyDirectory/CAT_FirstAnalysis_calibration_targets.obj -t MyDirectory/CAT_FirstAnalysis_model.obj -o MyDirectory -x CAT_SecondAnalysis
```
### with the enSFM feature selection and the DT model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs enSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/DT_FirstAnalysis_calibration_features.obj -ct MyDirectory/DT_FirstAnalysis_calibration_targets.obj -t MyDirectory/DT_FirstAnalysis_model.obj -o MyDirectory -x DT_SecondAnalysis
```
### with the riSFM feature selection and the EN model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs riSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/EN_FirstAnalysis_calibration_features.obj -ct MyDirectory/EN_FirstAnalysis_calibration_targets.obj -t MyDirectory/EN_FirstAnalysis_model.obj -o MyDirectory -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the ET model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x ET_FirstAnalysis -da manual -fs rfSFM -r ET -k 5 -pa tuning_parameters_ET.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/ET_FirstAnalysis_features.obj -fe MyDirectory/ET_FirstAnalysis_feature_encoder.obj -cf MyDirectory/ET_FirstAnalysis_calibration_features.obj -ct MyDirectory/ET_FirstAnalysis_calibration_targets.obj -t MyDirectory/ET_FirstAnalysis_model.obj -o MyDirectory -x ET_SecondAnalysis
```
### with the SKB feature selection and the GB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs SKB -r GB -k 5 -pa tuning_parameters_GB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/GB_FirstAnalysis_calibration_features.obj -ct MyDirectory/GB_FirstAnalysis_calibration_targets.obj -t MyDirectory/GB_FirstAnalysis_model.obj -o MyDirectory -x GB_SecondAnalysis
```
### with the SKB feature selection and the HGB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs SKB -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/HGB_FirstAnalysis_model.obj -o MyDirectory -x HGB_SecondAnalysis
```
### with the SKB feature selection and the HU model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HU_FirstAnalysis -da manual -fs SKB -r HU -k 5 -pa tuning_parameters_HU.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HU_FirstAnalysis_features.obj -fe MyDirectory/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HU_FirstAnalysis_calibration_features.obj -ct MyDirectory/HU_FirstAnalysis_calibration_targets.obj -t MyDirectory/HU_FirstAnalysis_model.obj -o MyDirectory -x HU_SecondAnalysis
```
### with the SKB feature selection and the KNN model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/KNN_FirstAnalysis_features.obj -fe MyDirectory/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectory/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectory/KNN_FirstAnalysis_model.obj -o MyDirectory -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LA_FirstAnalysis_features.obj -fe MyDirectory/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LA_FirstAnalysis_calibration_features.obj -ct MyDirectory/LA_FirstAnalysis_calibration_targets.obj -t MyDirectory/LA_FirstAnalysis_model.obj -o MyDirectory -x LA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectory/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectory/LGBM_FirstAnalysis_model.obj -o MyDirectory -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the MLP model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/MLP_FirstAnalysis_features.obj -fe MyDirectory/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectory/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectory/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectory/MLP_FirstAnalysis_model.obj -o MyDirectory -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/NSV_FirstAnalysis_features.obj -fe MyDirectory/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectory/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectory/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectory/NSV_FirstAnalysis_model.obj -o MyDirectory -x NSV_SecondAnalysis
```
### with the SKB feature selection and the PN model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x PN_FirstAnalysis -da manual -fs SKB -r PN -k 5 -pa tuning_parameters_PN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/PN_FirstAnalysis_features.obj -fe MyDirectory/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/PN_FirstAnalysis_calibration_features.obj -ct MyDirectory/PN_FirstAnalysis_calibration_targets.obj -t MyDirectory/PN_FirstAnalysis_model.obj -o MyDirectory -x PN_SecondAnalysis
```
### with the SKB feature selection and the RF model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RF_FirstAnalysis_features.obj -fe MyDirectory/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RF_FirstAnalysis_calibration_features.obj -ct MyDirectory/RF_FirstAnalysis_calibration_targets.obj -t MyDirectory/RF_FirstAnalysis_model.obj -o MyDirectory -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/RI_FirstAnalysis_features.obj -fe MyDirectory/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/RI_FirstAnalysis_calibration_features.obj -ct MyDirectory/RI_FirstAnalysis_calibration_targets.obj -t MyDirectory/RI_FirstAnalysis_model.obj -o MyDirectory -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVR model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/SVR_FirstAnalysis_features.obj -fe MyDirectory/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectory/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectory/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectory/SVR_FirstAnalysis_model.obj -o MyDirectory -x SVR_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/XGB_FirstAnalysis_model.obj -o MyDirectory -x XGB_SecondAnalysis
```
## using a Docker image
### without feature selection and with the ADA model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryDockerHub -x ADA_FirstAnalysis -da random -sp 80 -q 10 -r ADA -k 5 -pa tuning_parameters_ADA.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/ADA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/ADA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/ADA_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/ADA_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/ADA_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x ADA_SecondAnalysis
```
### with the SKB feature selection and the BRI model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x BRI_FirstAnalysis -da manual -fs SKB -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/BRI_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/BRI_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x BRI_SecondAnalysis
```
### with the laSFM feature selection and the CAT model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x CAT_FirstAnalysis -da manual -fs laSFM -r CAT -k 5 -pa tuning_parameters_CAT.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/CAT_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/CAT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/CAT_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/CAT_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/CAT_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x CAT_SecondAnalysis
```
### with the enSFM feature selection and the DT model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x DT_FirstAnalysis -da manual -fs enSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/DT_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/DT_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x DT_SecondAnalysis
```
### with the riSFM feature selection and the EN model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x EN_FirstAnalysis -da manual -fs riSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/EN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/EN_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the ET model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x ET_FirstAnalysis -da manual -fs rfSFM -r ET -k 5 -pa tuning_parameters_ET.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/ET_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/ET_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/ET_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/ET_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/ET_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x ET_SecondAnalysis
```
### with the SKB feature selection and the GB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x GB_FirstAnalysis -da manual -fs SKB -r GB -k 5 -pa tuning_parameters_GB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/GB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/GB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x GB_SecondAnalysis
```
### with the SKB feature selection and the HGB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x HGB_FirstAnalysis -da manual -fs SKB -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/HGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/HGB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x HGB_SecondAnalysis
```
### with the SKB feature selection and the HU model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x HU_FirstAnalysis -da manual -fs SKB -r HU -k 5 -pa tuning_parameters_HU.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/HU_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/HU_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/HU_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/HU_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x HU_SecondAnalysis
```
### with the SKB feature selection and the KNN model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/KNN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/KNN_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/LA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/LA_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/LA_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/LA_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x LA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/LGBM_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/LGBM_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the MLP model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/MLP_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/MLP_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/NSV_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/NSV_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x NSV_SecondAnalysis
```
### with the SKB feature selection and the PN model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x PN_FirstAnalysis -da manual -fs SKB -r PN -k 5 -pa tuning_parameters_PN.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/PN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/PN_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/PN_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/PN_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x PN_SecondAnalysis
```
### with the SKB feature selection and the RF model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/RF_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/RF_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/RF_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/RF_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/RI_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/RI_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/RI_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/RI_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVR model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/SVR_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/SVR_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x SVR_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.3.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/XGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/XGB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x XGB_SecondAnalysis
```
## using a Conda environment
### without feature selection and with the ADA model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryConda -x ADA_FirstAnalysis -da random -sp 80 -q 10 -r ADA -k 5 -pa tuning_parameters_ADA.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/ADA_FirstAnalysis_features.obj -fe MyDirectoryConda/ADA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/ADA_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/ADA_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/ADA_FirstAnalysis_model.obj -o MyDirectoryConda -x ADA_SecondAnalysis
```
### with the SKB feature selection and the BRI model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x BRI_FirstAnalysis -da manual -fs SKB -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/BRI_FirstAnalysis_features.obj -fe MyDirectoryConda/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/BRI_FirstAnalysis_model.obj -o MyDirectoryConda -x BRI_SecondAnalysis
```
### with the laSFM feature selection and the CAT model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x CAT_FirstAnalysis -da manual -fs laSFM -r CAT -k 5 -pa tuning_parameters_CAT.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/CAT_FirstAnalysis_features.obj -fe MyDirectoryConda/CAT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/CAT_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/CAT_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/CAT_FirstAnalysis_model.obj -o MyDirectoryConda -x CAT_SecondAnalysis
```
### with the enSFM feature selection and the DT model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x DT_FirstAnalysis -da manual -fs enSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/DT_FirstAnalysis_features.obj -fe MyDirectoryConda/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/DT_FirstAnalysis_model.obj -o MyDirectoryConda -x DT_SecondAnalysis
```
### with the riSFM feature selection and the EN model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x EN_FirstAnalysis -da manual -fs riSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/EN_FirstAnalysis_features.obj -fe MyDirectoryConda/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/EN_FirstAnalysis_model.obj -o MyDirectoryConda -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the ET model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x ET_FirstAnalysis -da manual -fs rfSFM -r ET -k 5 -pa tuning_parameters_ET.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/ET_FirstAnalysis_features.obj -fe MyDirectoryConda/ET_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/ET_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/ET_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/ET_FirstAnalysis_model.obj -o MyDirectoryConda -x ET_SecondAnalysis
```
### with the SKB feature selection and the GB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x GB_FirstAnalysis -da manual -fs SKB -r GB -k 5 -pa tuning_parameters_GB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/GB_FirstAnalysis_features.obj -fe MyDirectoryConda/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/GB_FirstAnalysis_model.obj -o MyDirectoryConda -x GB_SecondAnalysis
```
### with the SKB feature selection and the HGB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x HGB_FirstAnalysis -da manual -fs SKB -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/HGB_FirstAnalysis_features.obj -fe MyDirectoryConda/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/HGB_FirstAnalysis_model.obj -o MyDirectoryConda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the HU model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x HU_FirstAnalysis -da manual -fs SKB -r HU -k 5 -pa tuning_parameters_HU.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/HU_FirstAnalysis_features.obj -fe MyDirectoryConda/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/HU_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/HU_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/HU_FirstAnalysis_model.obj -o MyDirectoryConda -x HU_SecondAnalysis
```
### with the SKB feature selection and the KNN model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/KNN_FirstAnalysis_features.obj -fe MyDirectoryConda/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/KNN_FirstAnalysis_model.obj -o MyDirectoryConda -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/LA_FirstAnalysis_features.obj -fe MyDirectoryConda/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/LA_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/LA_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/LA_FirstAnalysis_model.obj -o MyDirectoryConda -x LA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryConda/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/LGBM_FirstAnalysis_model.obj -o MyDirectoryConda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the MLP model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/MLP_FirstAnalysis_features.obj -fe MyDirectoryConda/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/MLP_FirstAnalysis_model.obj -o MyDirectoryConda -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/NSV_FirstAnalysis_features.obj -fe MyDirectoryConda/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/NSV_FirstAnalysis_model.obj -o MyDirectoryConda -x NSV_SecondAnalysis
```
### with the SKB feature selection and the PN model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x PN_FirstAnalysis -da manual -fs SKB -r PN -k 5 -pa tuning_parameters_PN.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/PN_FirstAnalysis_features.obj -fe MyDirectoryConda/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/PN_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/PN_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/PN_FirstAnalysis_model.obj -o MyDirectoryConda -x PN_SecondAnalysis
```
### with the SKB feature selection and the RF model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/RF_FirstAnalysis_features.obj -fe MyDirectoryConda/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/RF_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/RF_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/RF_FirstAnalysis_model.obj -o MyDirectoryConda -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/RI_FirstAnalysis_features.obj -fe MyDirectoryConda/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/RI_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/RI_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/RI_FirstAnalysis_model.obj -o MyDirectoryConda -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVR model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/SVR_FirstAnalysis_features.obj -fe MyDirectoryConda/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/SVR_FirstAnalysis_model.obj -o MyDirectoryConda -x SVR_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/XGB_FirstAnalysis_features.obj -fe MyDirectoryConda/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/XGB_FirstAnalysis_model.obj -o MyDirectoryConda -x XGB_SecondAnalysis
```
## using a Conda package
### without feature selection and with the ADA model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryAnaconda -x ADA_FirstAnalysis -da random -sp 80 -q 10 -r ADA -k 5 -pa tuning_parameters_ADA.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/ADA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/ADA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/ADA_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/ADA_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/ADA_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x ADA_SecondAnalysis
```
### with the SKB feature selection and the BRI model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x BRI_FirstAnalysis -da manual -fs SKB -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/BRI_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/BRI_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x BRI_SecondAnalysis
```
### with the laSFM feature selection and the CAT model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x CAT_FirstAnalysis -da manual -fs laSFM -r CAT -k 5 -pa tuning_parameters_CAT.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/CAT_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/CAT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/CAT_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/CAT_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/CAT_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x CAT_SecondAnalysis
```
### with the enSFM feature selection and the DT model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x DT_FirstAnalysis -da manual -fs enSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/DT_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/DT_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x DT_SecondAnalysis
```
### with the riSFM feature selection and the EN model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x EN_FirstAnalysis -da manual -fs riSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/EN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/EN_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the ET model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x ET_FirstAnalysis -da manual -fs rfSFM -r ET -k 5 -pa tuning_parameters_ET.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/ET_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/ET_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/ET_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/ET_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/ET_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x ET_SecondAnalysis
```
### with the SKB feature selection and the GB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x GB_FirstAnalysis -da manual -fs SKB -r GB -k 5 -pa tuning_parameters_GB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/GB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/GB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x GB_SecondAnalysis
```
### with the SKB feature selection and the HGB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x HGB_FirstAnalysis -da manual -fs SKB -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/HGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/HGB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the HU model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x HU_FirstAnalysis -da manual -fs SKB -r HU -k 5 -pa tuning_parameters_HU.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/HU_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/HU_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/HU_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/HU_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/HU_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x HU_SecondAnalysis
```
### with the SKB feature selection and the KNN model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x KNN_FirstAnalysis -da manual -fs SKB -r KNN -k 5 -pa tuning_parameters_KNN.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/KNN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/KNN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/KNN_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/KNN_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/KNN_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LA_FirstAnalysis -da manual -fs SKB -r LA -k 5 -pa tuning_parameters_LA.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/LA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LA_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/LA_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/LA_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/LA_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x LA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/LGBM_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the MLP model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x MLP_FirstAnalysis -da manual -fs SKB -r MLP -k 5 -pa tuning_parameters_MLP.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/MLP_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/MLP_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/MLP_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/MLP_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/MLP_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x NSV_FirstAnalysis -da manual -fs SKB -r NSV -k 5 -pa tuning_parameters_NSV.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/NSV_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/NSV_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/NSV_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/NSV_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/NSV_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x NSV_SecondAnalysis
```
### with the SKB feature selection and the PN model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x PN_FirstAnalysis -da manual -fs SKB -r PN -k 5 -pa tuning_parameters_PN.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/PN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/PN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/PN_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/PN_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/PN_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x PN_SecondAnalysis
```
### with the SKB feature selection and the RF model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x RF_FirstAnalysis -da manual -fs SKB -r RF -k 5 -pa tuning_parameters_RF.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/RF_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/RF_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/RF_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/RF_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/RF_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x RI_FirstAnalysis -da manual -fs SKB -r RI -k 5 -pa tuning_parameters_RI.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/RI_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/RI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/RI_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/RI_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/RI_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVR model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x SVR_FirstAnalysis -da manual -fs SKB -r SVR -k 5 -pa tuning_parameters_SVR.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/SVR_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/SVR_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/SVR_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/SVR_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/SVR_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x SVR_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/XGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/XGB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x XGB_SecondAnalysis
```
# Examples of expected output (see inclosed directory called 'MyDirectory')
## feature importance
```
     feature  importance
 Locus_5_A16 4956.503906
 Locus_2_A13 3447.952393
 Locus_1_A12 2031.842773
  Locus_9_A4  799.916992
  Locus_1_A7  228.568069
  Locus_6_A9  203.670471
  Locus_1_A4  194.304108
  Locus_9_A9  187.599777
Locus_10_A18  126.711555
  Locus_4_A4   83.494476
  Locus_1_A6   76.759834
  Locus_6_A6   58.775444
  Locus_6_A5   51.835064
  Locus_7_A5   41.836422
  Locus_7_A7   36.828129
  Locus_3_A8   29.107296
  Locus_3_A3   20.935595
  Locus_9_A8   15.511682
 Locus_8_A13   15.191111
 Locus_4_A15   12.305605
```
## permutation importance
```
     feature  train_mean  train_std  test_mean  test_std
  Locus_3_A8    8.481091   0.949153   8.722864  0.918224
 Locus_5_A16    4.354647   0.506638   5.663910  1.070395
  Locus_6_A5    1.940999   0.321063   2.267965  0.272242
 Locus_4_A16    1.442172   0.408884   1.538514  0.555739
  Locus_1_A7    0.443297   0.072246   0.933021  0.134982
  Locus_4_A4    0.057576   0.014683   0.106935  0.065868
  Locus_7_A5    0.000076   0.000766  -0.000918  0.000655
  Locus_9_A8    0.000000   0.000000   0.000000  0.000000
  Locus_9_A4    0.000000   0.000000   0.000000  0.000000
  Locus_8_A3    0.000000   0.000000   0.000000  0.000000
 Locus_8_A13    0.000000   0.000000   0.000000  0.000000
  Locus_7_A7    0.000000   0.000000   0.000000  0.000000
  Locus_6_A9    0.000000   0.000000   0.000000  0.000000
  Locus_6_A6    0.000000   0.000000   0.000000  0.000000
Locus_10_A18    0.000000   0.000000   0.000000  0.000000
  Locus_6_A4    0.000000   0.000000   0.000000  0.000000
 Locus_1_A12    0.000000   0.000000   0.000000  0.000000
 Locus_4_A15    0.000000   0.000000   0.000000  0.000000
  Locus_3_A6    0.000000   0.000000   0.000000  0.000000
  Locus_3_A3    0.000000   0.000000   0.000000  0.000000
```
## performance metrics
```
from the training dataset: 
    RMSE      MSE    SMAPE     MAPE      MAE       R2     aR2
2.918598 8.518213 0.148581 0.167555 1.697231 0.972533 0.97069
from the testing dataset: 
    RMSE      MSE    SMAPE     MAPE      MAE       R2      aR2
1.670332 2.790008 0.098399 0.108888 1.199855 0.991387 0.988417
```
## prediction during the modeling subcommand
```
 sample  expectation  prediction     lower     upper
S0.1.01         20.0   25.908098 16.967949 32.375763
S0.1.03         15.0   16.706976  9.052378 24.460192
S0.1.04         48.0   48.135574 40.286213 55.694027
S0.1.06         16.0   15.921209  8.255709 23.663523
S0.1.10         46.0   46.134380 38.543644 53.951458
S0.1.11         15.0   15.921209  8.255709 23.663523
S0.1.13         13.0   14.855770  7.429060 22.836874
S0.1.14          5.0    4.810639 -2.872292 12.535522
S0.1.15          6.0    6.737475 -1.556509 13.851305
S0.1.16          6.0    6.737475 -0.338474 15.069340
S0.1.18          2.0    3.404263 -4.309009 11.098805
S0.1.19          2.0    2.790780 -4.984485 10.423329
S0.1.20          4.0    3.404263 -4.309009 11.098805
S1.1.01         24.0   25.908098 16.967949 32.375763
S1.1.02         26.0   25.908098 19.213318 34.621132
S1.1.03         16.0   16.706976  9.052378 24.460192
S1.1.04         48.0   48.135574 40.286213 55.694027
S1.1.06         16.0   15.921209  8.255709 23.663523
S1.1.07         43.0   45.408558 37.756302 53.164116
S1.1.08         17.0   17.444435  9.634922 25.042736
```
## prediction during the prediction subcommand
```
 sample  prediction     lower     upper
S2.1.01   34.574070 26.870163 42.277977
S2.1.02   34.793640 27.089733 42.497547
S2.1.03   30.931238 23.227331 38.635147
S2.1.04   42.754295 35.050388 50.458202
S2.1.05   36.664867 28.960960 44.368774
S2.1.06   34.712154 27.008247 42.416061
S2.1.07   45.460209 37.756302 53.164116
S2.1.08   32.273796 24.569889 39.977703
S2.1.09   51.157234 43.453327 58.861141
S2.1.10   46.247551 38.543644 53.951458
S2.1.11   34.712154 27.008247 42.416061
S2.1.12   53.703907 46.000000 61.407814
S2.1.13   30.889977 23.186069 38.593884
S2.1.14   29.686930 21.983023 37.390839
S2.1.15    3.394899 -4.309009 11.098805
S2.1.16   11.982977  4.279070 19.686884
S2.1.17   30.852961 23.149054 38.556870
S2.1.18    6.469562 -1.234345 14.173469
S2.1.19    2.719422 -4.984485 10.423329
S2.1.20    3.394899 -4.309009 11.098805
```
# Illustration
![workflow figure](https://github.com/Nicolas-Radomski/GenomicBasedRegression/blob/main/illustration.png)
# Funding
Ricerca Corrente - IZS AM 06/24 RC: "genomic data-based machine learning to predict categorical and continuous phenotypes by classification and regression".
# Acknowledgements
Many thanks to Andrea De Ruvo, Adriano Di Pasquale and ChatGPT for the insightful discussions that helped improve the algorithm.
# Reference
An article might potentially be published in the future.
# Repositories
Please cite:
 GitHub (https://github.com/Nicolas-Radomski/GenomicBasedRegression),
 Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedregression),
 and/or Anaconda Hub (https://anaconda.org/nicolasradomski/genomicbasedregression).
# Author
Nicolas Radomski
