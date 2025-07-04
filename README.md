# Usage
The repository GenomicBasedRegression provides a Python (recommended version 3.12) script called GenomicBasedRegression.py to perform regression-based modeling or prediction from binary (e.g., presence/absence of genes) or categorical (e.g., allele profiles) genomic data.
# Context
The scikit-learn (sklearn)-based Python workflow independently supports both modeling (i.e., training and testing) and prediction (i.e., using a pre-built model), and implements 5 feature selection methods, 17 model regressors, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction interval estimation, execution monitoring via progress bars, and parallel processing.
# Version (release)
1.1.0 (July 2025)
# Dependencies
The Python script GenomicBasedRegression.py was prepared and tested with the Python version 3.12 and Ubuntu 20.04 LTS Focal Fossa.
- catboost==1.2.8
- pandas==2.2.2
- xgboost==2.1.3
- lightgbm==4.6.0
- boruta==0.4.3
- scipy==1.16.0
- scikit-learn==1.5.2
- numpy==1.26.4
- joblib==1.5.1
- tqdm==4.67.1
- tqdm-joblib==0.0.4
# Implemented feature selection methods
- SelectKBest (SKB): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- SelectFromModel with Lasso (laSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with ElasticNet (enSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with Random Forest (rfSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- Boruta  with Random Forest (BO): https://pypi.org/project/Boruta/
# Implemented model regressors
- bayesian bidge (BRI): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
- cat boost (CB): https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
- decision tree (DT): https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
- elasticnet (EN): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
- gradient boosting (GB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- hist gradient boosting (HGB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- huber (HU): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
- k-nearest neighbors (KNN): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
- lassa (LA): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- light gradient goosting machine (LGBM): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
- multi-layer perceptron (MLP): https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
- nu support vector (NSV): https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
- polynomial (PN): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- ridge (RI): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- random forest (RF): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- support vector regressor (SVR): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
- extreme gradient boosting (XGB): https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFRegressor
# Recommended environments
## install Python libraries with pip
```
pip3.12 install catboost==1.2.8
pip3.12 install pandas==2.2.2
pip3.12 install xgboost==2.1.3
pip3.12 install lightgbm==4.6.0
pip3.12 install boruta==0.4.3
pip3.12 install scipy==1.15.3
pip3.12 install scikit-learn==1.5.2
pip3.12 install numpy==1.26.4
pip3.12 install joblib==1.5.1
pip3.12 install tqdm==4.67.1
pip3.12 install tqdm-joblib==0.0.4
```
## or install a Docker image
```
docker pull nicolasradomski/genomicbasedregression:1.1.0
```
## or install a Conda environment
```
conda update --all
conda create --name env_conda_GenomicBasedRegression_1.1.0 python=3.12
conda activate env_conda_GenomicBasedRegression_1.1.0
python --version
conda install -c conda-forge mamba=2.0.5
mamba install -c conda-forge catboost==1.2.8
mamba install -c conda-forge pandas==2.2.2
mamba install -c conda-forge xgboost==2.1.3
mamba install -c conda-forge lightgbm==4.6.0
mamba install -c nicolasradomski boruta==0.4.3
mamba install -c conda-forge scipy==1.16.0
mamba install -c conda-forge scikit-learn==1.5.2
mamba install -c conda-forge numpy==1.26.4
mamba install -c conda-forge joblib==1.5.1
mamba install -c conda-forge tqdm==4.67.1
mamba install -c nicolasradomski tqdm-joblib=0.0.4
conda list -n env_conda_GenomicBasedRegression_1.1.0
conda deactivate # after usage
```
## or install a Conda package
```
conda update --all
conda create -n env_anaconda_GenomicBasedRegression_1.1.0 -c nicolasradomski -c conda-forge -c defaults genomicbasedregression=1.1.0
conda activate env_anaconda_GenomicBasedRegression_1.1.0
conda deactivate # after usage
```
# Helps
## modeling
```
usage: GenomicBasedRegression.py modeling [-h] -m INPUTPATH_MUTATIONS -ph INPUTPATH_PHENOTYPES [-da {random,manual}] [-sp SPLITTING]
                                          [-q QUANTILES] [-l LIMIT] [-fs FEATURESELECTION] [-r REGRESSOR] [-k FOLD] [-pa PARAMETERS] [-j JOBS]
                                          [-pi] [-nr NREPEATS] [-a ALPHA] [-o OUTPUTPATH] [-x PREFIX] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample
                        identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns:
                        profiles of mutations (header: labels of mutations). [MANDATORY]
  -ph INPUTPATH_PHENOTYPES, --phenotypes INPUTPATH_PHENOTYPES
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First
                        column: sample identifiers identical to those in the input file of mutations (header: e.g., sample). Second column:
                        categorical phenotype (header: e.g., phenotype). Third column: 'training' or 'testing' dataset (header: e.g., dataset).
                        [MANDATORY]
  -da {random,manual}, --dataset {random,manual}
                        Perform random (i.e., 'random') or manual (i.e., 'manual') splitting of training and testing datasets through the holdout
                        method. [OPTIONAL, DEFAULT: 'random']
  -sp SPLITTING, --split SPLITTING
                        Percentage of random splitting when preparing the training dataset using the holdout method. [OPTIONAL, DEFAULT: None]
  -q QUANTILES, --quantiles QUANTILES
                        Number of quantiles used to discretize the phenotype values when preparing the training dataset using the holdout method.
                        [OPTIONAL, DEFAULT: None]
  -l LIMIT, --limit LIMIT
                        Recommended minimum number of samples in both the training and testing datasets to reliably estimate performance metrics.
                        [OPTIONAL, DEFAULT: 10]
  -fs FEATURESELECTION, --featureselection FEATURESELECTION
                        Acronym of the feature selection method to use: SelectKBest (SKB), SelectFromModel with Lasso (laSFM), SelectFromModel
                        with ElasticNet (enSFM), SelectFromModel with Random Forest (rfSFM), or Boruta (BO). Listed in order of increasing
                        computational demand, these methods were chosen for their efficiency, interpretability, and suitability for high-
                        dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]
  -r REGRESSOR, --regressor REGRESSOR
                        Acronym of the regressor to use among bayesian bidge (BRI), cat boost (CB), decision tree (DT), elasticnet (EN), gradient
                        boosting (GB), hist gradient boosting (HGB), huber (HU), k-nearest neighbors (KNN), lassa (LA), light gradient goosting
                        machine (LGBM), multi-layer perceptron (MLP), nu support vector (NSV), polynomial (PN), ridge (RI), random forest (RF),
                        support vector regressor (SVR) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]
  -k FOLD, --fold FOLD  Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-
                        validation method. [OPTIONAL, DEFAULT: 5]
  -pa PARAMETERS, --parameters PARAMETERS
                        Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument
                        of the GridSearchCV function. (OPTIONAL)
  -j JOBS, --jobs JOBS  Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function.
                        [OPTIONAL, DEFAULT: -1]
  -pi, --permutationimportance
                        Compute permutation importance, which can be computationally expensive, especially with many features and/or high
                        repetition counts. [OPTIONAL, DEFAULT: False]
  -nr NREPEATS, --nrepeats NREPEATS
                        Number of repetitions per feature for permutation importance (higher = more stable but slower). [OPTIONAL, DEFAULT: 10]
  -a ALPHA, --alpha ALPHA
                        Significance level α (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − α) × 100]%
                        coverage. [OPTIONAL, DEFAULT: 0.05]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. (DEFAULT: 0)
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
## prediction
```
usage: GenomicBasedRegression.py prediction [-h] -m INPUTPATH_MUTATIONS -f INPUTPATH_FEATURES -fe INPUTPATH_FEATURE_ENCODER -cf
                                            INPUTPATH_CALIBRATION_FEATURES -ct INPUTPATH_CALIBRATION_TARGETS -t INPUTPATH_MODEL [-a ALPHA]
                                            [-o OUTPUTPATH] [-x PREFIX] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column:
                        sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns:
                        profiles of mutations (header: labels of mutations). [MANDATORY]
  -f INPUTPATH_FEATURES, --features INPUTPATH_FEATURES
                        Absolute or relative input path of an object (obj) file including features from the training dataset (i.e., mutations).
                        [MANDATORY]
  -fe INPUTPATH_FEATURE_ENCODER, --featureencoder INPUTPATH_FEATURE_ENCODER
                        Absolute or relative input path of an object (obj) file including encoder from the training dataset (i.e., mutations).
                        [MANDATORY]
  -cf INPUTPATH_CALIBRATION_FEATURES, --calibrationfeatures INPUTPATH_CALIBRATION_FEATURES
                        Absolute or relative input path of an object (obj) file including calibration features from the training dataset (i.e.,
                        mutations). [MANDATORY]
  -ct INPUTPATH_CALIBRATION_TARGETS, --calibrationtargets INPUTPATH_CALIBRATION_TARGETS
                        Absolute or relative input path of an object (obj) file including calibration targets from the training dataset (i.e.,
                        mutations). [MANDATORY]
  -t INPUTPATH_MODEL, --model INPUTPATH_MODEL
                        Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]
  -a ALPHA, --alpha ALPHA
                        Significance level α (a float between 0 and 1) used to compute prediction intervals corresponding to a [(1 − α) × 100]%
                        coverage. [OPTIONAL, DEFAULT: 0.05]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Absolute or relative output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output_]
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
### example with SKB as the feature selection method and XGB as the model regressor (tuning_parameters_XGB.txt)
```
{
# --- Feature selection (SelectKBest) ---
# used only to modify selected SKB behavior
'feature_selection__k': [25, 50], # number of top features to keep
'feature_selection__score_func': [mutual_info_regression], # score function
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
- tuning_parameters_BRI.txt
- tuning_parameters_CB.txt
- tuning_parameters_DT.txt
- tuning_parameters_EN.txt
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
## genomic profils for prediction (e.g. genomic_profiles_for_prediction.tsv). "A" and "L" stand for alleles and locus, respectively
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
# Examples of commands (additional examples can be found at the beginning of the script)
## import the GitHub repository
```
git clone --branch v1.1.0 --single-branch https://github.com/Nicolas-Radomski/GenomicBasedRegression.git
cd GenomicBasedRegression
```
## using Python libraries
### without feature selection and with the BRI model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x BRI_FirstAnalysis -da random -sp 80 -q 10 -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/BRI_FirstAnalysis_features.obj -fe MyDirectory/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectory/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectory/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectory/BRI_FirstAnalysis_model.obj -o MyDirectory -x BRI_SecondAnalysis
```
### with the SKB feature selection and the CB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CB_FirstAnalysis -da manual -fs SKB -r CB -k 5 -pa tuning_parameters_CB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/CB_FirstAnalysis_features.obj -fe MyDirectory/CB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/CB_FirstAnalysis_calibration_features.obj -ct MyDirectory/CB_FirstAnalysis_calibration_targets.obj -t MyDirectory/CB_FirstAnalysis_model.obj -o MyDirectory -x CB_SecondAnalysis
```
### with the laSFM feature selection and the DT model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs laSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectory/DT_FirstAnalysis_calibration_features.obj -ct MyDirectory/DT_FirstAnalysis_calibration_targets.obj -t MyDirectory/DT_FirstAnalysis_model.obj -o MyDirectory -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs enSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectory/EN_FirstAnalysis_calibration_features.obj -ct MyDirectory/EN_FirstAnalysis_calibration_targets.obj -t MyDirectory/EN_FirstAnalysis_model.obj -o MyDirectory -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the GB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs rfSFM -r GB -k 5 -pa tuning_parameters_GB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/GB_FirstAnalysis_calibration_features.obj -ct MyDirectory/GB_FirstAnalysis_calibration_targets.obj -t MyDirectory/GB_FirstAnalysis_model.obj -o MyDirectory -x GB_SecondAnalysis
```
### with the BO feature selection and the HGB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs BO -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/HGB_FirstAnalysis_model.obj -o MyDirectory -x HGB_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectory/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectory/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectory/LGBM_FirstAnalysis_model.obj -o MyDirectory -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
python3.12 GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectory/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
python3.12 GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectory/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectory/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectory/XGB_FirstAnalysis_model.obj -o MyDirectory -x XGB_SecondAnalysis
```
## using a Docker image
### without feature selection and with the BRI model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryDockerHub -x BRI_FirstAnalysis -da random -sp 80 -q 10 -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/BRI_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/BRI_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x BRI_SecondAnalysis
```
### with the SKB feature selection and the CB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x CB_FirstAnalysis -da manual -fs SKB -r CB -k 5 -pa tuning_parameters_CB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/CB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/CB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/CB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/CB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/CB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x CB_SecondAnalysis
```
### with the laSFM feature selection and the DT model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x DT_FirstAnalysis -da manual -fs laSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/DT_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/DT_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x EN_FirstAnalysis -da manual -fs enSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/EN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/EN_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the GB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x GB_FirstAnalysis -da manual -fs rfSFM -r GB -k 5 -pa tuning_parameters_GB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/GB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/GB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x GB_SecondAnalysis
```
### with the BO feature selection and the HGB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x HGB_FirstAnalysis -da manual -fs BO -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/HGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/HGB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x HGB_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/LGBM_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/LGBM_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryDockerHub/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedregression:1.1.0 prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryDockerHub/XGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryDockerHub/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryDockerHub/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryDockerHub/XGB_FirstAnalysis_model.obj -o MyDirectoryDockerHub -x XGB_SecondAnalysis
```
## using a Conda environment
### without feature selection and with the BRI model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryConda -x BRI_FirstAnalysis -da random -sp 80 -q 10 -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/BRI_FirstAnalysis_features.obj -fe MyDirectoryConda/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/BRI_FirstAnalysis_model.obj -o MyDirectoryConda -x BRI_SecondAnalysis
```
### with the SKB feature selection and the CB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x CB_FirstAnalysis -da manual -fs SKB -r CB -k 5 -pa tuning_parameters_CB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/CB_FirstAnalysis_features.obj -fe MyDirectoryConda/CB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/CB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/CB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/CB_FirstAnalysis_model.obj -o MyDirectoryConda -x CB_SecondAnalysis
```
### with the laSFM feature selection and the DT model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x DT_FirstAnalysis -da manual -fs laSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/DT_FirstAnalysis_features.obj -fe MyDirectoryConda/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/DT_FirstAnalysis_model.obj -o MyDirectoryConda -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x EN_FirstAnalysis -da manual -fs enSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/EN_FirstAnalysis_features.obj -fe MyDirectoryConda/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/EN_FirstAnalysis_model.obj -o MyDirectoryConda -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the GB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x GB_FirstAnalysis -da manual -fs rfSFM -r GB -k 5 -pa tuning_parameters_GB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/GB_FirstAnalysis_features.obj -fe MyDirectoryConda/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/GB_FirstAnalysis_model.obj -o MyDirectoryConda -x GB_SecondAnalysis
```
### with the BO feature selection and the HGB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x HGB_FirstAnalysis -da manual -fs BO -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/HGB_FirstAnalysis_features.obj -fe MyDirectoryConda/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/HGB_FirstAnalysis_model.obj -o MyDirectoryConda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryConda/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/LGBM_FirstAnalysis_model.obj -o MyDirectoryConda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
python GenomicBasedRegression.py modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryConda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
python GenomicBasedRegression.py prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryConda/XGB_FirstAnalysis_features.obj -fe MyDirectoryConda/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryConda/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryConda/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryConda/XGB_FirstAnalysis_model.obj -o MyDirectoryConda -x XGB_SecondAnalysis
```
## using a Conda package
### without feature selection and with the BRI model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryAnaconda -x BRI_FirstAnalysis -da random -sp 80 -q 10 -r BRI -k 5 -pa tuning_parameters_BRI.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/BRI_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/BRI_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/BRI_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/BRI_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/BRI_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x BRI_SecondAnalysis
```
### with the SKB feature selection and the CB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x CB_FirstAnalysis -da manual -fs SKB -r CB -k 5 -pa tuning_parameters_CB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/CB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/CB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/CB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/CB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/CB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x CB_SecondAnalysis
```
### with the laSFM feature selection and the DT model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x DT_FirstAnalysis -da manual -fs laSFM -r DT -k 5 -pa tuning_parameters_DT.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/DT_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/DT_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/DT_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/DT_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/DT_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x EN_FirstAnalysis -da manual -fs enSFM -r EN -k 5 -pa tuning_parameters_EN.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/EN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/EN_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/EN_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/EN_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/EN_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x EN_SecondAnalysis
```
### with the rfSFM feature selection and the GB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x GB_FirstAnalysis -da manual -fs rfSFM -r GB -k 5 -pa tuning_parameters_GB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/GB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/GB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/GB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/GB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/GB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x GB_SecondAnalysis
```
### with the BO feature selection and the HGB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x HGB_FirstAnalysis -da manual -fs BO -r HGB -k 5 -pa tuning_parameters_HGB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/HGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/HGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/HGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/HGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/HGB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the LGBM model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LGBM_FirstAnalysis -da manual -fs SKB -r LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LGBM_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/LGBM_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/LGBM_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/LGBM_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the XGB model regressor
```
genomicbasedregression modeling -m genomic_profils_for_modeling.tsv -ph MyDirectoryAnaconda/BRI_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x XGB_FirstAnalysis -da manual -fs SKB -r XGB -k 5 -pa tuning_parameters_XGB.txt -pi
genomicbasedregression prediction -m genomic_profils_for_prediction.tsv -f MyDirectoryAnaconda/XGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/XGB_FirstAnalysis_feature_encoder.obj -cf MyDirectoryAnaconda/XGB_FirstAnalysis_calibration_features.obj -ct MyDirectoryAnaconda/XGB_FirstAnalysis_calibration_targets.obj -t MyDirectoryAnaconda/XGB_FirstAnalysis_model.obj -o MyDirectoryAnaconda -x XGB_SecondAnalysis
```
# Examples of expected output (see inclosed directory called 'MyDirectory')
## performance metrics
```
from the training dataset: 
    RMSE      MSE    SMAPE     MAPE      MAE       R2      aR2
2.715707 7.375063 0.118654 0.143888 1.455134 0.976412 0.974829
from the testing dataset: 
    RMSE      MSE    SMAPE     MAPE     MAE       R2      aR2
2.898826 8.403193 0.197436 0.200232 1.85178 0.973256 0.964034
Note: RMSE stands for root mean squared error. 
Note: MSE stands for mean square error. 
Note: MAPE stands for mean absolute percentage error. 
Note: MAE stands for mean absolute error. 
Note: R2 stands for R-squared.
```
## feature importance
```
feature      importance
Locus_5_A16  14422.541016
Locus_1_A12  3198.04541
Locus_8_A3   1286.663574
Locus_2_A13  1000.411133
Locus_3_A9   779.497375
Locus_9_A9   728.699463
Locus_1_A7   637.111145
Locus_1_A6   374.916473
Locus_4_A4   345.687439
Locus_8_A8   318.044495
```
## permutation importance
```
feature      mean       std
Locus_1_A4   13.996449  1.041301
Locus_5_A16  3.77384    0.39245
Locus_2_A13  1.273994   0.314726
Locus_8_A7   1.233288   0.153829
Locus_7_A7   1.215397   0.220058
Locus_2_A5   0.988849   0.205289
Locus_1_A8   0.918872   0.090483
Locus_9_A7   0.853365   0.166026
Locus_6_A5   0.531326   0.070469
Locus_1_A7   0.470229   0.073976
```
## prediction during the modeling subcommand
```
 sample expectation  prediction     lower     upper
S0.1.02          20   24.665600 16.671402 32.915779
S0.1.03          15   16.521149  8.596568 24.840946
S0.1.04          48   47.760227 39.745174 55.989552
S0.1.05          14   15.315463  7.222312 23.466690
S0.1.06          16   16.551388  8.355962 24.600342
S0.1.07          47   43.925987 35.768700 52.013077
S0.1.08          17   16.452509  8.668841 24.913219
S0.1.09          58   58.456882 50.293499 66.537880
S0.1.10          46   46.132385 38.092991 54.337368
```
## prediction during the prediction subcommand
```
 sample  prediction     lower     upper
S2.1.01   26.382029 18.259838 34.504219
S2.1.02   24.793591 16.671402 32.915779
S2.1.03   16.718758  8.596568 24.840946
S2.1.04   19.105890 10.983701 27.228081
S2.1.05   13.710520  5.588330 21.832710
S2.1.06   16.478151  8.355962 24.600342
S2.1.07   19.093390 10.971200 27.215580
S2.1.08   16.791031  8.668841 24.913219
S2.1.09   56.856041 48.733852 64.978233
S2.1.10   46.215179 38.092991 54.337368
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
