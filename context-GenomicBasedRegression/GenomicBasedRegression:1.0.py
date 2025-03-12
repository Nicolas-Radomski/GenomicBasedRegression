# pip3.12 install --force-reinstall pandas==2.2.2
# pip3.12 install -U scikit-learn==1.5.2

# USAGE: python3.12 GenomicBasedRegression:1.0.py modeling -m genomic-profils-for-modeling.tsv -p phenotypes.tsv -o MyDirectory -x FirstAnalysis -s 80 -d 4
# USAGE: python3.12 GenomicBasedRegression:1.0.py prediction -m genomic-profils-for-prediction.tsv -t MyDirectory/FirstAnalysis_model.obj -f MyDirectory/FirstAnalysis_features.obj -ef MyDirectory/FirstAnalysis_encoded_features.obj -o MyDirectory -x SecondAnalysis -d 4

# import packages
import sys as sys # no individual installation because is part of the Python Standard Library
import os as os # no individual installation because is part of the Python Standard Library
import datetime as dt # no individual installation because is part of the Python Standard Library
import argparse as ap # no individual installation because is part of the Python Standard Library
import pandas as pd
import pickle as pi # no individual installation because is part of the Python Standard Library
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# step control
step1_start = dt.datetime.now()

# set workflow reference
reference = 'Please, site GitHub (https://github.com/Nicolas-Radomski/GenomicBasedRegression) and/or Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedregression)'

# create the main parser
parser = ap.ArgumentParser(
	prog='GenomicBasedLinearRegression:1.0.py', 
	description='Perform linear regression-based modeling or prediction from categorical genomic data.',
	epilog=reference
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
	help='Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). (REQUIRED)'
	)
parser_modeling.add_argument(
	'-p', '--phenotypes', 
	dest='inputpath_phenotypes', 
	action='store', 
	required=True, 
	help='Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes. First column: sample identifiers identical to those in the input file of mutations (header: e.g. sample). Second column: continuous phenotype (header: e.g. phenotype). (REQUIRED)'
	)
parser_modeling.add_argument(
	'-s', '--split', 
	dest='splitting', 
	type=int,
	action='store', 
	required=False, 
	default=80, 
	help='Percentage of random splitting to prepare the training dataset through the holdout method. (DEFAULT: 80)'
	)
parser_modeling.add_argument(
	'-o', '--output', 
	dest='outputpath', 
	action='store', 
	required=False, 
	default='.',
	help='Output path. (DEFAULT: .)'
	)
parser_modeling.add_argument(
	'-x', '--prefix', 
	dest='prefix', 
	action='store', 
	required=False, 
	default='output',
	help='Prefix of output files. (DEFAULT: output)'
	)
parser_modeling.add_argument(
	'-d', '--debug', 
	dest='debug', 
	type=int,
	action='store', 
	required=False, 
	default=0, 
	help='Traceback level when an error occurs. (DEFAULT: 0)'
	)
parser_modeling.add_argument(
	'-nc', '--no-check', 
	dest='nocheck', 
	action='store_true', 
	required=False, 
	default=False, 
	help='Do not check versions of Python and packages. (DEFAULT: False)'
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
	help='Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). (REQUIRED)'
	)
parser_prediction.add_argument(
	'-t', '--model', 
	dest='inputpath_model', 
	action='store', 
	required=True, 
	help='Absolute or relative input path of an object (obj) file including a trained scikit-learn model. (REQUIRED)'
	)
parser_prediction.add_argument(
	'-f', '--features', 
	dest='inputpath_features', 
	action='store', 
	required=True, 
	help='Absolute or relative input path of an object (obj) file including trained scikit-learn features (i.e. mutations). (REQUIRED)'
	)
parser_prediction.add_argument(
	'-ef', '--encodedfeatures', 
	dest='inputpath_encoded_features', 
	action='store', 
	required=True, 
	help='Absolute or relative input path of an object (obj) file including trained scikit-learn encoded features (i.e. mutations). (REQUIRED)'
	)
parser_prediction.add_argument(
	'-o', '--output', 
	dest='outputpath', 
	action='store', 
	required=False, 
	default='.',
	help='Absolute or relative output path. (DEFAULT: .)'
	)
parser_prediction.add_argument(
	'-x', '--prefix', 
	dest='prefix', 
	action='store', 
	required=False, 
	default='output',
	help='Prefix of output files. (DEFAULT: output_)'
	)
parser_prediction.add_argument(
	'-d', '--debug', 
	dest='debug', 
	type=int,
	action='store', 
	required=False, 
	default=0, 
	help='Traceback level when an error occurs. (DEFAULT: 0)'
	)
parser_prediction.add_argument(
	'-nc', '--no-check', 
	dest='nocheck', 
	action='store_true', 
	required=False, 
	default=False, 
	help='Do not check versions of Python and packages. (DEFAULT: False)'
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
	OUTPUTPATH=args.outputpath
	SPLITTING=args.splitting
	PREFIX=args.prefix
	DEBUG=args.debug
	NOCHECK=args.nocheck
elif args.subcommand == 'prediction':
	INPUTPATH_MUTATIONS=args.inputpath_mutations
	INPUTPATH_FEATURES=args.inputpath_features
	INPUTPATH_ENCODED_FEATURES=args.inputpath_encoded_features
	INPUTPATH_MODEL=args.inputpath_model
	OUTPUTPATH=args.outputpath
	PREFIX=args.prefix
	DEBUG=args.debug
	NOCHECK=args.nocheck

# set tracebacklimit
sys.tracebacklimit = DEBUG

# control versions
if NOCHECK == False :
	## control Python version
	if sys.version_info[0] != 3 or sys.version_info[1] != 12 :
		raise Exception("Python 3.12 version is recommended")
		exit()
	# control versions of packages
	if ap.__version__ != "1.1":
		raise Exception('argparse 1.1 (1.4.1) version is recommended')
		exit()
	if pd.__version__ != "2.2.2":
		raise Exception('pandas 2.2.2 version is recommended')
		exit()
	if pi.format_version != "4.0":
		raise Exception('pickle 4.0 version is recommended')
		exit()
	if sk.__version__ != "1.5.2":
		raise Exception('sklearn 1.5.2 version is recommended')
		exit()
	message_versions = 'The recommended versions of Python and packages were properly controlled'
else:
	message_versions = 'The recommended versions of Python and packages were not controlled'

# print a message about version control
print(message_versions)

# check the subcommand and execute corresponding code
if args.subcommand == 'modeling':

	# read input files
	## mutations
	df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t')
	## phenotypes
	df_phenotypes = pd.read_csv(INPUTPATH_PHENOTYPES, sep='\t')

	# replace missing genomic data by a string
	df_mutations = df_mutations.fillna('missing')

	# rename labels of headers
	## mutations
	df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
	## phenotypes
	df_phenotypes.rename(columns={df_phenotypes.columns[0]: 'sample'}, inplace=True)
	df_phenotypes.rename(columns={df_phenotypes.columns[1]: 'phenotype'}, inplace=True)

	# sort by samples
	## mutations
	df_mutations = df_mutations.sort_values(by='sample')
	## phenotypes
	df_phenotypes = df_phenotypes.sort_values(by='sample')

	# check if lists of samples are identical
	## convert DataFrame column as a list
	lst_mutations = df_mutations['sample'].tolist()
	lst_phenotypes = df_phenotypes['sample'].tolist()
	## compare lists
	if lst_mutations == lst_phenotypes: 
		message_sample_identifiers = "The samples identifiers are identical"
		print (message_sample_identifiers)
	else: 
		message_sample_identifiers = "The samples identifiers are not identical"
		raise Exception(message_sample_identifiers)
		exit()

	# separate mutations (X) and phenotypes (y) indexing the sample columns
	X = df_mutations.set_index('sample')
	y = df_phenotypes.set_index('sample')

	# split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = SPLITTING/100) # random_state=42 for reproducible

	# encode categorical data into binary data for the training dataset
	## instantiate encoder object
	encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False).set_output(transform='pandas')
	## transform data with the encoder into an array for the training dataset
	X_train_encoded = encoder.fit_transform(X_train[X_train.columns])
	## save the features
	features = X_train.columns
	## save the encoded features
	encoded_features = encoder.categories_

	# encode categorical data into binary data for the testing dataset
	## instantiate encoder object
	encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, categories = encoded_features).set_output(transform='pandas')
	## transform data with the encoder into an array for the testing dataset using produced features
	X_test_encoded = encoder.fit_transform(X_test[features])

	# creat the model
	## select a model
	model = LinearRegression()
	## build the model
	model.fit(X_train_encoded, y_train)

	# evaluate model
	## from the training dataset
	y_pred_train = model.predict(X_train_encoded)
	mse_train = mean_squared_error(y_train, y_pred_train)
	## from the testing dataset
	y_pred_test = model.predict(X_test_encoded)
	mse_test = mean_squared_error(y_test, y_pred_test)

	# combine expectation and prediction from the training
	## transform numpy.ndarray into pandas.core.frame.DataFrame
	y_pred_train_df = pd.DataFrame(y_pred_train)
	## retrieve the sample index in a column
	y_train_df = y_train.reset_index().rename(columns={"index":"sample"})
	## concatenate with reset index
	combined_train_df = pd.concat([y_train_df.reset_index(drop=True), y_pred_train_df.reset_index(drop=True)], axis=1)
	## rename labels of headers
	combined_train_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
	combined_train_df.rename(columns={0: 'prediction'}, inplace=True)

	# combine expectation and prediction from the testing
	## transform numpy.ndarray into pandas.core.frame.DataFrame
	y_pred_test_df = pd.DataFrame(y_pred_test)
	## retrieve the sample index in a column
	y_test_df = y_test.reset_index().rename(columns={"index":"sample"})
	## concatenate with reset index
	combined_test_df = pd.concat([y_test_df.reset_index(drop=True), y_pred_test_df.reset_index(drop=True)], axis=1)
	## rename labels of headers
	combined_test_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
	combined_test_df.rename(columns={0: 'prediction'}, inplace=True)

	# check if the output directory does not exists and make it
	if not os.path.exists(OUTPUTPATH):
		os.makedirs(OUTPUTPATH)
		message_output_directory = "The output directory was created successfully"
		print(message_output_directory)
	else:
		message_output_directory = "The output directory already exists"
		print(message_output_directory)

	# output results
	## output path
	outpath_train = OUTPUTPATH + '/' + PREFIX + '_training_prediction' + '.tsv'
	outpath_test = OUTPUTPATH + '/' + PREFIX + '_testing_prediction' + '.tsv'
	outpath_features = OUTPUTPATH + '/' + PREFIX + '_features' + '.obj'
	outpath_encoded_features = OUTPUTPATH + '/' + PREFIX + '_encoded_features' + '.obj'
	outpath_model = OUTPUTPATH + '/' + PREFIX + '_model' + '.obj'
	outpath_log = OUTPUTPATH + '/' + PREFIX + '_training_log' + '.txt'
	## write output in a tsv file
	combined_train_df.to_csv(outpath_train, sep="\t", index=False, header=True)
	combined_test_df.to_csv(outpath_test, sep="\t", index=False, header=True)
	## save the features
	pi.dump(features, open(outpath_features, 'wb'))
	## save the encoded_features
	pi.dump(encoded_features, open(outpath_encoded_features, 'wb'))
	## save the model
	pi.dump(model, open(outpath_model, 'wb'))
	## write output in a txt file
	log_file = open(outpath_log, "w")
	log_file.writelines(["########################\n###### reference  ######\n########################\n"])
	print(parser.epilog, file=log_file)
	log_file.writelines(["########################\n####### versions #######\n########################\n"])
	log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
	log_file.writelines("argparse: " + str(ap.__version__) + "\n")
	log_file.writelines("pandas: " + str(pd.__version__) + "\n")
	log_file.writelines("pickle: " + str(pi.format_version) + "\n")
	log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
	log_file.writelines(["########################\n####### settings #######\n########################\n"])
	settings_str = str(args)
	settings_str = settings_str[:-1]
	settings_str = settings_str.replace(settings_str[:10], '')
	settings_str = settings_str.replace(", ", "\n")
	print(settings_str, file=log_file)
	log_file.writelines(["########################\n######## checks ########\n########################\n"])
	log_file.writelines(message_versions + "\n")
	log_file.writelines(message_sample_identifiers + "\n")
	log_file.writelines(message_output_directory + "\n")
	log_file.writelines(["########################\n##### output files #####\n########################\n"])
	log_file.writelines(outpath_train + "\n")
	log_file.writelines(outpath_test + "\n")
	log_file.writelines(outpath_features + "\n")
	log_file.writelines(outpath_encoded_features + "\n")
	log_file.writelines(outpath_model + "\n")
	log_file.writelines(outpath_log + "\n")
	log_file.writelines(["########################\n## mean squared error ##\n########################\n"])
	log_file.writelines(f"from the training dataset: {mse_train} \n")
	log_file.writelines(f"from the testing dataset: {mse_test} \n")
	log_file.writelines(["########################\n### training dataset ###\n########################\n"])
	print(combined_train_df.to_string(index=False), file=log_file)
	log_file.writelines(["########################\n### testing  dataset ###\n########################\n"])
	print(combined_test_df.to_string(index=False), file=log_file)
	log_file.close()

	# print outputpath message
	print('The results are ready: ' + OUTPUTPATH)

elif args.subcommand == 'prediction':

	# read input files
	## mutations
	df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t')
	## features
	features =  pi.load(open(INPUTPATH_FEATURES, 'rb'))
	## encoded features
	encoded_features = pi.load(open(INPUTPATH_ENCODED_FEATURES, 'rb'))
	## model
	loaded_model = pi.load(open(INPUTPATH_MODEL, 'rb'))

	# replace missing genomic data by a sting
	df_mutations = df_mutations.fillna('missing')
	# rename labels of headers
	df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
	# sort by samples
	df_mutations = df_mutations.sort_values(by='sample')
	# prepare mutations indexing the sample columns
	X_mutations = df_mutations.set_index('sample')

	# encode categorical data into binary data for the dataset to predict
	## instantiate encoder object
	encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, categories = encoded_features).set_output(transform='pandas')
	# transform data with the encoder into an array for the testing dataset using produced features
	X_mutations_encoded = encoder.fit_transform(X_mutations[features])
	
	# perform prediction
	y_pred_mutations = loaded_model.predict(X_mutations_encoded)

	# prepare output results
	## transform numpy.ndarray into pandas.core.frame.DataFrame
	y_pred_mutations_df = pd.DataFrame(y_pred_mutations)
	## retrieve the sample index in a column
	y_samples_df = pd.DataFrame(X_mutations_encoded.reset_index().iloc[:, 0])
	## concatenate with reset index
	combined_mutations_df = pd.concat([y_samples_df.reset_index(drop=True), y_pred_mutations_df.reset_index(drop=True)], axis=1)
	## rename labels of headers
	combined_mutations_df.rename(columns={0: 'prediction'}, inplace=True)

	# check if the output directory does not exists and make it
	if not os.path.exists(OUTPUTPATH):
		os.makedirs(OUTPUTPATH)
		message_output_directory = "The output directory was created successfully"
		print(message_output_directory)
	else:
		message_output_directory = "The output directory already exists"
		print(message_output_directory)

	# output results
	## output path
	outpath_prediction = OUTPUTPATH + '/' + PREFIX + '_prediction' + '.tsv'
	outpath_log = OUTPUTPATH + '/' + PREFIX + '_prediction_log' + '.txt'
	## write output in a tsv file
	combined_mutations_df.to_csv(outpath_prediction, sep="\t", index=False, header=True)
	## write output in a txt file
	log_file = open(outpath_log, "w")
	log_file.writelines(["########################\n###### reference  ######\n########################\n"])
	print(parser.epilog, file=log_file)
	log_file.writelines(["########################\n####### versions #######\n########################\n"])
	log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
	log_file.writelines("argparse: " + str(ap.__version__) + "\n")
	log_file.writelines("pandas: " + str(pd.__version__) + "\n")
	log_file.writelines("pickle: " + str(pi.format_version) + "\n")
	log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
	log_file.writelines(["########################\n####### settings #######\n########################\n"])
	settings_str = str(args)
	settings_str = settings_str[:-1]
	settings_str = settings_str.replace(settings_str[:10], '')
	settings_str = settings_str.replace(", ", "\n")
	print(settings_str, file=log_file)
	log_file.writelines(["########################\n######## checks ########\n########################\n"])
	log_file.writelines(message_versions + "\n")
	log_file.writelines(message_output_directory + "\n")
	log_file.writelines(["########################\n##### output files #####\n########################\n"])
	log_file.writelines(outpath_prediction + "\n")
	log_file.writelines(outpath_log + "\n")
	log_file.writelines(["########################\n## prediction dataset ##\n########################\n"])
	print(combined_mutations_df.to_string(index=False), file=log_file)
	log_file.close()

	# print outputpath message
	print('The results are ready: ' + OUTPUTPATH)

# step control
step1_end = dt.datetime.now()
step1_diff = step1_end - step1_start

# print final message
print('The script lasted '+ str(step1_diff.microseconds) + ' μs')
print(parser.epilog)
