# Usage
The repository GenomicBasedRegression provides a Python (recommended version 3.12) script called GenomicBasedRegression:1.0.py to perform linear regression-based modeling or prediction from categorical genomic data.
# Dependencies
The Python script GenomicBasedRegression:1.0.py was prepared and tested with the Python version 3.12 and Ubuntu 20.04 LTS Focal Fossa.
- pandas # version 2.2.2
- sklearn # version 1.5.2
# Recommended environments
## install python libraries
```
# pip3.12 install pandas==2.2.2
# pip3.12 install -U scikit-learn==1.5.2
```
## install docker image
```
docker pull nicolasradomski/genomicbasedregression:1.0
```
# Helps
## modeling
```
usage: GenomicBasedLinearRegression:1.0.py modeling [-h] -m INPUTPATH_MUTATIONS -p INPUTPATH_PHENOTYPES [-s SPLITTING] [-o OUTPUTPATH] [-x PREFIX] [-d DEBUG] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes
                        (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). (REQUIRED)
  -p INPUTPATH_PHENOTYPES, --phenotypes INPUTPATH_PHENOTYPES
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes. First column: sample identifiers identical to those in the input file of mutations
                        (header: e.g. sample). Second column: continuous phenotype (header: e.g. phenotype). (REQUIRED)
  -s SPLITTING, --split SPLITTING
                        Percentage of random splitting to prepare the training dataset through the holdout method. (DEFAULT: 80)
  -o OUTPUTPATH, --output OUTPUTPATH
                        Output path. (DEFAULT: .)
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. (DEFAULT: output)
  -d DEBUG, --debug DEBUG
                        Traceback level when an error occurs. (DEFAULT: 0)
  -nc, --no-check       Do not check versions of Python and packages. (DEFAULT: False)
```
## prediction
```
usage: GenomicBasedLinearRegression:1.0.py prediction [-h] -m INPUTPATH_MUTATIONS -t INPUTPATH_MODEL -f INPUTPATH_FEATURES -ef INPUTPATH_ENCODED_FEATURES [-o OUTPUTPATH] [-x PREFIX] [-d DEBUG] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes
                        (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). (REQUIRED)
  -t INPUTPATH_MODEL, --model INPUTPATH_MODEL
                        Absolute or relative input path of an object (obj) file including a trained scikit-learn model. (REQUIRED)
  -f INPUTPATH_FEATURES, --features INPUTPATH_FEATURES
                        Absolute or relative input path of an object (obj) file including trained scikit-learn features (i.e. mutations). (REQUIRED)
  -ef INPUTPATH_ENCODED_FEATURES, --encodedfeatures INPUTPATH_ENCODED_FEATURES
                        Absolute or relative input path of an object (obj) file including trained scikit-learn encoded features (i.e. mutations). (REQUIRED)
  -o OUTPUTPATH, --output OUTPUTPATH
                        Absolute or relative output path. (DEFAULT: .)
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. (DEFAULT: output_)
  -d DEBUG, --debug DEBUG
                        Traceback level when an error occurs. (DEFAULT: 0)
  -nc, --no-check       Do not check versions of Python and packages. (DEFAULT: False)
```
# Expected input files
## phenotypes for training (e.g. phenotypes.tsv)
```
sample	feature
S1.1.01	24
S1.1.02	25
S1.1.03	16
S1.1.04	48
S1.1.05	14
S1.1.06	16
S1.1.07	43
S1.1.08	17
S1.1.09	58
S1.1.10	46
S1.1.11	15
S1.1.12	46
S1.1.13	13
S1.1.14	5
S1.1.15	6
S1.1.16	7
S1.1.17	8
S1.1.19	2
S1.1.18	4
S1.1.20	4
```
## genomic data for training (e.g. genomic-profils-for-training.tsv). "A" and "L" stand for alleles and locus, respectively
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10
S1.1.01	A3	A2	A3	A4	A5	A6	A7	A3	A4	A10
S1.1.02	A8	A5	A3	A4	A5	A6	A7	A3	A4	A10
S1.1.03	A6	A7	A6	A2	A17	A5	A6	A7	A8	A18
S1.1.04	A12	A13	A8	A5	A16	A4	A5	A6	A12	A17
S1.1.05	A6	A7	A15	A16	A3	A14	A6	A7	A8	A18
S1.1.06	A6	A7	A15	A16	A8	A5	A6	A7	A8	A18
S1.1.07	A7		A9	A10	A11	A14	A3	A2	A10	A16
S1.1.08	A6	A7	A15	A16	A17	A5	A7	A5	A8	A18
S1.1.09	A12	A13	A14	A15	A16	A4	A5	A6	A3	A2
S1.1.10	A12	A13	A14	A15	A16	A4	A5	A6	A8	A8
S1.1.11	A6	A7	A15	A16	A17	A5	A3	A2	A8	A18
S1.1.12	A12	A13	A14	A15	A16	A4		A8	A12	A17
S1.1.13	A1	A2	A3	A4		A14	A7	A3	A4	A10
S1.1.14	A7	A8	A16	A17	A15	A5	A7	A8	A9	A19
S1.1.16	A7	A8	A8	A7	A18	A6	A7	A8	A9	A19
S1.1.17	A7	A8	A8	A5	A18	A6	A7	A8	A9	A19
S1.1.18	A8	A2	A6	A7	A8	A9	A10	A13	A7	A13
S1.1.19	A4	A5	A16	A17	A18	A6	A7	A8	A9	A19
S1.1.15	A4	A5	A6	A7	A8	A9	A10	A13	A7	A13
S1.1.20	A4	A5	A6	A7	A8	A9	A10	A13	A7	A13
```
## genomic profils for prediction (e.g. genomic-profils-for-prediction.tsv)
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10	L_11
S2.1.01	A3	A2	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.02	A8	A5	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.03	A6	A7	A6	A2	A17	A5	A6	A7	A8	A1	A18
S2.1.04		A13	A8	A5	A16	A4	A5	A6	A12	A1	A17
S2.1.05	A6	A24	A15	A16	A3	A14	A6	A7	A8	A1	A18
S2.1.06	A6	A7	A15	A16	A8	A5	A6	A7	A8	A1	A18
S2.1.07	A7	A8	A9	A10	A11	A14	A3	A2	A88	A1	A16
S2.1.08	A6	A7	A15	A16	A17	A5	A7	A5	A8	A1	A18
S2.1.09	A12	A13	A14	A25	A16	A4	A5		A3	A1	A2
S2.1.10	A12	A13	A14	A15	A16	A4	A5	A6	A8	A1	A8
S2.1.11	A6	A7	A15	A16	A17	A5	A3	A2	A8	A2	A18
S2.1.12	A12	A13	A14	A15	A16	A4	A8	A8	A12	A2	A17
S2.1.13	A1	A2	A3	A4		A14	A7	A3	A4	A2	A10
S2.1.14	A7	A8	A16	A17	A15	A5	A7	A8	A9	A2	
S2.1.16	A7	A8	A8	A7	A18	A6	A57	A8	A9	A2	A19
S2.1.17	A7	A8	A8	A5	A18	A6	A7	A8	A9	A2	A19
S2.1.18	A8	A2	A6	A7	A8	A9	A10	A13	A7	A2	A13
S2.1.19	A4	A5	A16	A17	A18	A6	A7	A8	A9	A2	A19
S2.1.15	A4	A5	A6	A7	A8	A9	A10	A13	A7	A2	A13
S2.1.20	A4	A5	A6	A7	A8	A9	A10	A13	A7	A2	A13
```
# Examples of commands
## Import the GitHub repository
```
git clone https://github.com/Nicolas-Radomski/GenomicBasedRegression.git
cd GenomicBasedRegression
```
## with Python
### call help
```
python3.12 GenomicBasedRegression:1.0.py -h
python3.12 GenomicBasedRegression:1.0.py modeling -h
python3.12 GenomicBasedRegression:1.0.py prediction -h
```
### build a model
```
python3.12 GenomicBasedRegression:1.0.py modeling -m genomic-profils-for-modeling.tsv -p phenotypes.tsv -o MyDirectory -x FirstAnalysis -s 80 -d 4
```
### predict with a pre-built model
```
python3.12 GenomicBasedRegression:1.0.py prediction -m genomic-profils-for-prediction.tsv -t MyDirectory/FirstAnalysis_model.obj -f MyDirectory/FirstAnalysis_features.obj -ef MyDirectory/FirstAnalysis_encoded_features.obj -o MyDirectory -x SecondAnalysis -d 4
```
## with Docker
### call help
```
docker run --rm --name nicolas -u $(id -u):$(id -g) nicolasradomski/genomicbasedregression:1.0 -h
docker run --rm --name nicolas -u $(id -u):$(id -g) nicolasradomski/genomicbasedregression:1.0 modeling -h
docker run --rm --name nicolas -u $(id -u):$(id -g) nicolasradomski/genomicbasedregression:1.0 prediction -h
```
### build a model
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd genomicbasedregression:1.0 modeling -m genomic-profils-for-modeling.tsv -p phenotypes.tsv -o MyDirectoryDockerHub -x FirstAnalysis -s 80 -d 4
```
### predict with a pre-built model
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd genomicbasedregression:1.0 prediction -m genomic-profils-for-prediction.tsv -t MyDirectoryDockerHub/FirstAnalysis_model.obj -f MyDirectoryDockerHub/FirstAnalysis_features.obj -ef MyDirectoryDockerHub/FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x SecondAnalysis -d 4
```
# Expected output files (see corresponding output directory)
## training
```
 sample  expectation  prediction
S1.1.20            4         4.0
S1.1.18            4         4.0
S1.1.17            8         8.0
S1.1.07           43        43.0
S1.1.06           16        16.0
S1.1.01           24        24.0
S1.1.12           46        46.0
S1.1.04           48        48.0
S1.1.05           14        14.0
S1.1.03           16        16.0
S1.1.13           13        13.0
S1.1.11           15        15.0
S1.1.09           58        58.0
S1.1.02           25        25.0
S1.1.08           17        17.0
S1.1.19            2         2.0
```
## testing
```
 sample  expectation  prediction
S1.1.14            5    9.680897
S1.1.16            7    6.388594
S1.1.10           46   51.578045
S1.1.15            6    4.000000
```
## prediction
```
 sample  prediction
S2.1.01   23.840727
S2.1.02   19.799411
S2.1.03   16.871680
S2.1.04   41.563688
S2.1.05   15.743359
S2.1.06   16.871680
S2.1.07   33.283019
S2.1.08   16.493532
S2.1.09   41.468397
S2.1.10   45.375678
S2.1.11   24.762863
S2.1.12   41.870374
S2.1.13   12.840727
S2.1.14    8.268915
S2.1.15    7.128839
S2.1.16   13.378558
S2.1.17   11.268915
S2.1.18    6.128839
S2.1.19    5.268915
S2.1.20    7.128839
```
# Illustration
![workflow figure](https://github.com/Nicolas-Radomski/GenomicBasedRegression/blob/main/illustration.png)
# Reference
Ricerca Corrente - IZS AM 06/24 RC: "genomic data-based machine learning to predict categorical and continuous phenotypes by classification and regression".
# Please site
https://github.com/Nicolas-Radomski/GenomicBasedRegression
https://hub.docker.com/r/nicolasradomski/genomicbasedregression
# Acknowledgment
The GENPAT-IZSAM Staff for our discussions aiming at designing workflows.
# Author
Nicolas Radomski
