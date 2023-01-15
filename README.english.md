# Fraud detections (School Project for 'Fouilles de Données Massives' SISE Master2 ) 

Aussi disponible en français : <br>
[![français](https://img.shields.io/badge/lang-français-green.svg)](https://github.com/AxelEutarici/SISE_Fraudes_Bancaires/blob/main/README.md)

## Table of contents

 - [Content of the GitHub Depository](#Content-of-the-GitHub-Depository)
 - [Downloading the datasets](#Downloading-the-datasets)
 - [MLflow](#MLflow)


## Content of the GitHub Depository
Each file contains jupyter notebook python script.

`/1_preprocess` Contains the data cleaning code as well as the variable selection code and the resampling one using three methods (SMOTE, TomekLinks, SMOTETomek) <br>
`/2_descriptive_statistics` Contains the script to visualize the data with graphs.<br>
`/3_methods` Contains the script to find the best algorithm for the classification problematic by stocking them with MLflow <br>


## Downloading the datasets

You can find the data sets on Kaggle by clicking this [link](https://www.kaggle.com/datasets/axeltrc/fraudes-bancaires-smotetomek10)<br>
In the `code_methodes` notebook you also can import the dataset directely from Kaggle with the following code. 

```sh
kaggle_username = '<enter username>'
kaggle_key = '< enter key >'
path = 'path/to/where/to/stock/datasets'
os.chdir(path)
```

```sh
!kaggle config set -n username -v $kaggle_username
!kaggle config set -n key -v $kaggle_key
!kaggle datasets download -d axeltrc/fraudes-bancaires-smotetomek10 -p $path
```

```sh
with zipfile.ZipFile('fraudes-bancaires-smotetomek10.zip', 'r') as zip_ref :
    zip_ref.extractall()
    for file in os.listdir():
        Xtrain = pd.read_csv('Xtrain_SMOTETomek.csv', sep=",")
        ytrain = pd.read_csv('ytrain_SMOTETomek.csv', sep=",")
        ytest = pd.read_csv('ytest.csv', sep=",")
        Xtest = pd.read_csv('Xtest.csv', sep=",")
```

## MLflow
The models and the results are stocked using MLflow. To launch MLFlow use this command in the terminal.
```sh
mlflow ui
```



