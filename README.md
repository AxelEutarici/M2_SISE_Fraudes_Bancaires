# Detection de fraudes bancaires (Projet du cours de Fouilles de Données Massives M2 SISE) 

Also available in English : <br>
[![english](https://img.shields.io/badge/lang-english-red.svg)](https://github.com/AxelEutarici/SISE_Fraudes_Bancaires/blob/main/README.english.md)


## Sommaire

 - [Aperçu du contenu GitHub](#Aperçu-du-contenu-GitHub)
 - [Récuperation des données](#Récuperation-des-données)
 - [MLflow](#MLflow)


## Aperçu du contenu GitHub
Chaque dossier contient des script python au format jupyter notebook. 

`/1_preprocess` Contient le script qui permet de faire un rapide nettoyage des données ainsi que les scripts qui permettent de faire une selection de variable et le resampling <br>
`/2_descriptive_statistics` Contient le script qui permet de visualiser quelques statistiques descriptives du jeu de données initial avant pré-orocess<br>
`/3_methods` Contient un script de pré-process sur les données ainsi qu'un script de recherche du meilleur algorithme de classification<br>


## Récuperation des données

Vous pouvez trouver les jeux de données sur kaggle à travers ce [lien](https://www.kaggle.com/datasets/axeltrc/fraudes-bancaires-smotetomek10)<br>
Dans le notebook `code_methods` vous avez aussi le code pour importer le jeu de données directement depuis Kaggle en ligne de commande.

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
Les résultats des algorithmes de classification (f-score, AUC,...) sont stockés grâce à MLflow. Afin de le lancer, tapez cette commande dans un terminal.

```sh
mlflow ui
```



