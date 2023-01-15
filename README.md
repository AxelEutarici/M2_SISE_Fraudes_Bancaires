# Detection de fraudes bancaires (Projet du cours de Fouilles de Données Massives M2 SISE) 

## Sommaire

 - [Aperçu du contenu GitHub](#Aperçu-du-contenu-GitHub)
 - [Récuperation des données](#Récuperation-des-données)
 - [MLflow](#MLflow)
 - [Liens](#liens)

## Aperçu du contenu GitHub
Chaque dossier contient des script python au format jupyter notebook. 

`/nettoyage` Contient le script qui permet de faire un rapide nettoyage des données ainsi que les scripts qui permettent de faire une selection de variable et le resampling <br>
`/stat desc` Contient le script qui permet de visualiser quelques statistiques descriptives du jeu de données initial avant pré-orocess<br>
`/méthodes` Contient un script de pré-preocess sur les données ainsi qu'un script de recherche du meilleur algorithme de classification<br>


## Récuperation des données

Vous pouvez trouver les jeux de données sur kaggle à travers ce [lien](https://www.kaggle.com/datasets/axeltrc/fraudes-bancaires-smotetomek10)<br>
Dans le notebook `code_methodes` vous avez aussi le code pour importer le jeu de données directement depuis Kaggle en ligne de commande.

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



