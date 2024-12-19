

# Tutoriel : Déploiement de Modèles avec MLflow 

Ce tutoriel explore les différentes méthodes de déploiement de modèles de Machine Learning (ML) avec MLflow. Nous couvrirons les approches les plus courantes, avec des exemples pratiques pour faciliter leur mise en œuvre.

## Introduction au Déploiement avec MLflow

MLflow offre des outils robustes pour le suivi des expériences, la gestion des modèles et leur déploiement. Le déploiement est l'étape qui suit l'entraînement d'un modèle, où il est rendu disponible pour faire des prédictions sur de nouvelles données. MLflow permet de déployer des modèles de différentes manières, allant du simple service local à des environnements cloud complexes.

## Types de Déploiement avec MLflow

MLflow propose plusieurs options de déploiement, qui répondent à différents besoins :

1.  **Déploiement Local (Serveur REST MLflow)** : Utiliser un serveur REST MLflow pour héberger localement un modèle. Idéal pour le développement, les tests et des petites applications.
2.  **Déploiement sous forme d'Application Python (MLflow Serving)** : Intégrer un modèle dans une application Python et exposer une API. Utile pour un déploiement plus intégré.
3.  **Déploiement avec Docker (MLflow Deployments)** : Conteneuriser un modèle avec Docker pour un déploiement portable et isolé. Bon choix pour la scalabilité et la cohérence.
4.  **Déploiement sur des plateformes cloud (MLflow Deployments)** : Déployer des modèles directement sur des plateformes de cloud (AWS SageMaker, Azure ML, etc.). Pratique pour les déploiements à grande échelle.

Examinons chacune de ces méthodes en détail.

## 1. Déploiement Local (Serveur REST MLflow)

### Description

MLflow fournit un serveur REST intégré pour servir les modèles. C'est la méthode la plus simple pour exposer un modèle via une API locale.

### Avantages

*   Facile à mettre en place.
*   Idéal pour le développement et le test.
*   Permet de faire des prédictions via des requêtes HTTP.

### Inconvénients

*   Non conçu pour les environnements de production.
*   Peu scalable et moins sécurisé.

### Exemple Pratique

**1. Enregistrer un Modèle**

Supposons que nous ayons un modèle simple que nous avons enregistré avec MLflow :

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Chargement des données
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

# Enregistrement du modèle
with mlflow.start_run() as run:
  mlflow.sklearn.log_model(model, "iris_model")
  model_uri = mlflow.get_artifact_uri("iris_model")
  run_id = run.info.run_id
print(f"Le model a été sauvegardé avec l'uri: {model_uri}, run_id: {run_id}")
```

**2. Lancer le Serveur MLflow**

Ouvrez un terminal et exécutez la commande suivante en remplaçant `<run_id>` avec le `run_id` donné par la sortie du code python ci-dessus :

```bash
mlflow models serve --model-uri runs:/<run_id>/iris_model --port 5000
```

**3. Envoyer une Requête de Prédiction**

Dans un autre terminal, envoyez une requête POST au serveur :

```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' http://127.0.0.1:5000/invocations
```

Le serveur retournera une prédiction JSON.

## 2. Déploiement sous forme d'Application Python (MLflow Serving)

### Description

Cette méthode consiste à intégrer le modèle dans une application Python, qui peut utiliser une bibliothèque web comme Flask pour créer une API.

### Avantages

*   Flexibilité maximale pour personnaliser l'API.
*   Contrôle précis sur la logique de prédiction.
*   Adaptable à divers frameworks d'API web.

### Inconvénients

*   Demande plus de configuration et de code.
*   Nécessite une bonne connaissance du framework d'API web utilisé.

### Exemple Pratique (Flask)

**1. Créer l'Application Flask**

Créez un fichier `app.py` avec le contenu suivant :

```python
import mlflow
from flask import Flask, request, jsonify
import numpy as np
import os

# Chargement du modèle
# On vérifie si une variable d'environnement MLFLOW_MODEL_URI est fournie
model_uri = os.environ.get("MLFLOW_MODEL_URI")
if not model_uri:
   raise Exception("Please set the MLFLOW_MODEL_URI environment variable")

loaded_model = mlflow.sklearn.load_model(model_uri)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if "inputs" not in data:
        return jsonify({'error':'Please provide an array of "inputs"'}), 400
    
    inputs = data["inputs"]
    try:
        inputs = np.array(inputs)
        prediction = loaded_model.predict(inputs).tolist() # Make the prediction with the loaded model
    except:
        return jsonify({'error':'Please provide numerical values in inputs'}), 400
    return jsonify({'predictions': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

**2. Exécuter l'Application Flask**
Avant d'exécuter l'application, nous devons passer à l'application l'URI du model à charger. Pour ce faire, dans votre terminal, exécuter :
```bash
export MLFLOW_MODEL_URI="runs:/<run_id>/iris_model"
```
Notez que `<run_id>` doit être remplacé par le `run_id` fourni lors de l'enregistrement du model.
Puis lancez l'application Flask :

```bash
python app.py
```
**3. Envoyer une Requête de Prédiction**

Dans un autre terminal, envoyez une requête POST au serveur Flask :
```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' http://127.0.0.1:5001/predict
```
Le serveur Flask retournera une prédiction JSON.

## 3. Déploiement avec Docker (MLflow Deployments)

### Description

Cette approche consiste à conteneuriser votre modèle à l'aide de Docker, ce qui permet de créer un environnement portable, isolé et reproductible.

### Avantages

*   Déploiement reproductible sur différents environnements.
*   Facilite la mise à l'échelle et la gestion des dépendances.
*   Compatible avec des orchestrateurs de conteneurs comme Kubernetes.

### Inconvénients

*   Demande une connaissance de Docker.
*   Nécessite une étape de construction d'image Docker.

### Exemple Pratique

**1.  Créer un fichier `Dockerfile`**

Créez un fichier `Dockerfile` à côté de `app.py` :

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 5001

CMD ["python", "app.py"]
```
**2. Créer un fichier `requirements.txt`**
Créez un fichier `requirements.txt` à côté de `app.py` et `Dockerfile` avec le contenu suivant :

```txt
mlflow
scikit-learn
Flask
numpy
```

**3. Construire l'Image Docker**
Dans le même dossier que le `Dockerfile`, exécutez la commande suivante :
```bash
docker build -t mlflow-app .
```
**4. Exécuter le Container Docker**
Avant d'exécuter le container, nous devons passer à l'application l'URI du model à charger. Pour ce faire, dans votre terminal, exécuter :
```bash
export MLFLOW_MODEL_URI="runs:/<run_id>/iris_model"
```
Notez que `<run_id>` doit être remplacé par le `run_id` fourni lors de l'enregistrement du model.
Puis, dans le terminal, exécutez :
```bash
docker run -p 5001:5001 -e MLFLOW_MODEL_URI=$MLFLOW_MODEL_URI mlflow-app
```

**5. Envoyer une Requête de Prédiction**

Dans un autre terminal, envoyez une requête POST au serveur Docker :

```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' http://127.0.0.1:5001/predict
```

Le serveur Docker retournera une prédiction JSON.

## 4. Déploiement sur des Plateformes Cloud (MLflow Deployments)

### Description

MLflow permet aussi de déployer vos modèles directement sur des services de cloud comme AWS SageMaker ou Azure ML.

### Avantages

*   Scalabilité et fiabilité des services cloud.
*   Moins de gestion d'infrastructure.
*   Intégration avec les outils de déploiement cloud.

### Inconvénients

*   Dépendance d'une plateforme cloud spécifique.
*   Nécessite une configuration spécifique pour la plateforme cloud choisie.


### Exemple Pratique (AWS SageMaker)

**1. Configuration d'AWS SageMaker**

*   Assurez-vous que vous avez un compte AWS et que vous avez configuré les permissions nécessaires pour SageMaker.
*   Votre modèle doit être enregistré dans le registre de modèles de MLflow.
*   Utiliser l'interface en ligne de commandes AWS (`awscli`) pour pouvoir déployer des modèles.

**2. Déploiement via CLI**

```bash
mlflow deployments create --target sagemaker \
    -m 'runs:/<run_id>/iris_model' \
    --name iris-deployment \
    --region <aws_region> \
    --flavor sklearn
```
**3. Tester le déploiement**
Une fois déployé vous pouvez tester le model avec la commande :

```bash
mlflow deployments predict  \
  --target sagemaker \
  --name iris-deployment  \
  --input-path test.json \
  --region <aws_region>
```
Où le fichier `test.json` a le format :
```json
{"inputs": [[5.1, 3.5, 1.4, 0.2]]}
```

**Note:**  Le déploiement vers Azure ML ou d'autres plateformes cloud se fait de manière similaire en changeant les commandes et en suivant la [documentation de MLflow](https://mlflow.org/docs/latest/deployments_index.html).


## Déploiement de Modèles MLflow sur Databricks

Cette section du tutoriel se concentre sur le déploiement de modèles MLflow directement sur Databricks, une plateforme cloud de traitement de données et d'apprentissage automatique.

### Description

Databricks offre une intégration native avec MLflow, simplifiant le déploiement de modèles directement dans leur environnement. Cette méthode tire parti des capacités de calcul de Databricks et de sa gestion des environnements.

### Avantages

*   **Intégration facile** avec l'écosystème Databricks.
*   **Scalabilité automatique** grâce aux clusters Databricks.
*   **Gestion de l'environnement** et des dépendances simplifiée.
*   **Facilité de mise en production** pour les projets développés sur Databricks.

### Inconvénients

*   **Dépendance** à l'écosystème Databricks.
*   Nécessite un **compte Databricks** et une configuration adéquate.
*   Moins de flexibilité si vous avez besoin de déployer en dehors de l'environnement Databricks.

### Exemple Pratique

**Prérequis :**

*   Vous devez avoir un espace de travail Databricks.
*   Vous devez avoir un modèle MLflow enregistré (dans le registre de modèles Databricks, ou un autre registre d'MLflow compatible avec Databricks).
*   Votre espace de travail Databricks doit avoir configuré les permissions nécessaires.
*   Vous devez avoir le CLI `databricks` configuré.

**1. Enregistrer un Modèle (Si ce n'est pas déjà fait)**

Si vous n'avez pas encore de modèle enregistré, vous pouvez utiliser le code suivant (similaire à l'exemple précédent) pour enregistrer un modèle dans MLflow :

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Chargement des données
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

# Enregistrement du modèle
with mlflow.start_run() as run:
  mlflow.sklearn.log_model(model, "iris_model")
  model_uri = mlflow.get_artifact_uri("iris_model")
  run_id = run.info.run_id
print(f"Le model a été sauvegardé avec l'uri: {model_uri}, run_id: {run_id}")
```
Notez que le modèle sera automatiquement enregistré dans le registre de modèles de votre espace de travail Databricks.

**2. Déploiement via CLI Databricks**

Utilisez le CLI Databricks pour déployer le modèle. Vous aurez besoin de l'URI de votre modèle.

```bash
databricks models serve --name iris-deployment --model-uri runs:/<run_id>/iris_model
```

Remplacez `<run_id>` par l'identifiant de l'exécution MLflow où le modèle a été enregistré.

**3. Vérification du Déploiement (Via l'Interface Databricks)**
Après avoir exécuté la commande ci-dessus, vous pouvez vérifier que le déploiement du model a été fait, en allant sur l'interface web de votre workspace, puis dans la section `Serving` dans le menu de gauche.

Vous devriez y trouver le nom du déploiement `iris-deployment`. Vous pouvez cliquer sur l'interface du déploiement pour voir l'interface vous donnant les informations de comment contacter le model déployé.

**4. Tester le Déploiement (Via l'interface databricks ou via une commande curl)**

*   **Via l'interface Databricks:** Sur l'interface web du déploiement, vous pouvez copier l'endpoint REST, et fournir des exemples dans l'onglet `Query` pour tester le modèle directement.

*  **Via une commande curl:** Vous pouvez utiliser une commande curl pour requêter le endpoint REST (que vous aurez trouvé dans l'interface) de la manière suivante (en remplaçant les bons identifiants) :
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -H "Authorization: Bearer <your_databricks_token>" \
         -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' \
         <your_databricks_endpoint>/invocations
    ```
    Remplacez `<your_databricks_token>` par votre token d'accès à Databricks, et `<your_databricks_endpoint>` par l'url de votre endpoint.
    Vous devriez recevoir la prédiction de votre model.

**Note:**

*   Vous pouvez aussi déployer une version spécifique d'un modèle enregistré dans le registre de modèles.
*   Vous pouvez contrôler le cluster sur lequel le modèle est déployé (taille, type, etc.) grâce aux paramètres de la commande de déploiement de databricks.

---

## Déploiement de Modèles MLflow sur Kubernetes (k8s)

Cette section ajoute une méthode de déploiement très robuste : l'utilisation de Kubernetes. Kubernetes est un système d'orchestration de conteneurs open-source qui automatise le déploiement, la mise à l'échelle et la gestion des applications conteneurisées.

### Description

Le déploiement sur Kubernetes implique la création de conteneurs Docker pour vos modèles MLflow, puis leur déploiement et gestion via Kubernetes. Cela offre une grande scalabilité, une tolérance aux pannes et une gestion centralisée des applications.

### Avantages

*   **Scalabilité** : Mise à l'échelle dynamique des modèles en fonction de la demande.
*   **Haute disponibilité** : Tolérance aux pannes et redondance.
*   **Gestion centralisée** : Configuration et gestion des applications simplifiées.
*   **Flexibilité** : Prise en charge de différents types de déploiement (blue/green, canary).
*   **Ecosystème** : Large communauté et outils disponibles pour la surveillance et la gestion.

### Inconvénients

*   **Complexité** : Nécessite une bonne compréhension de Kubernetes et de ses concepts.
*   **Configuration** : Plus de configuration initiale par rapport à des déploiements plus simples.
*   **Maintenance** : La gestion et la maintenance d'un cluster Kubernetes peuvent être complexes.

### Exemple Pratique (Utilisation d'AWS EKS comme exemple)

Dans cet exemple, nous allons utiliser AWS Elastic Kubernetes Service (EKS) comme cluster Kubernetes, mais les concepts généraux s'appliquent à n'importe quel cluster Kubernetes. Nous allons éviter d'entrer dans les détails techniques spécifiques d'AWS, et nous concentrer sur les étapes générales nécessaires au déploiement sur k8s.

**Prérequis :**

*   Vous devez avoir un cluster Kubernetes (AWS EKS, GCP GKE, etc.) disponible et fonctionnel.
*   Vous devez avoir `kubectl`, l'outil de ligne de commande de Kubernetes, installé et configuré pour se connecter à votre cluster.
*   Un registre de conteneurs (AWS ECR, Docker Hub, etc) doit être disponible pour stocker votre image Docker.
*   Vous devez avoir un modèle MLflow enregistré.

**1. Créer une Image Docker (Si ce n'est pas déjà fait)**

Si vous n'avez pas encore d'image Docker, vous pouvez utiliser le `Dockerfile` et le `requirements.txt` (vu précédemment) avec le code suivant:

```python
import mlflow
from flask import Flask, request, jsonify
import numpy as np
import os

# Chargement du modèle
# On vérifie si une variable d'environnement MLFLOW_MODEL_URI est fournie
model_uri = os.environ.get("MLFLOW_MODEL_URI")
if not model_uri:
   raise Exception("Please set the MLFLOW_MODEL_URI environment variable")

loaded_model = mlflow.sklearn.load_model(model_uri)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if "inputs" not in data:
        return jsonify({'error':'Please provide an array of "inputs"'}), 400
    
    inputs = data["inputs"]
    try:
        inputs = np.array(inputs)
        prediction = loaded_model.predict(inputs).tolist() # Make the prediction with the loaded model
    except:
        return jsonify({'error':'Please provide numerical values in inputs'}), 400
    return jsonify({'predictions': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

et le Dockerfile :
```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 5001

CMD ["python", "app.py"]
```
et le `requirements.txt`:

```txt
mlflow
scikit-learn
Flask
numpy
```

Vous devez ensuite construire l'image Docker en vous positionnant dans le même répertoire que les fichiers précédents :

```bash
docker build -t <your_docker_registry>/mlflow-app:<tag> .
```

Remplacez `<your_docker_registry>` par l'URI de votre registre Docker et `<tag>` par le tag de votre image.

Vous devrez également charger l'image construite dans votre registre Docker:
```bash
docker push <your_docker_registry>/mlflow-app:<tag>
```
**2. Définir les Fichiers de Déploiement Kubernetes**
Créez un fichier `deployment.yaml` (qui décrit le déploiement) avec un contenu similaire à ce qui suit :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-app-deployment
  labels:
    app: mlflow-app
spec:
  replicas: 2 # Nombre de réplicas de votre application
  selector:
    matchLabels:
      app: mlflow-app
  template:
    metadata:
      labels:
        app: mlflow-app
    spec:
      containers:
      - name: mlflow-app-container
        image: <your_docker_registry>/mlflow-app:<tag> # Remplacez par l'image créée
        ports:
        - containerPort: 5001
        env:
          - name: MLFLOW_MODEL_URI
            value: "runs:/<run_id>/iris_model"  # Remplacez par l'URI de votre modèle MLflow
```
Remplacez `<your_docker_registry>`, `<tag>`, et `<run_id>` par les valeurs appropriées.

Créez un fichier `service.yaml` (qui expose votre application) avec un contenu similaire à ce qui suit :

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mlflow-app-service
spec:
  selector:
    app: mlflow-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5001
  type: LoadBalancer
```

Ce `service.yaml` expose votre application au travers d'un load balancer.

**3. Déployer sur Kubernetes**

Utilisez `kubectl` pour déployer l'application en exécutant les commandes suivantes dans le répertoire ou vous avez créé les fichiers:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

**4. Tester le Déploiement**

Attendez que le service ait été provisionné, et récupérer l'adresse externe de votre service pour pouvoir envoyer des requêtes.

Vous pouvez utiliser `kubectl get services` pour obtenir l'adresse externe du Load Balancer. Une fois que vous avez l'adresse, vous pouvez envoyer des requêtes `curl` vers le modèle :

```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' http://<adresse_externe>:80/predict
```

Remplacez `<adresse_externe>` par l'adresse externe de votre load balancer.

## Points Importants

*   **Environnement** : Votre cluster Kubernetes doit avoir accès à votre registre d'images et à votre modèle MLflow.
*   **Sécurité** : Configurez correctement les accès et les politiques de sécurité de votre cluster.
*   **Surveillance** : Utilisez des outils de surveillance (Prometheus, Grafana) pour suivre les performances de votre modèle déployé.
*   **Mise à l'échelle** : Adaptez le nombre de réplicas en fonction de la demande.
*   **Versionning** : Gérez le versioning de vos images Docker et de vos modèles.

## Conclusion (Ajout)

Cette section a enrichi le tutoriel en ajoutant une approche de déploiement très robuste et utilisée en production : le déploiement sur Kubernetes. Bien que cette méthode demande une bonne connaissance de Kubernetes, elle offre une scalabilité et une flexibilité qui font d'elle un excellent choix pour des applications critiques. En combinant cette approche avec les méthodes présentées précédemment, vous disposez d'un éventail complet de solutions pour le déploiement de modèles MLflow. N'hésitez pas à consulter la documentation officielle de Kubernetes et d'MLflow pour aller plus loin.

---