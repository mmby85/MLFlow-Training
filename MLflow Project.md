### **MLflow Projects Cheatsheet**  

MLflow Projects est un composant clé de MLflow permettant d’organiser et d’exécuter des projets reproductibles. Il repose sur des fichiers de configuration (`MLproject`) pour définir les dépendances et les paramètres nécessaires à l’exécution d’un workflow de Machine Learning.

---

## **1. Structure d’un Projet MLflow**  
Un projet MLflow doit contenir :  
1. **Un fichier `MLproject`** : Décrit les entrées, dépendances et commandes.  
2. **Un environnement reproductible** : Via `conda` ou un Dockerfile.  
3. **Scripts ou notebooks** : Le code à exécuter.

### **Exemple de structure d’un projet :**  
```plaintext
my_mlflow_project/
├── MLproject
├── train.py
├── conda.yaml
└── data/
    └── train.csv
```

---

## **2. Fichier `MLproject`**  
Le fichier `MLproject` est essentiel pour décrire le projet.  
Voici un exemple typique :  
```yaml
name: my_mlflow_project

conda_env: conda.yaml  # Définit l'environnement conda
# docker_env: docker_env.yaml  # Alternative : Définit l’environnement Docker

entry_points:
  train:
    parameters:
      alpha: {type: float, default: 0.5}  # Paramètre avec une valeur par défaut
      l1_ratio: {type: float, default: 0.01}
    command: >
      python train.py --alpha {alpha} --l1_ratio {l1_ratio}
```

### **Clés Importantes :**  
- `name` : Nom unique du projet.  
- `conda_env` ou `docker_env` : Décrit l’environnement d’exécution.  
- `entry_points` : Définit les points d’entrée avec les paramètres et la commande.

---

## **3. Exécution d’un Projet MLflow**  

### **Commande de base pour exécuter un projet localement :**  
```bash
mlflow run . -P alpha=0.75 -P l1_ratio=0.2
```

### **Exécution depuis un dépôt Git :**  
```bash
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.75
```

### **Exécution depuis un répertoire local :**  
```bash
mlflow run /path/to/project -P l1_ratio=0.5
```

### **Exécution d’un projet avec Docker :**  
Si le projet utilise un environnement Docker :  
```bash
mlflow run . --env-manager docker
```

### **Exécution d’un projet avec Conda :**  
Si le projet utilise un environnement Conda (par défaut) :  
```bash
mlflow run . --env-manager conda
```

---

## **4. Paramètres dans un Projet**  
Les paramètres permettent de personnaliser les exécutions. Ils sont définis dans `entry_points`.  

### **Définir des paramètres :**  
```yaml
parameters:
  alpha: {type: float, default: 0.5}
  max_iter: {type: int, default: 100}
  solver: {type: str, default: "lbfgs"}
```

### **Passer des paramètres en ligne de commande :**  
```bash
mlflow run . -P alpha=0.7 -P solver="saga"
```

---

## **5. Fichier `conda.yaml` pour l’Environnement**  
Le fichier `conda.yaml` contient la configuration de l’environnement :  
```yaml
name: my_env
channels:
  - defaults
dependencies:
  - python=3.8
  - pip
  - numpy
  - pandas
  - scikit-learn
  - pip:
      - mlflow
```

### **Créer un environnement Conda manuellement :**  
```bash
conda env create -f conda.yaml
```

---

## **6. Travailler avec Artefacts MLflow**  
Les artefacts générés par un projet MLflow peuvent être enregistrés et visualisés :  

### **Enregistrer un fichier comme artefact :**  
```python
mlflow.log_artifact("path/to/output_file.csv", artifact_path="results")
```

### **Accéder aux artefacts :**  
Après l'exécution d'un projet :  
```bash
mlflow artifacts list --run-id <run_id>
mlflow artifacts download --run-id <run_id> --artifact-path results
```

---

## **7. Intégration avec le Cloud**  

### **Exécution d’un Projet et Enregistrement des Artefacts sur S3**  
Configurez un bucket S3 comme stockage d’artefacts :  
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACTS_URI=s3://<bucket-name>
mlflow run . -P alpha=0.5
```

### **Exécution sur Databricks :**  
```bash
mlflow run databricks:///repos/<repo-path> \
    -P alpha=0.7 \
    --backend databricks
```

---

## **8. Exemple Complet**  
### **Fichier `MLproject` :**  
```yaml
name: regression-project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.01}
    command: >
      python train.py --alpha {alpha} --l1_ratio {l1_ratio}
```

### **Fichier `train.py` :**  
```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet

# Simulation des données
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

def train_model(alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--l1_ratio", type=float, required=True)
    args = parser.parse_args()
    
    mlflow.start_run()
    train_model(args.alpha, args.l1_ratio)
    mlflow.end_run()
```

### **Exécution :**  
```bash
mlflow run . -P alpha=0.7 -P l1_ratio=0.2
```

---

## **Liens Utiles**  
- [Documentation MLflow Projects](https://mlflow.org/docs/latest/projects.html)  
- [Exemples sur GitHub](https://github.com/mlflow/mlflow-example)  