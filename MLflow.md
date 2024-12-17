### **MLflow Cheatsheet**  

---

## **Introduction à MLflow**
MLflow est une plateforme open-source pour la gestion du cycle de vie des modèles Machine Learning. Il prend en charge :  
1. **Suivi des expériences** (tracking).  
2. **Gestion des modèles** (model registry).  
3. **Reproductibilité des environnements**.  
4. **Déploiement des modèles** (deployment).  

---

## **Installation**
Ajoutez MLflow à votre projet :  
```bash
pip install mlflow
```

Pour activer le suivi des expériences avec un serveur MLflow local :  
```bash
mlflow ui
```

---

## **Configuration**
### **Suivi des expériences (Tracking)**
Définissez le **tracking URI** pour enregistrer vos expérimentations :  
```python
mlflow.set_tracking_uri("http://localhost:5000")  # Serveur local
# Ou pour utiliser un bucket S3 comme stockage d'artefacts :
mlflow.set_tracking_uri("s3://<your-bucket-name>/mlflow-tracking")
```

---

## **Fonctionnalités Principales**
### **1. Suivi des Expériences**
Enregistrez les hyperparamètres, métriques, et artefacts d’un modèle :  
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# Démarrer une nouvelle expérimentation
with mlflow.start_run():
    # Log des hyperparamètres
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Log des métriques
    mlflow.log_metric("mse", 0.25)
    
    # Log du modèle entraîné
    model = RandomForestRegressor(n_estimators=100, max_depth=5)
    mlflow.sklearn.log_model(model, "model")
```

### **2. Gestion des Artefacts**
Pour enregistrer des fichiers ou des graphiques en tant qu’artefacts :  
```python
mlflow.log_artifact("path/to/local/file", "artefacts")
```

---

## **Utilisation du Model Registry**
MLflow permet d’enregistrer, de versionner et de déployer des modèles :  

### **Enregistrer un modèle dans le registre**
```python
mlflow.register_model("runs:/<run_id>/model", "MyModelName")
```

### **Changer le statut d’un modèle**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="MyModelName",
    version=1,
    stage="Production"
)
```

---

## **Lancer MLflow avec un Backend Distante**
Pour utiliser un bucket S3 comme stockage d’artefacts et une base de données comme stockage principal :  
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://<your-bucket-name>/mlflow-artifacts \
    --host 0.0.0.0
```

---

## **Commandes Utiles**
| Commande | Description |
|----------|-------------|
| `mlflow ui` | Lancer l’interface utilisateur MLflow. |
| `mlflow run` | Exécuter un projet MLflow défini dans un fichier `MLproject`. |
| `mlflow experiments create --experiment-name <name>` | Créer une nouvelle expérience. |
| `mlflow experiments list` | Lister toutes les expériences. |

---

## **Exemple Complet**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Charger les données
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Démarrer une expérimentation
with mlflow.start_run():
    # Entraîner un modèle
    model = RandomForestRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Log des hyperparamètres et des métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mse = mean_squared_error(y_test, model.predict(X_test))
    mlflow.log_metric("mse", mse)
    
    # Log du modèle
    mlflow.sklearn.log_model(model, "model")
```

---

## **Liens Utiles**
- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)  
- [Guide du Tracking MLflow](https://mlflow.org/docs/latest/tracking.html)  
- [GitHub MLflow](https://github.com/mlflow/mlflow)  

--- 
**Astuce :** Assurez-vous que vos configurations (comme les clés AWS) sont correctement définies dans les variables d’environnement pour éviter des erreurs de permission avec des artefacts S3. 

---

```python
import mlflow
from mlflow.tracking import MlflowClient

def display_experiment_runs_metadata():
    """
    Liste les expériences, les runs, sélectionne un run et affiche ses métadonnées.
    """
    client = MlflowClient()

    print("------------------- Experiments --------------------")
    experiments = client.search_experiments()
    for exp in experiments:
        print(f"Experiment Name: {exp.name}, ID: {exp.experiment_id}")

    if not experiments:
        print("No experiments found. Please run an MLflow experiment first.")
        return

    print("--------------------- Runs -----------------------")
    all_runs = client.search_runs()
    if not all_runs:
        print("No runs found for the current experiment. Please run an MLflow experiment first.")
        return
    for run in all_runs:
        print(f"Run ID: {run.info.run_id}, Experiment ID: {run.info.experiment_id}")
    
    selected_run_id = input("Enter the run ID of the run you want to inspect :")

    # Rechercher le run sélectionné
    try:
        selected_run = client.get_run(selected_run_id)
    except Exception as e:
        print(f"Error: Run with ID '{selected_run_id}' not found or an error occured. Details: {e}")
        return
        
    print("----------------- Selected Run Metadata ------------------")
    print(f"Run ID: {selected_run.info.run_id}")
    print(f"Experiment ID: {selected_run.info.experiment_id}")
    print(f"Start Time: {selected_run.info.start_time}")
    print(f"Status: {selected_run.info.status}")
    print("--------------------- Parameters -----------------------")
    for key, value in selected_run.data.params.items():
      print(f"  {key}: {value}")

    print("--------------------- Metrics -----------------------")
    for key, value in selected_run.data.metrics.items():
      print(f"  {key}: {value}")

    print("--------------------- Tags -----------------------")
    for key, value in selected_run.data.tags.items():
       print(f"  {key}: {value}")

    print("--------------------- Artifacts -----------------------")
    artifacts_list = client.list_artifacts(selected_run_id)
    for artifact in artifacts_list:
        print(f"  {artifact.path}")

if __name__ == "__main__":
    display_experiment_runs_metadata()

```