
---

# Optuna : Optimisation d'Hyperparamètres Détaillée

## Introduction à Optuna

### Optuna : L'Optimisation d'Hyperparamètres Facilitée et Efficace

*   **Optuna** est une bibliothèque Python puissante et flexible pour l'optimisation d'hyperparamètres.
*   Conçue pour être **intuitive et facile à utiliser**.
*   Offre des **algorithmes d'optimisation avancés** et des outils de visualisation.
*   Idéale pour améliorer les performances des modèles d'**apprentissage automatique**.

## Concepts Clés d'Optuna

### Les Fondamentaux d'Optuna

*   **Étude (Study)** : Un objet qui représente l'ensemble du processus d'optimisation. Il stocke tous les essais (trials) et leurs résultats.
*   **Essai (Trial)** : Une seule exécution de la fonction objective avec un ensemble spécifique d'hyperparamètres.
*   **Fonction Objective (Objective Function)** : La fonction que vous souhaitez optimiser (minimiser ou maximiser). Elle reçoit les hyperparamètres et retourne une valeur scalaire.
*   **Échantillonneur (Sampler)** : L'algorithme utilisé pour suggérer de nouveaux ensembles d'hyperparamètres (Random, TPE, etc.).
*   **Élagueur (Pruner)** : Technique pour arrêter précocement les essais peu prometteurs (Median, Quantile, etc.).
*   **Distribution** : Définit le type et la plage de valeurs possibles pour un hyperparamètre (entier, flottant, catégoriel).
*   **Direction** : Indique si l'objectif est la minimisation ou la maximisation.

## Fonction Objective : Le Cœur de l'Optimisation

### Définir Votre Fonction Objective

*   La fonction objective est l'élément **central**. Elle prend un objet `trial` en argument.
*   Utilisez `trial.suggest_*` pour **échantillonner les hyperparamètres** à tester.
*   Effectuez l'**entraînement de votre modèle** et évaluez sa performance (par exemple, perte, exactitude).
*   Retournez la **valeur à optimiser** (minimiser ou maximiser).

**Exemple de Code:**

```python
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('num_layers', 1, 5)
    # ... (calcul de la perte)
    loss = (lr - 0.01)**2 + abs(n_layers - 3) # dummy loss
    return loss
```

## Échantillonneurs (Samplers) : Stratégies de Recherche

### Choisir le Bon Échantillonneur

*   **`RandomSampler`**: Échantillonnage aléatoire (exploration de base, utile au départ).
*   **`TPESampler` (Tree-structured Parzen Estimator)**: Algorithme plus sophistiqué, efficace pour les espaces complexes.
*   **`CmaEsSampler`**: Adaptation de la Matrice de Covariance pour les espaces continus et de haute dimension.
*   **`GridSampler`**: Recherche exhaustive dans une grille prédéfinie d'hyperparamètres (utile dans un espace de recherche restreint).

**Exemple de Code:**

```python
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
```

*   **Note:** `TPE` est un bon choix par défaut pour commencer.

## Élagueurs (Pruners) : Économiser du Temps de Calcul

### Élagueurs : Accélérer l'Optimisation

*   **`MedianPruner`**: Arrête les essais dont les résultats intermédiaires sont inférieurs à la médiane.
*   **`ThresholdPruner`**: Arrête les essais si les résultats atteignent un seuil prédéfini.
*   **`HyperbandPruner`**: Utilise l'algorithme Hyperband (efficace avec des évaluations partielles).

**Exemple de Code:**

```python
study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction='minimize')
```

## Valeurs Intermédiaires et Élagueurs

### Utilisation des Valeurs Intermédiaires

*   Les essais peuvent rapporter des **valeurs intermédiaires** pour une métrique à optimiser.
*   Utile pour arrêter l'entraînement d'un essai en cours si sa performance ne s'améliore pas.
*   Fonctionne en **collaboration avec les élagueurs**.

**Exemple de Code:**

```python
def objective(trial):
    for step in range(100):
        # Calcule la perte
        loss = (step/100 - 1)**2 # Dummy Loss
        trial.report(loss, step)
        if trial.should_prune():
             raise optuna.TrialPruned() # Prune this trial
    return loss
```

*   **Note:** `trial.should_prune` renvoie `true` seulement si un élagueur est spécifié à l'objet `étude`.

## Exécution Parallèle des Essais

### Optimisation en Parallèle

*   Optuna permet l'**exécution parallèle des essais** pour accélérer le processus.
*   **Multiprocessing**: La librairie partage les informations entre les différents processus (utilisation de `n_jobs=-1` ou un entier).
*   **Asynchrone (Base de Données)**: Les essais sont gérés à travers une base de données (SQLite, MySQL, PostgreSQL)
*   **Multithreading**: Utilisation de plusieurs threads dans le même processus (utilisation de `n_jobs`).

**Exemple de Code (multiprocessing):**

```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=-1)
```

*   **Note**: L'approche asynchrone est parfaite pour la mise en place d'une optimisation distribuée (plusieurs machines simultanément).

## Distributions d'Hyperparamètres

### Définir l'Espace de Recherche

*   `trial.suggest_int(name, low, high, step=1, log=False)`: Valeurs entières.
*   `trial.suggest_float(name, low, high, log=False)`: Valeurs flottantes.
*   `trial.suggest_categorical(name, choices)`: Choix parmi une liste de catégories.
*   `trial.suggest_uniform(name, low, high)`: Distribution uniforme dans un intervalle.
*   `trial.suggest_loguniform(name, low, high)`: Distribution log-uniforme.
*   `trial.suggest_discrete_uniform(name, low, high, q)`: Distribution uniforme discrète avec pas `q`.

## Espace de Recherche Conditionnel

### Gérer la Dépendance entre Hyperparamètres

*   Optuna permet de définir des espaces de recherche **dépendants des valeurs** d'autres hyperparamètres.
*   Utilisez les instructions `if/else` au sein de la fonction objectif pour un contrôle précis.

**Exemple de Code:**

```python
def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    if optimizer == 'Adam':
        lr = trial.suggest_float('adam_lr', 1e-5, 1e-2)
    else:
        lr = trial.suggest_float('sgd_lr', 1e-3, 1e-1)
    #...
```

##  Visualisation des Résultats

### Analyser l'Optimisation

*   Optuna offre des outils de visualisation pour **comprendre l'optimisation**.
*   `plot_optimization_history(study)` : Historique des meilleurs résultats.
*   `plot_param_importances(study)`: Importance relative des hyperparamètres.
*   `plot_slice(study)`: Visualisation de l'influence de chaque paramètre.
*   **Note:** L'installation de `plotly` est requis pour ces visualisations.

##  Conseils et Bonnes Pratiques

### Maximiser l'Efficacité d'Optuna

*   Commencez simple : random sampling puis TPE.
*   Visualisez vos résultats pour bien comprendre l'optimisation.
*   Utilisez les élagueurs pour économiser du temps.
*   Adaptez l'échantillonneur, l'élagueur et l'espace de recherche à votre problème.
*   Considérez l'exécution parallèle si vous avez des ressources.
*   Choisissez des plages de valeurs d'hyperparamètres qui ne sont pas trop grandes.
*   Enregistrez la base de donnée si vous utilisez une optimisation distribuée.
*   Re-créez facilement le meilleur essai avec `study.best_trial.params`.
*  Utiliser `seed` pour plus de reproductibilité.

##  Conclusion

### Optuna : Votre Allié pour l'Optimisation d'Hyperparamètres

*   Optuna est un outil **puissant et polyvalent**.
*   **Facile à utiliser**, mais avec des options avancées pour les utilisateurs exigeants.
*   Idéal pour améliorer les performances de vos modèles ML.
*  N'hésitez pas à explorer la [documentation d'Optuna](https://optuna.org/) pour plus de détails.

---





# Optuna Cheat Sheet (Commands Focused)

This document provides a quick reference for Optuna commands.

## Table of Contents

1.  **Installation**
2.  **Basic Concepts (Commands)**
3.  **Core Functionality (Commands)**
    *   Creating Studies
    *   Defining Objective Functions
    *   Running Optimization
    *   Accessing Trial Data
    *   Working with Parameters
    *   Pruning
    *   Storing/Loading Studies
4.  **Advanced Features (Commands)**
    *   Callbacks
    *   Multi-Objective Optimization
    *   Visualization
    *   Distributed Optimization
    *   Integration
5.  **Example Use Cases (Commands)**

## 1. Installation

```bash
pip install optuna
```

## 2. Basic Concepts (Commands)

*   **Study**: `optuna.create_study()`, `optuna.load_study()`
*   **Trial**: `trial.suggest_float()`, `trial.suggest_int()`, `trial.suggest_categorical()`, `trial.report()`, `trial.should_prune()`
*   **Objective**: Function with `trial` as input, returns single/tuple value.
*   **Sampler**: `optuna.samplers.RandomSampler()`, `optuna.samplers.TPESampler()`, `optuna.samplers.CmaEsSampler()`
*   **Pruner**: `optuna.pruners.MedianPruner()`, `optuna.pruners.PercentilePruner()`, `optuna.pruners.HyperbandPruner()`

## 3. Core Functionality (Commands)

### Creating Studies

```python
import optuna

study = optuna.create_study() # Minimize by default
study_max = optuna.create_study(direction="maximize")
study_db = optuna.create_study(storage="sqlite:///example.db")
study_sampler = optuna.create_study(sampler=optuna.samplers.TPESampler())
```

### Defining Objective Functions

```python
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", 1, 10)
    z = trial.suggest_categorical("z", ["a", "b", "c"])
    return x**2 + y  # Example
```

### Running Optimization

```python
study.optimize(objective, n_trials=10)
study.optimize(objective, timeout=60) # Seconds
```

### Accessing Trial Data

```python
study.best_trial # Best trial object
study.best_value # Best objective value
study.best_params # Best parameter dict
study.trials  # List of all trial objects
trial.number, trial.value, trial.params  # Access info of a single trial
trial.state # Trial state (RUNNING, COMPLETE, PRUNED, FAIL)
```

### Working with Parameters

```python
trial.suggest_float("float_param", -5.0, 5.0)
trial.suggest_int("int_param", 1, 10)
trial.suggest_float("log_float", 1e-5, 1, log=True)
trial.suggest_int("log_int", 1, 100, log=True)
trial.suggest_categorical("cat_param", ["a", "b", "c"])
trial.suggest_float("disc_param", -10, 10, step=1)
```

### Pruning

```python
trial.report(value, step)
if trial.should_prune():
   raise optuna.exceptions.TrialPruned()
study_pruner = optuna.create_study(pruner=optuna.pruners.MedianPruner())
```

### Storing/Loading Studies

```python
study_db = optuna.create_study(storage="sqlite:///example.db", study_name="my_study")
loaded_study = optuna.load_study(storage="sqlite:///example.db", study_name="my_study")
```

## 4. Advanced Features (Commands)

### Callbacks

```python
def callback(study, trial):
  print(f"Trial {trial.number} finished")
study.optimize(objective, n_trials=10, callbacks=[callback])
```

### Multi-Objective Optimization

```python
study_multi = optuna.create_study(directions=["minimize", "maximize"])
def multi_objective(trial):
  return trial.suggest_float('x',-10,10)**2, -trial.suggest_float('y',-10,10)
study_multi.optimize(multi_objective, n_trials=10)
study_multi.best_trials # Pareto front trials
```

### Visualization

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_contour
)
plot_optimization_history(study).show()
plot_parallel_coordinate(study).show()
plot_param_importances(study).show()
plot_slice(study).show()
plot_contour(study, params=["x","y"]).show() # If those params exist
```

### Distributed Optimization

*   Same code, use database storage in `create_study`.

### Integration

*   Use Optuna to tune parameters of other libraries (e.g., scikit-learn, PyTorch).

## 5. Example Use Cases (Commands)

```python
# Hyperparameter Tuning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def objective(trial):
  n_estimators = trial.suggest_int("n_estimators", 100, 500)
  max_depth = trial.suggest_int("max_depth", 5, 15)
  model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  return accuracy_score(y_test, y_pred)

```
```python
# simulation optimization
import random
def simulate(param1, param2):
  return param1 + param2 + random.random() # simulates a function
def objective_simulation(trial):
  param1 = trial.suggest_float("param1", -10,10)
  param2 = trial.suggest_float("param2", -10,10)
  return simulate(param1, param2)
```
```python
# Model Selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "LogisticRegression", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 0.1, 10, log=True)
        model = SVC(C=svc_c)
    elif classifier_name == "LogisticRegression":
        lr_c = trial.suggest_float("lr_c", 0.1, 10, log=True)
        model = LogisticRegression(C=lr_c)
    elif classifier_name == "RandomForest":
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 500)
        model = RandomForestClassifier(n_estimators=rf_n_estimators)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
```

# Exemple Final

```python
import optuna
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # Exemple de jeu de données
import logging

# Configuration du logging pour mieux comprendre le déroulement de l'optimisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def objective(trial):
    """
    Fonction objective pour l'optimisation des hyperparamètres des classifieurs.

    Args:
        trial (optuna.Trial): Objet Optuna représentant un essai.

    Returns:
        float: Score d'exactitude du classifieur sur l'ensemble de test.
    """
    logging.info(f"Début de l'essai : {trial.number}")

    # Choix du classifieur
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "LogisticRegression", "RandomForest"])
    logging.info(f"Classifieur choisi: {classifier_name}")

    # Paramétrage du classifieur
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 0.1, 10, log=True)
        model = SVC(C=svc_c, random_state=42) # Ajout du random_state pour la reproductibilité
        logging.info(f"  - Paramètres SVC: C = {svc_c}")

    elif classifier_name == "LogisticRegression":
        lr_c = trial.suggest_float("lr_c", 0.1, 10, log=True)
        model = LogisticRegression(C=lr_c, random_state=42, solver='liblinear') # Ajout du random_state et du solver
        logging.info(f"  - Paramètres LogisticRegression: C = {lr_c}")

    elif classifier_name == "RandomForest":
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 500)
        model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42) # Ajout du random_state
        logging.info(f"  - Paramètres RandomForest: n_estimators = {rf_n_estimators}")


    # Chargement et séparation des données (Iris dataset pour l'exemple)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random_state ajouté pour reproductibilité

    # Entraînement du modèle
    model.fit(X_train, y_train)
    logging.info(f"  - Modèle entraîné")

    # Prédiction et évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"  - Score d'exactitude: {accuracy:.4f}")
    logging.info(f"Fin de l'essai: {trial.number}")
    return accuracy


if __name__ == "__main__":
    # Création de l'étude Optuna
    study = optuna.create_study(direction="maximize")  # On maximise l'accuracy

    # Lancement de l'optimisation
    n_trials = 50 # Nombre d'essais
    logging.info(f"Lancement de l'optimisation avec {n_trials} essais...")
    study.optimize(objective, n_trials=n_trials)

    # Affichage des meilleurs résultats
    best_trial = study.best_trial
    print("\nMeilleur essai:")
    print(f"  Score d'exactitude: {best_trial.value:.4f}")
    print("  Paramètres:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    logging.info("Optimisation terminée.")
```
