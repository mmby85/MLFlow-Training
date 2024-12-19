# Tutoriel sur les commandes kubectl courantes

Ce tutoriel vous guidera à travers les commandes `kubectl` les plus utilisées pour interagir avec votre cluster Kubernetes. `kubectl` est l'outil en ligne de commande principal pour exécuter des commandes sur les clusters Kubernetes.

## Prérequis

*   Un cluster Kubernetes en fonctionnement.
*   `kubectl` installé et configuré pour se connecter à votre cluster.

## Commandes de Base

### 1. Afficher l'état du cluster

```bash
kubectl cluster-info
```

Cette commande affiche des informations sur le point de terminaison du cluster et les services Kubernetes en cours d'exécution.

### 2. Afficher les nœuds

```bash
kubectl get nodes
```

Affiche la liste de tous les nœuds du cluster, ainsi que leur état.

```bash
kubectl get nodes -o wide
```
Affiche des informations supplémentaires sur les nœuds, comme leurs adresses IP et leurs versions.

### 3. Afficher les pods

```bash
kubectl get pods
```

Affiche la liste de tous les pods (unités d'exécution d'applications) dans le namespace actuel.

```bash
kubectl get pods -A
```

Affiche la liste de tous les pods dans tous les namespaces. `-A` est un raccourci pour `--all-namespaces`.

```bash
kubectl get pods -n <namespace>
```

Affiche la liste des pods dans un namespace spécifique. Remplacez `<namespace>` par le nom du namespace.

```bash
kubectl get pods -o wide
```

Affiche des informations supplémentaires sur les pods, comme leurs nœuds d'exécution et leurs adresses IP.

### 4. Afficher les déploiements

```bash
kubectl get deployments
```

Affiche la liste des déploiements dans le namespace actuel.

```bash
kubectl get deployments -A
```

Affiche la liste des déploiements dans tous les namespaces.

```bash
kubectl get deployments -n <namespace>
```

Affiche la liste des déploiements dans un namespace spécifique.

### 5. Afficher les services

```bash
kubectl get services
```

Affiche la liste des services dans le namespace actuel.

```bash
kubectl get services -A
```

Affiche la liste des services dans tous les namespaces.

```bash
kubectl get services -n <namespace>
```

Affiche la liste des services dans un namespace spécifique.

## Commandes pour Manipuler les Ressources

### 6. Créer une ressource

```bash
kubectl create -f <file.yaml>
```

Crée une ressource (pod, déploiement, service, etc.) à partir d'un fichier YAML. Remplacez `<file.yaml>` par le nom de votre fichier de configuration.

### 7. Appliquer une ressource

```bash
kubectl apply -f <file.yaml>
```

Applique ou met à jour une ressource à partir d'un fichier YAML. Cette commande est recommandée pour les mises à jour. Remplacez `<file.yaml>` par le nom de votre fichier de configuration.

### 8. Supprimer une ressource

```bash
kubectl delete -f <file.yaml>
```

Supprime une ressource définie dans un fichier YAML. Remplacez `<file.yaml>` par le nom de votre fichier de configuration.

```bash
kubectl delete pod <pod_name>
```

Supprime un pod spécifique. Remplacez `<pod_name>` par le nom du pod.

```bash
kubectl delete deployment <deployment_name>
```

Supprime un déploiement spécifique. Remplacez `<deployment_name>` par le nom du déploiement.

### 9. Décrire une ressource

```bash
kubectl describe pod <pod_name>
```

Affiche des informations détaillées sur un pod spécifique. Remplacez `<pod_name>` par le nom du pod.

```bash
kubectl describe service <service_name>
```

Affiche des informations détaillées sur un service spécifique. Remplacez `<service_name>` par le nom du service.

### 10. Afficher les logs d'un pod

```bash
kubectl logs <pod_name>
```

Affiche les logs d'un pod spécifique. Remplacez `<pod_name>` par le nom du pod.

```bash
kubectl logs <pod_name> -f
```

Affiche les logs d'un pod spécifique et suit les nouvelles entrées (comme `tail -f`).

```bash
kubectl logs <pod_name> -c <container_name>
```

Affiche les logs d'un container spécifique dans un pod. Remplacez `<pod_name>` par le nom du pod et `<container_name>` par le nom du container.

## Commandes Utiles

### 11. Exécuter une commande dans un pod

```bash
kubectl exec -it <pod_name> -- /bin/bash
```

Ouvre un shell interactif dans un pod spécifique. Remplacez `<pod_name>` par le nom du pod.

```bash
kubectl exec -it <pod_name> -c <container_name> -- /bin/bash
```

Ouvre un shell interactif dans un container spécifique d'un pod.

```bash
kubectl exec <pod_name> -- <command>
```

Exécute une commande spécifique dans un pod. Remplacez `<pod_name>` par le nom du pod et `<command>` par la commande à exécuter.

### 12. Redimensionner un déploiement

```bash
kubectl scale deployment <deployment_name> --replicas=<count>
```

Redimensionne un déploiement pour avoir un nombre spécifique de réplicas. Remplacez `<deployment_name>` par le nom du déploiement et `<count>` par le nombre de réplicas.

## Commandes Avancées (Pour information)

### 13. Port forwarding

```bash
kubectl port-forward service/<service_name> <local_port>:<service_port>
```

Crée un tunnel entre votre machine locale et un port de service dans votre cluster. Remplacez `<service_name>` par le nom du service, `<local_port>` par le port sur votre machine locale et `<service_port>` par le port du service.

### 14. Gestion des Namespaces

```bash
kubectl create namespace <namespace_name>
```

Crée un nouveau namespace.

```bash
kubectl config set-context --current --namespace=<namespace_name>
```

Définit le namespace actuel pour les commandes kubectl.

```bash
kubectl get namespaces
```

Affiche la liste de tous les namespaces disponibles.

## Conclusion

Ce tutoriel couvre les commandes `kubectl` les plus courantes pour la gestion de votre cluster Kubernetes. Avec ces commandes, vous pouvez manipuler les pods, les déploiements, les services et d'autres ressources. Pour une exploration plus approfondie, consultez la documentation officielle de Kubernetes et de `kubectl`.
