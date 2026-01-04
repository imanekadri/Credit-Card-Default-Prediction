#  Projet 4 – Prédiction de Défaut de Paiement  
### Classification binaire déséquilibrée et massive

---

##  Description du projet

Ce projet vise à prédire si un client va **faire défaut sur le paiement de sa carte de crédit**.  
Le problème est formulé comme une **classification binaire fortement déséquilibrée**, où les clients défaillants sont minoritaires.

L’objectif est de fournir un **outil fiable et rapide d’aide à la décision bancaire**, permettant de réduire les risques liés à l’octroi de crédit.

---

##  Jeu de données

- Dataset : **UCI Credit Card Dataset**
- Variable cible :
  - `default.payment.next.month`
    - 0 : client non défaillant
    - 1 : client défaillant
- Déséquilibre des classes :
  - ≈ 77.9 % non défaut
  - ≈ 22.1 % défaut

---

##  Préparation des données & EDA

- Chargement des données avec **Pandas**
- Vérification de la qualité :
  - Pas de valeurs manquantes significatives
  - Suppression des doublons
- Analyse descriptive :
  - Moyenne, min, max, écart-type
- Analyse de la variable cible :
  - Dataset fortement déséquilibré
- Observations principales :
  - Population majoritairement âgée de 30–40 ans
  - Clients majoritairement mariés
  - Hommes légèrement plus nombreux que femmes
  - La majorité des clients paient leurs factures à temps

---

##  Feature Engineering

Création de nouvelles variables afin d’améliorer la capacité prédictive des modèles :

### 🔹 Agrégations
- `TOTAL_BILL` : somme des montants facturés
- `TOTAL_PAY` : somme des montants payés

### 🔹 Tendances temporelles
- `BILL_TREND = BILL_AMT6 - BILL_AMT1`
- `PAY_TREND = PAY_AMT6 - PAY_AMT1`

### 🔹 Indicateurs comportementaux
- `PAY_RATIO = TOTAL_PAY / (TOTAL_BILL + 1)`
- `NB_LATE_PAYMENTS` : nombre de retards de paiement
- `BILL_STD` : volatilité des montants facturés

Ces nouvelles features permettent une **meilleure séparation entre clients défaillants et non défaillants**.

---

##  Modèles utilisés

### 1️⃣ Gradient Boosting Trees (GBDT)
- **LightGBM**
- **XGBoost**
- **CatBoost**

Avantages :
- Très performants sur données tabulaires
- Rapides et robustes
- Bonne gestion du déséquilibre des classes

### 2️⃣ Réseau de Neurones Artificiels (ANN / MLP)
- Architecture simple (Dense layers)
- Capacité à capturer des relations non linéaires
- Données standardisées (mean = 0, std = 1)

---

##  Stacking multi-niveaux

### 🔹 Niveau 1
- Entraînement de plusieurs modèles :
  - LightGBM
  - XGBoost
  - CatBoost
  - Neural Network (MLP)
- Les probabilités prédites sont utilisées comme nouvelles features

### 🔹 Niveau 2 (Meta-model)
- Modèle : **Ridge Regression**
- Objectif :
  - Combiner les forces des modèles de base
  - Améliorer la généralisation
  - Réduire le biais et la variance

---

##  Évaluation

- Métriques utilisées :
  - ROC-AUC (principale)
  - Precision, Recall, F1-score
  - Confusion Matrix
- Performance :
  - ROC-AUC ≈ 0.77 – 0.80
- Bon compromis entre détection des clients à risque et limitation des faux positifs

---

##  Conclusion

Ce projet démontre l’efficacité d’une approche **hybride combinant Gradient Boosting Trees et Réseaux de Neurones**, renforcée par un **feature engineering avancé**.

Le modèle final fournit une **prédiction fiable du risque de défaut**, utile pour les institutions financières dans leurs décisions d’octroi de crédit.

---

##  Perspectives

- Ajustement du seuil de décision selon le coût métier
- Utilisation de TabNet ou modèles deep learning plus avancés
