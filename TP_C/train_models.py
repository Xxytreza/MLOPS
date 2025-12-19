#!/usr/bin/env python3
"""
Script d'entraînement de modèles sur le dataset houses.csv
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np


def train_linear_regression():
    """
    Entraîne une régression linéaire pour prédire le prix des maisons
    """
    print("=" * 60)
    print("Entraînement d'une régression linéaire")
    print("=" * 60)
    
    # Charger les données
    df = pd.read_csv('houses.csv')
    print(f"\nDataset chargé: {len(df)} lignes")
    print(f"Colonnes: {list(df.columns)}")
    
    # Features et target
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    
    print(f"\nFeatures utilisées: {list(X.columns)}")
    print(f"Target: price")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Évaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nPerformances:")
    print(f"  R² train: {train_score:.4f}")
    print(f"  R² test:  {test_score:.4f}")
    
    print(f"\nCoefficients du modèle:")
    print(f"  Intercept: {model.intercept_:.2f}")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.2f}")
    
    # Sauvegarder
    joblib.dump(model, "regression.joblib")
    print(f"\nModèle sauvegardé dans: regression.joblib")
    
    # Test de prédiction
    test_features = np.array([[200.0, 3.0, 1.0]])
    prediction = model.predict(test_features)
    print(f"\nTest de prédiction:")
    print(f"  Features: {test_features[0]}")
    print(f"  Prédiction: {prediction[0]:.2f}")
    
    return model


def train_logistic_regression():
    """
    Entraîne une régression logistique pour classifier les maisons chères/pas chères
    """
    print("\n" + "=" * 60)
    print("Entraînement d'une régression logistique")
    print("=" * 60)
    
    # Charger les données
    df = pd.read_csv('houses.csv')
    print(f"\nDataset chargé: {len(df)} lignes")
    
    # Features
    X = df[['size', 'nb_rooms', 'garden']]
    
    # Créer une target binaire: maison chère si prix > médiane
    price_median = df['price'].median()
    y = (df['price'] > price_median).astype(int)
    
    print(f"\nFeatures utilisées: {list(X.columns)}")
    print(f"Target: maison_chere (prix > {price_median:.2f})")
    print(f"  Classe 0 (pas chère): {(y==0).sum()} exemples")
    print(f"  Classe 1 (chère):     {(y==1).sum()} exemples")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Évaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nPerformances:")
    print(f"  Accuracy train: {train_score:.4f}")
    print(f"  Accuracy test:  {test_score:.4f}")
    
    print(f"\nCoefficients du modèle:")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
    
    # Sauvegarder
    joblib.dump(model, "logistic_regression.joblib")
    print(f"\nModèle sauvegardé dans: logistic_regression.joblib")
    
    # Test de prédiction
    test_features = np.array([[200.0, 3.0, 1.0]])
    prediction_proba = model.predict_proba(test_features)[0]
    prediction = model.predict(test_features)[0]
    
    print(f"\nTest de prédiction:")
    print(f"  Features: {test_features[0]}")
    print(f"  Probabilité classe 0: {prediction_proba[0]:.4f}")
    print(f"  Probabilité classe 1: {prediction_proba[1]:.4f}")
    print(f"  Classe prédite: {prediction}")
    
    return model


if __name__ == "__main__":
    # Entraîner les deux modèles
    linear_model = train_linear_regression()
    logistic_model = train_logistic_regression()
    
    print("\n" + "=" * 60)
    print("Entraînement terminé!")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - regression.joblib (régression linéaire)")
    print("  - logistic_regression.joblib (régression logistique)")
    print("\nPour transpiler:")
    print("  python transpile_simple_model.py regression.joblib --compile")
    print("  python transpile_simple_model.py logistic_regression.joblib --logistic --compile")
