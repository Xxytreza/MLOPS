#!/usr/bin/env python3
"""
Script de vérification : compare les prédictions Python et C
"""

import joblib
import numpy as np
import subprocess
import os


def verify_linear_regression():
    """
    Vérifie que le modèle C produit les mêmes prédictions que Python
    """
    print("=" * 60)
    print("Vérification de la régression linéaire")
    print("=" * 60)
    
    # Charger le modèle Python
    model = joblib.load("regression.joblib")
    
    # Données de test
    test_features = np.array([[200.0, 3.0, 1.0]])
    
    # Prédiction Python
    python_pred = model.predict(test_features)[0]
    print(f"\nPrédiction Python: {python_pred:.2f}")
    
    # Exécuter le programme C
    if os.path.exists("./model"):
        result = subprocess.run(["./model"], capture_output=True, text=True)
        print(f"\nSortie du programme C:")
        print(result.stdout)
        
        # Extraire la prédiction C (dernière ligne)
        for line in result.stdout.strip().split('\n'):
            if "Prediction:" in line:
                c_pred = float(line.split(":")[-1].strip())
                print(f"\nPrédiction C: {c_pred:.2f}")
                
                # Comparaison
                diff = abs(python_pred - c_pred)
                print(f"\nDifférence: {diff:.6f}")
                
                if diff < 0.01:
                    print("Les prédictions sont identiques!")
                else:
                    print("✗ Attention: différence significative détectée")
    else:
        print("\n✗ Programme C non trouvé. Compilez d'abord avec:")
        print("  gcc model.c -o model -lm")


def verify_logistic_regression():
    """
    Vérifie que le modèle logistique C produit les mêmes prédictions que Python
    """
    print("\n" + "=" * 60)
    print("Vérification de la régression logistique")
    print("=" * 60)
    
    # Charger le modèle Python
    model = joblib.load("logistic_regression.joblib")
    
    # Données de test
    test_features = np.array([[200.0, 3.0, 1.0]])
    
    # Prédiction Python
    python_proba = model.predict_proba(test_features)[0][1]
    python_class = model.predict(test_features)[0]
    print(f"\nPrédiction Python:")
    print(f"  Probabilité: {python_proba:.6f}")
    print(f"  Classe: {python_class}")
    
    # Exécuter le programme C
    if os.path.exists("./model_logistic"):
        result = subprocess.run(["./model_logistic"], capture_output=True, text=True)
        print(f"\nSortie du programme C:")
        print(result.stdout)
        
        # Extraire la prédiction C
        for line in result.stdout.strip().split('\n'):
            if "Prediction (probability):" in line:
                c_proba = float(line.split(":")[-1].strip())
                print(f"\nPrédiction C:")
                print(f"  Probabilité: {c_proba:.6f}")
                
                # Comparaison
                diff = abs(python_proba - c_proba)
                print(f"\nDifférence: {diff:.6f}")
                
                if diff < 0.001:
                    print("Les prédictions sont identiques!")
                else:
                    print(f"⚠ Différence acceptable (approximation de exp)")
    else:
        print("\n✗ Programme C non trouvé. Compilez d'abord avec:")
        print("  gcc model_logistic.c -o model_logistic -lm")


if __name__ == "__main__":
    verify_linear_regression()
    verify_logistic_regression()
    
    print("\n" + "=" * 60)
    print("Vérification terminée")
    print("=" * 60)
