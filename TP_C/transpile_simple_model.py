import joblib
import numpy as np
import os
import sys


def transpile_linear_regression(model, output_file="model.c"):
    coefficients = model.coef_
    intercept = model.intercept_
    n_features = len(coefficients)
    
    # Génération du code C
    c_code = f"""#include <stdio.h>
#include <math.h>

float prediction(float *features, int n_features) {{
    float result = {intercept}f;  // Intercept
    
    // Coefficients
    float coefficients[{n_features}] = {{
"""
    
    # Ajouter les coefficients
    for i, coef in enumerate(coefficients):
        c_code += f"        {coef}f"
        if i < n_features - 1:
            c_code += ",\n"
        else:
            c_code += "\n"
    
    c_code += """    };
    
    for (int i = 0; i < n_features; i++) {
        result += coefficients[i] * features[i];
    }
    
    return result;
}

int main() {
    // Exemple de données de test
    float test_features[] = {200.0f, 3.0f, 1.0f};
    int n_features = """ + str(n_features) + """;
    
    float pred = prediction(test_features, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%.2f", test_features[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
    printf("Prediction: %.2f\\n", pred);
    
    return 0;
}
"""
    
    # Sauvegarder le fichier
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f"Code C généré dans: {output_file}")
    
    # Commande de compilation
    output_executable = output_file.replace('.c', '')
    compile_cmd = f"gcc {output_file} -o {output_executable} -lm"
    print(f" Commande de compilation: {compile_cmd}")
    
    return compile_cmd


def transpile_logistic_regression(model, output_file="model_logistic.c"):
    coefficients = model.coef_[0]  # LogisticRegression stocke dans un array 2D
    intercept = model.intercept_[0]
    n_features = len(coefficients)
    
    # Génération du code C avec sigmoid
    c_code = f"""#include <stdio.h>
#include <math.h>

/**
 * Approximation de la fonction exponentielle
 */
float exp_approx(float x, int n_term) {{
    float sum = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i <= n_term; i++) {{
        term = term * x / i;
        sum += term;
    }}
    
    return sum;
}}


float sigmoid(float x) {{
    return 1.0f / (1.0f + exp_approx(-x, 10));
}}


float prediction(float *features, int n_features) {{
    float z = {intercept}f;  // Intercept
    
    // Coefficients
    float coefficients[{n_features}] = {{
"""
    
    for i, coef in enumerate(coefficients):
        c_code += f"        {coef}f"
        if i < n_features - 1:
            c_code += ",\n"
        else:
            c_code += "\n"
    
    c_code += """    };
    
    for (int i = 0; i < n_features; i++) {
        z += coefficients[i] * features[i];
    }
    
    return sigmoid(z);
}

int main() {
    // Exemple de données de test
    float test_features[] = {200.0f, 3.0f, 1.0f};
    int n_features = """ + str(n_features) + """;
    
    float pred = prediction(test_features, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%.2f", test_features[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
    printf("Prediction (probability): %.6f\\n", pred);
    printf("Class prediction: %d\\n", pred >= 0.5 ? 1 : 0);
    
    return 0;
}
"""
    
    # Sauvegarder le fichier
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f" Code C généré dans: {output_file}")
    
    # Commande de compilation
    output_executable = output_file.replace('.c', '')
    compile_cmd = f"gcc {output_file} -o {output_executable} -lm"
    print(f" Commande de compilation: {compile_cmd}")
    
    return compile_cmd


def transpile_decision_tree(model, output_file="model_tree.c"):
    tree = model.tree_
    n_features = tree.n_features
    
    # Fonction pour générer récursivement le code de l'arbre
    def generate_tree_code(node_id, indent="    "):
        # Si c'est une feuille
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Pour la classification, prendre la classe majoritaire
            if hasattr(model, 'classes_'):
                value = tree.value[node_id][0]
                predicted_class = int(np.argmax(value))
                return f"{indent}return {predicted_class};\n"
            else:
                # Pour la régression, retourner la valeur moyenne
                value = tree.value[node_id][0][0]
                return f"{indent}return {value}f;\n"
        
        # Sinon, c'est un nœud de décision
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        code = f"{indent}if (features[{feature_idx}] <= {threshold}f) {{\n"
        code += generate_tree_code(left_child, indent + "    ")
        code += f"{indent}}} else {{\n"
        code += generate_tree_code(right_child, indent + "    ")
        code += f"{indent}}}\n"
        
        return code
    
    # Déterminer le type de retour
    if hasattr(model, 'classes_'):
        return_type = "int"
        function_name = "predict_class"
    else:
        return_type = "float"
        function_name = "predict"
    
    # Génération du code C
    c_code = f"""#include <stdio.h>

{return_type} {function_name}(float *features, int n_features) {{
"""
    
    # Générer le code de l'arbre
    c_code += generate_tree_code(0)
    
    c_code += """}

int main() {
    // Exemple de données de test
    float test_features[] = {1.0f, -1.0f};
    int n_features = """ + str(n_features) + """;
    
    """ + return_type + """ pred = """ + function_name + """(test_features, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%.2f", test_features[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\\n");
"""
    
    if hasattr(model, 'classes_'):
        c_code += """    printf("Predicted class: %d\\n", pred);
"""
    else:
        c_code += """    printf("Prediction: %.6f\\n", pred);
"""
    
    c_code += """    
    return 0;
}
"""
    
    # Sauvegarder le fichier
    with open(output_file, 'w') as f:
        f.write(c_code)
    
    print(f" Code C généré dans: {output_file}")
    
    # Commande de compilation
    output_executable = output_file.replace('.c', '')
    compile_cmd = f"gcc {output_file} -o {output_executable} -lm"
    print(f" Commande de compilation: {compile_cmd}")
    
    return compile_cmd


def main():
    if len(sys.argv) < 2:
        print("Usage: python transpile_simple_model.py <model.joblib> [--logistic|--tree] [--compile]")
        print("  --logistic : transpiler une régression logistique au lieu de linéaire")
        print("  --tree     : transpiler un arbre de décision")
        print("  --compile  : compiler automatiquement le fichier C généré")
        sys.exit(1)
    
    model_path = sys.argv[1]
    is_logistic = '--logistic' in sys.argv
    is_tree = '--tree' in sys.argv
    auto_compile = '--compile' in sys.argv
    
    # Charger le modèle
    print(f"Chargement du modèle: {model_path}")
    model = joblib.load(model_path)
    
    # Déterminer le type de modèle
    model_type = type(model).__name__
    print(f"Type de modèle détecté: {model_type}")
    
    # Transpiler selon le type
    if is_tree or 'Tree' in model_type or 'DecisionTree' in model_type:
        output_file = "model_tree.c"
        compile_cmd = transpile_decision_tree(model, output_file)
    elif is_logistic or 'Logistic' in model_type:
        output_file = "model_logistic.c"
        compile_cmd = transpile_logistic_regression(model, output_file)
    else:
        output_file = "model.c"
        compile_cmd = transpile_linear_regression(model, output_file)
    
    # Compiler si demandé
    if auto_compile:
        print(f"\nCompilation en cours...")
        exit_code = os.system(compile_cmd)
        if exit_code == 0:
            print(" Compilation réussie!")
            executable = output_file.replace('.c', '')
            print(f"\nPour exécuter: ./{executable}")
        else:
            print("Erreur de compilation")
            sys.exit(1)
    else:
        print(f"\nPour compiler: {compile_cmd}")
        print(f"Pour exécuter: ./{output_file.replace('.c', '')}")


if __name__ == "__main__":
    main()
