#include <stdio.h>
#include <math.h>

/**
 * Approximation de la fonction exponentielle
 */
float exp_approx(float x, int n_term) {
    float sum = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i <= n_term; i++) {
        term = term * x / i;
        sum += term;
    }
    
    return sum;
}


float sigmoid(float x) {
    return 1.0f / (1.0f + exp_approx(-x, 10));
}


float prediction(float *features, int n_features) {
    float z = -3.4422886235720287f;  // Intercept
    
    // Coefficients
    float coefficients[3] = {
        0.00661474771272227f,
        0.3967224252872799f,
        2.238912462859416f
    };
    
    for (int i = 0; i < n_features; i++) {
        z += coefficients[i] * features[i];
    }
    
    return sigmoid(z);
}

int main() {
    // Exemple de données de test
    float test_features[] = {200.0f, 3.0f, 1.0f};
    int n_features = 3;
    
    float pred = prediction(test_features, n_features);
    
    printf("Features: [");
    for (int i = 0; i < n_features; i++) {
        printf("%.2f", test_features[i]);
        if (i < n_features - 1) printf(", ");
    }
    printf("]\n");
    printf("Prediction (probability): %.6f\n", pred);
    printf("Class prediction: %d\n", pred >= 0.5 ? 1 : 0);
    
    return 0;
}
