#include <stdio.h>
#include <math.h>

float prediction(float *features, int n_features) {
    float result = -59725.915009571065f;  // Intercept
    
    // Coefficients
    float coefficients[3] = {
        803.9336224105622f,
        48348.08126447237f,
        109706.77017703892f
    };
    
    for (int i = 0; i < n_features; i++) {
        result += coefficients[i] * features[i];
    }
    
    return result;
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
    printf("Prediction: %.2f\n", pred);
    
    return 0;
}
