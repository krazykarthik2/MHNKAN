#include <Arduino.h>
#include <math.h>
#include "esp32_cifar100_inference.h"

// Softplus activation helper
inline float softplus_val(float z) {
    return logf(1.0f + expf(z));
}

/**
 * Evaluates the trained EML-KAN Classifier on a given 576-element feature vector.
 * Reconstructs weights dynamically on-the-fly using Int8 de-quantization.
 */
void evaluate_eml_kan_classifier(const float* features, float* output_logits) {
    // There are 100 classes
    for (int c = 0; c < 100; c++) {
        // 1. Compute de-quantized linear dot product: z = Sum(W_quant * scale * x)
        float z = 0.0f;
        for (int i = 0; i < 576; i++) {
            float weight = (float)pgm_read_byte(&FC_WEIGHTS_QUANT[c * 576 + i]) * FC_SCALE;
            z += weight * features[i];
        }
        
        // 2. Apply EML-KAN Activation Function with K=2 components
        float weight_base = pgm_read_float(&ACT3_W_BASE[c]);
        float out = weight_base * z;
        
        for (int k = 0; k < 2; k++) {
            float a = pgm_read_float(&ACT3_A[c * 2 + k]);
            float b = pgm_read_float(&ACT3_B[c * 2 + k]);
            float c_param = pgm_read_float(&ACT3_C[c * 2 + k]);
            float d = pgm_read_float(&ACT3_D[c * 2 + k]);
            float w_eml = pgm_read_float(&ACT3_W_EML[c * 2 + k]);
            
            // Stable clamped exponential and softplus evaluation
            float arg_x = a * z + b;
            if (arg_x < -10.0f) arg_x = -10.0f;
            if (arg_x > 10.0f) arg_x = 10.0f;
            
            float arg_y = softplus_val(c_param * z + d) + 1e-6f;
            out += w_eml * (expf(arg_x) - logf(arg_y));
        }
        
        output_logits[c] = out;
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 EML-KAN Quantized Classifier Benchmark Start!");
}

void loop() {
    // Generate dummy extracted feature vector (576 elements) representing a test image
    float dummy_features[576];
    for (int i = 0; i < 576; i++) {
        dummy_features[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // range [-1.0, 1.0]
    }
    
    float logits[100];
    
    // Benchmark latency
    unsigned long start_time = micros();
    evaluate_eml_kan_classifier(dummy_features, logits);
    unsigned long duration = micros() - start_time;
    
    // Find the predicted class (highest logit)
    int max_class = 0;
    float max_logit = logits[0];
    for (int c = 1; c < 100; c++) {
        if (logits[c] > max_logit) {
            max_logit = logits[c];
            max_class = c;
        }
    }
    
    Serial.print("Predicted Class: ");
    Serial.print(max_class);
    Serial.print(" | Max Logit: ");
    Serial.print(max_logit, 4);
    Serial.print(" | Execution Latency: ");
    Serial.print(duration);
    Serial.println(" us");
    
    delay(1000);
}
