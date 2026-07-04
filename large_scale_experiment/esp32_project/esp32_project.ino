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
        float z = 0.0f;
        for (int i = 0; i < 576; i++) {
            float weight = (float)FC_WEIGHTS_QUANT[c * 576 + i] * FC_SCALE;
            z += weight * features[i];
        }
        
        float weight_base = ACT3_W_BASE[c];
        float out = weight_base * z;
        
        for (int k = 0; k < 2; k++) {
            float a = ACT3_A[c * 2 + k];
            float b = ACT3_B[c * 2 + k];
            float c_param = ACT3_C[c * 2 + k];
            float d = ACT3_D[c * 2 + k];
            float w_eml = ACT3_W_EML[c * 2 + k];
            
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
    Serial.println("ESP32 EML-KAN Serial HIL Tester Ready!");
}

void loop() {
    // Check if sync command "SYNC" is received
    if (Serial.find("SYNC")) {
        static float features[576]; // allocated statically to prevent stack overflow
        int expected_bytes = 2304; // 576 * sizeof(float)
        
        // Read the binary feature array directly from Serial buffer in one go
        Serial.setTimeout(3000); // Set 3 seconds timeout
        int bytes_read = Serial.readBytes((char*)features, expected_bytes);
        
        if (bytes_read == expected_bytes) {
            float logits[100];
            
            // Measure dequantized classification duration
            unsigned long start_time = micros();
            evaluate_eml_kan_classifier(features, logits);
            unsigned long duration = micros() - start_time;
            
            // Find predicted class index (highest logit)
            int max_class = 0;
            float max_logit = logits[0];
            for (int c = 1; c < 100; c++) {
                if (logits[c] > max_logit) {
                    max_logit = logits[c];
                    max_class = c;
                }
            }
            
            // Send debug features and prediction back to the Python script
            Serial.print("FEAT: ");
            Serial.print(features[0], 6);
            Serial.print(" ");
            Serial.print(features[1], 6);
            Serial.print(" PRED: ");
            Serial.print(max_class);
            Serial.print(" TIME: ");
            Serial.println(duration);
        } else {
            // Error handling for timeout or partial read
            Serial.print("FEAT: 0.0 0.0 PRED: -1 TIME: 0 (TIMEOUT, read ");
            Serial.print(bytes_read);
            Serial.println(" bytes)");
        }
    }
}
