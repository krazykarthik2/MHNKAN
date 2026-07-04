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
            float weight = (float)((int8_t)pgm_read_byte(&FC_WEIGHTS_QUANT[c * 576 + i])) * FC_SCALE;
            z += weight * features[i];
        }
        
        float weight_base = pgm_read_float(&ACT3_W_BASE[c]);
        float out = weight_base * z;
        
        for (int k = 0; k < 2; k++) {
            float a = pgm_read_float(&ACT3_A[c * 2 + k]);
            float b = pgm_read_float(&ACT3_B[c * 2 + k]);
            float c_param = pgm_read_float(&ACT3_C[c * 2 + k]);
            float d = pgm_read_float(&ACT3_D[c * 2 + k]);
            float w_eml = pgm_read_float(&ACT3_W_EML[c * 2 + k]);
            
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
        float features[576];
        byte* feature_bytes = (byte*)features;
        int expected_bytes = 2304; // 576 * sizeof(float)
        
        // Wait and read the binary feature array from Serial
        int bytes_read = 0;
        unsigned long timeout_start = millis();
        while (bytes_read < expected_bytes && (millis() - timeout_start) < 3000) {
            if (Serial.available() > 0) {
                feature_bytes[bytes_read] = Serial.read();
                bytes_read++;
            }
        }
        
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
            
            // Send prediction and latency back to the Python script
            Serial.print("PRED: ");
            Serial.print(max_class);
            Serial.print(" TIME: ");
            Serial.println(duration);
        } else {
            // Error handling for timeout
            Serial.println("PRED: -1 TIME: 0 (TIMEOUT)");
        }
    }
}
