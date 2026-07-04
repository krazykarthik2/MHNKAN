#include <Arduino.h>
#include <math.h>
#include "esp32_cifar100_inference.h"

// The EML-KAN classifier evaluation function evaluate_eml_kan_classifier()
// is fully generated as an optimized C++ DAG graph inside esp32_cifar100_inference.h.

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 EML-KAN Serial HIL Tester Ready!");
    
    // Sanity check to verify memory alignment and read accuracy
    Serial.print("INIT_CHECK: scale=");
    Serial.print(FC_SCALE, 8);
    Serial.print(" weight_0=");
    Serial.print(FC_WEIGHTS_QUANT[0]);
    Serial.print(" base_0=");
    Serial.println(ACT3_W_BASE[0], 6);
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
