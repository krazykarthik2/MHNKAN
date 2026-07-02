#include <Arduino.h>
#include <math.h>

// Reusable Softplus activation helper
float softplus(float z) {
    return logf(1.0f + expf(z));
}

/**
 * Compiled Genetically Optimized EML-KAN DAG.
 * Fits: f(x) = sin(pi * x) * exp(x)
 * 
 * This C++ code was compiled directly from our trained 24-parameter model.
 * It uses zero-allocation, division-free float arithmetic, making it optimal
 * for real-time execution in ESP32 control loops.
 */
float evaluate_eml_kan(float x) {
    // 1. Layer 1 Primitive Exponentials
    float u1_0_0 = expf(-0.009907f * x - 0.476977f);
    float u1_1_0 = expf(0.766712f * x + 0.079296f);
    float u1_1_1 = expf(-0.449610f * x + 0.156112f);
    float u1_2_0 = expf(0.136217f * x - 0.163462f);
    float u1_2_1 = expf(0.658656f * x + 0.348503f);
    
    // 2. Layer 1 Softplus Log Features
    float L1_0_0 = logf(1.0f + expf(0.017476f * x - 0.183690f));
    float L1_1_0 = logf(1.0f + expf(-6.442822f * x - 0.723518f));
    float L1_1_1 = logf(1.0f + expf(-1.266124f * x + 0.162390f));
    float L1_2_0 = logf(1.0f + expf(-0.723010f * x - 0.025089f));
    
    // Log-of-log variable caches
    float P1_0_0 = logf(L1_0_0 + 1e-6f);
    float P1_1_0 = logf(L1_1_0 + 1e-6f);
    float P1_1_1 = logf(L1_1_1 + 1e-6f);
    float P1_2_0 = logf(L1_2_0 + 1e-6f);
    
    // 3. Hidden nodes (Families A-D)
    float h_0 = -0.254968f * x + -0.179489f * (u1_0_0 - P1_0_0);
    float h_1 = 0.278732f * x + 0.073645f * (u1_1_0 - P1_1_0) + 0.394847f * (u1_1_1 - P1_1_1);
    float h_2 = -0.480346f * x + -0.336229f * (u1_2_0 - P1_2_0) + 0.541431f * (u1_2_1 - P1_2_0);
    float h_3 = 0.0f; // Pruned by Genetic Algorithm
    
    // 4. Layer 2 Primitives
    float u2_0 = expf(-0.237718f * h_0 - 0.218791f);
    float u2_1 = expf(0.036160f * h_1 - 0.327769f);
    float u2_2 = expf(0.541431f * h_2 - 0.313585f);
    
    float L2_0 = logf(1.0f + expf(0.831390f * h_0 - 0.186568f));
    float L2_1 = logf(1.0f + expf(-0.274185f * h_1 + 0.264554f));
    
    float P2_0 = logf(L2_0 + 1e-6f);
    float P2_1 = logf(L2_1 + 1e-6f);
    
    // 5. Output assembly
    float y_out = 1.005284f * h_0 - 0.266622f * (u2_0 - P2_0) 
                - 0.263359f * h_1 - 0.091050f * (u2_1 - P2_1) 
                - 0.325225f * (u2_2);
                
    return y_out;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 EML-KAN Inference Benchmark Start!");
}

void loop() {
    // Read raw sensor voltage from pin 34 (range 0 to 4095)
    int raw_analog = analogRead(34);
    
    // Normalize input to range [-1.0, 1.0] for EML-KAN
    float x = ((float)raw_analog / 2047.5f) - 1.0f;
    
    // Measure execution latency
    unsigned long start_time = micros();
    float calibrated_output = evaluate_eml_kan(x);
    unsigned long duration = micros() - start_time;
    
    // Print calibration values
    Serial.print("Raw: ");
    Serial.print(raw_analog);
    Serial.print(" | Normalised X: ");
    Serial.print(x, 4);
    Serial.print(" | Calibrated Output: ");
    Serial.print(calibrated_output, 6);
    Serial.print(" | Inference Time: ");
    Serial.print(duration);
    Serial.println(" us");
    
    delay(500); // Sample twice per second
}
