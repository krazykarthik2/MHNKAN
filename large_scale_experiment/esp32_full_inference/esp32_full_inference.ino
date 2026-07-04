#include <Arduino.h>

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include hex model array
#include "tflite_model.h"

// Define Tensor Arena size (160 KB for MobileNetV3 + KAN activations)
constexpr int kTensorArenaSize = 160 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Standalone EML-KAN MobileNet Inference Start!");
    
    // 1. Load TFLite Model from Program memory
    model = tflite::GetModel(g_tflite_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model schema version ");
        Serial.print(model->version());
        Serial.print(" is not equal to supported version ");
        Serial.println(TFLITE_SCHEMA_VERSION);
        return;
    }
    
    // 2. Setup Operations Resolver (AllOpsResolver handles convolutions + custom elements)
    static tflite::AllOpsResolver resolver;
    
    // 3. Initialize Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    // 4. Allocate Tensors in the Arena
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfliteOk) {
        Serial.println("AllocateTensors() failed!");
        return;
    }
    
    // 5. Get input/output pointers
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("TFLite Micro initialized successfully. Standalone execution ready.");
}

void loop() {
    // Generate dummy image frame representing a 128x128 RGB test frame
    // In production, you would read this from an ESP32-CAM frame buffer
    float input_scale = input->params.scale;
    int32_t input_zero_point = input->params.zero_point;
    
    // Fill the quantized Int8 input tensor: shape [1, 3, 128, 128]
    // Normalized to match CIFAR stats: (pixel - mean) / std
    float mean[3] = {0.5071f, 0.4867f, 0.4408f};
    float std[3] = {0.2675f, 0.2565f, 0.2761f};
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                // Mock pixel value in range [0, 1]
                float raw_pixel = (float)rand() / (float)RAND_MAX;
                float normalized = (raw_pixel - mean[c]) / std[c];
                
                // Perform quantization to Int8
                int index = c * 128 * 128 + y * 128 + x;
                int8_t quantized_val = (int8_t)round(normalized / input_scale) + input_zero_point;
                input->data.int8[index] = quantized_val;
            }
        }
    }
    
    // Measure Standalone Inference Latency (conv backbone + KAN classifier)
    unsigned long start_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long duration = millis() - start_time;
    
    if (invoke_status != kTfliteOk) {
        Serial.println("Interpreter invoke failed!");
        delay(2000);
        return;
    }
    
    // De-quantize outputs to get logits and find the maximum class prediction
    float output_scale = output->params.scale;
    int32_t output_zero_point = output->params.zero_point;
    
    int max_class = 0;
    float max_logit = (output->data.int8[0] - output_zero_point) * output_scale;
    
    for (int c = 1; c < 100; c++) {
        float logit = (output->data.int8[c] - output_zero_point) * output_scale;
        if (logit > max_logit) {
            max_logit = logit;
            max_class = c;
        }
    }
    
    Serial.print("Predicted Class: ");
    Serial.print(max_class);
    Serial.print(" | De-quantized Logit: ");
    Serial.print(max_logit, 4);
    Serial.print(" | Full Pipeline Latency: ");
    Serial.print(duration);
    Serial.println(" ms");
    
    delay(2000); // Run once every 2 seconds
}
