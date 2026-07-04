import os
import sys
import numpy as np

def install_dependencies():
    print("Verifying/Installing converter dependencies (tensorflow, onnx2tf)...")
    import subprocess
    packages = ["tensorflow", "onnx2tf", "flatbuffers", "onnx"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def convert_onnx_to_saved_model(onnx_path, output_dir):
    print(f"Converting ONNX model {onnx_path} to TensorFlow SavedModel using onnx2tf...")
    import subprocess
    # Run onnx2tf command line tool to perform structural conversion
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "--non_verbose"
    ]
    subprocess.check_call(cmd)
    print(f"SavedModel generated successfully at {output_dir}")

def quantize_saved_model_to_tflite(saved_model_dir, tflite_output_path):
    print("Loading SavedModel and applying Int8 Post-Training Quantization...")
    import tensorflow as tf
    
    # Setup representative dataset generator for calibration
    def representative_dataset_gen():
        # Generate 100 sample images matching CIFAR-100 normalized inputs [1, 3, 128, 128]
        for _ in range(100):
            data = np.random.randn(1, 3, 128, 128).astype(np.float32)
            yield [data]
            
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce fully quantized Int8 operations for microcontrollers
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Running TFLite conversion (this may take a minute to calibrate operations)...")
    tflite_model = converter.convert()
    
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Quantized TFLite model saved successfully at {tflite_output_path}")
    print(f"Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")
    return tflite_model

def generate_cpp_header(tflite_model, header_path):
    print(f"Generating C++ hex header at {header_path}...")
    dir_name = os.path.dirname(header_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    with open(header_path, "w") as f:
        f.write("/* Automatically generated Quantized TFLite Model hex array for ESP32 */\n")
        f.write("#ifndef TFLITE_MODEL_H\n")
        f.write("#define TFLITE_MODEL_H\n\n")
        f.write("#include <Arduino.h>\n\n")
        
        f.write(f"const unsigned char g_tflite_model[] PROGMEM = {{\n    ")
        for idx, val in enumerate(tflite_model):
            f.write(f"0x{val:02x}")
            if idx < len(tflite_model) - 1:
                f.write(", ")
            if (idx + 1) % 12 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")
        f.write(f"const unsigned int g_tflite_model_len = {len(tflite_model)};\n\n")
        f.write("#endif // TFLITE_MODEL_H\n")
        
    print("C++ header generated successfully.")

def main():
    # Setup paths (resolve relative paths if executed inside large_scale_experiment)
    base_dir = "large_scale_experiment" if os.path.exists("large_scale_experiment") else "."
    onnx_path = os.path.join(base_dir, "eml_kan_mobilenet.onnx")
    saved_model_dir = os.path.join(base_dir, "tf_saved_model")
    tflite_path = os.path.join(base_dir, "eml_kan_quantized.tflite")
    header_path = os.path.join(base_dir, "esp32_full_inference", "tflite_model.h")
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Please run train_cifar100_esp32.py first.")
        sys.exit(1)
        
    install_dependencies()
    
    try:
        convert_onnx_to_saved_model(onnx_path, saved_model_dir)
        tflite_model = quantize_saved_model_to_tflite(saved_model_dir, tflite_path)
        generate_cpp_header(tflite_model, header_path)
        print("\nDeployment files successfully prepared for ESP32!")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
