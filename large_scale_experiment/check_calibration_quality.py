import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

# A comprehensive list of diverse, grammatically correct sentences to calibrate self-attention layers
CALIBRATION_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the landscape of modern technology and edge computing.",
    "Kolmogorov-Arnold Networks offer a promising alternative to multi-layer perceptrons.",
    "Deep learning models require careful quantization to run on low-power microcontrollers.",
    "We are evaluating the distillation performance of feed-forward networks in transformer models.",
    "Python is a versatile programming language widely used in data science and web development.",
    "The Hubble Space Telescope has captured stunning images of distant galaxies and nebulae.",
    "Quantum computing utilizes qubits and superposition to solve complex computational problems.",
    "Climate change poses a significant threat to global biodiversity and environmental stability.",
    "A healthy diet and regular exercise are essential for maintaining physical and mental well-being.",
    "The history of civilization is marked by technological innovations and cultural exchanges.",
    "Natural language processing allows computers to understand and generate human language.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "The Great Wall of China is one of the most remarkable engineering feats in human history.",
    "Blockchain technology provides a decentralized and secure ledger for digital transactions.",
    "Learning a new language opens up new perspectives and opportunities for communication.",
    "The human brain is a complex network of billions of neurons communicating via synapses.",
    "Renewable energy sources such as solar and wind power are crucial for a sustainable future.",
    "Good communication skills are vital for personal relationships and professional success.",
    "The study of economics helps us understand how resources are allocated in society.",
    "Music has the power to evoke strong emotions and connect people across different cultures.",
    "The theory of relativity proposed by Albert Einstein revolutionized our understanding of space and time.",
    "Proper sleep hygiene is important for cognitive function and overall health preservation.",
    "The Amazon rainforest is home to a vast array of unique plant and animal species.",
    "Cryptography ensures secure communication in the presence of adversarial third parties.",
    "The scientific method involves observation, hypothesis formulation, and rigorous testing.",
    "Urban planning plays a key role in creating livable and sustainable cities.",
    "The discovery of penicillin by Alexander Fleming marked a turning point in medicine.",
    "Virtual reality creates immersive digital environments for gaming, education, and training.",
    "A positive attitude and perseverance can help individuals overcome major challenges in life.",
    "The ocean covers more than seventy percent of the Earth's surface and remains largely unexplored.",
    "Machine learning algorithms can identify patterns in large datasets to make predictions.",
    "The printing press invented by Johannes Gutenberg democratized access to information.",
    "Self-driving cars use sensors and artificial intelligence to navigate roads safely.",
    "The study of philosophy encourages critical thinking and questioning of fundamental assumptions.",
    "Biodegradable materials can help reduce plastic pollution and protect marine life.",
    "The internet has revolutionized the way we access information and communicate globally.",
    "A balanced ecosystem requires a delicate harmony between predators and prey.",
    "Microprocessors are the brain of modern electronic devices, from smartphones to supercomputers.",
    "The renaissance was a period of intense artistic and intellectual revival in Europe.",
    "Genomics is the study of the complete set of DNA within an organism.",
    "Sustainable agriculture practices help conserve soil fertility and water resources.",
    "The concept of democracy originated in ancient Greece and has evolved over centuries.",
    "Enzyme catalysts speed up chemical reactions in biological systems without being consumed.",
    "E-commerce has transformed the retail industry and changed consumer shopping habits.",
    "The standard model of particle physics describes the fundamental forces of nature.",
    "Public transport systems reduce traffic congestion and carbon emissions in urban areas.",
    "The library is a repository of human knowledge and a hub for community learning.",
    "Active listening is a critical component of effective leadership and teamwork.",
    "The industrial revolution shifted economies from agrarian to industrial and manufacturing-based."
]

def check_calibration_quality():
    print("Loading BERT-small model...")
    model_name = "prajjwal1/bert-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()
    
    # Real-world test sentences (not in the calibration corpus)
    test_sentences = [
        "In machine learning, model compression is critical for edge device deployment.",
        "Kolmogorov-Arnold Networks replace linear weights with learnable activation functions.",
        "We are evaluating the distillation performance of BERT feed-forward layers.",
        "Deep learning models require optimization to run efficiently on low-power microcontrollers.",
        "Let's measure the cosine similarity of the distilled EML-KAN model replica."
    ]
    
    inputs_test = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
    inputs_calib = tokenizer(CALIBRATION_CORPUS, padding=True, truncation=True, return_tensors="pt")
    
    for layer_idx in range(4):
        print(f"\n--- Layer {layer_idx} ---")
        
        # 1. Capture real FFN test inputs
        ffn_inputs_test = []
        def test_hook_fn(module, input_tensor, output_tensor):
            ffn_inputs_test.append(input_tensor[0].detach())
            
        hook_handle = bert.encoder.layer[layer_idx].intermediate.register_forward_hook(test_hook_fn)
        with torch.no_grad():
            _ = bert(**inputs_test)
        hook_handle.remove()
        
        real_ffn_input = torch.cat(ffn_inputs_test, dim=0).view(-1, 512)
        
        # 2. Capture calibration FFN inputs (using our text corpus)
        ffn_inputs_calib = []
        def calib_hook_fn(module, input_tensor, output_tensor):
            ffn_inputs_calib.append(input_tensor[0].detach())
            
        hook_handle = bert.encoder.layer[layer_idx].intermediate.register_forward_hook(calib_hook_fn)
        with torch.no_grad():
            _ = bert(**inputs_calib)
        hook_handle.remove()
        
        calib_ffn_input = torch.cat(ffn_inputs_calib, dim=0).view(-1, 512)
        
        # Print Stats to verify alignment
        print(f"Real Test Hidden States | Mean: {real_ffn_input.mean().item():.4f} | Std: {real_ffn_input.std().item():.4f} | Min: {real_ffn_input.min().item():.4f} | Max: {real_ffn_input.max().item():.4f}")
        print(f"Calibration Text Hidden | Mean: {calib_ffn_input.mean().item():.4f} | Std: {calib_ffn_input.std().item():.4f} | Min: {calib_ffn_input.min().item():.4f} | Max: {calib_ffn_input.max().item():.4f}")

if __name__ == "__main__":
    check_calibration_quality()
