import torch
import torchaudio
import numpy as np
import soundfile as sf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import Wav2Vec2Processor

# Load model (Example: NVIDIA RNN Noise Suppression Model)
MODEL_PATH = "nvidia/rnnoise"  # Change to your model path if using a custom model

# Load a pre-trained speech enhancement model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("snakers4/silero-models", "silero_vad", device=device)
model.eval()

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_audio(file_path):
    """Load an audio file as tensor."""
    audio, sr = torchaudio.load(file_path)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    return audio, 16000

def gpu_denoise(audio_tensor):
    """Denoise audio using CUDA TensorRT model."""
    audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        enhanced_audio = model(audio_tensor)
    return enhanced_audio.cpu().numpy()

def process_and_save(input_file, output_file):
    """Load, process, and save enhanced audio."""
    audio, sr = load_audio(input_file)
    enhanced_audio = gpu_denoise(audio)
    sf.write(output_file, enhanced_audio.T, sr)
    print(f"Enhanced audio saved to {output_file}")

# Example usage
input_audio = "noisy_audio.wav"
output_audio = "enhanced_audio.wav"

process_and_save(input_audio, output_audio)
