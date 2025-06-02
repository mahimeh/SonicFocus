import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import librosa
import os
import warnings
warnings.filterwarnings('ignore')

class AudioEnhancer:
    def __init__(self):
        # Check for CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("CUDA not available. Using CPU instead.")
        
        # Initialize audio processing parameters
        self.sample_rate = 16000  # Standard sample rate for processing
        self.n_fft = 2048
        self.hop_length = 512
        self.freq_mask_param = 30
        self.time_mask_param = 20
        
        # Initialize noise reduction parameters
        self.noise_reduce_strength = 0.15
        self.noise_threshold = 0.15
        
    def load_audio(self, input_path):
        """Load audio file and convert to mono if necessary"""
        try:
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return None, None

    def estimate_noise_profile(self, audio_tensor):
        """Estimate noise profile using GPU-accelerated processing"""
        try:
            # Convert to frequency domain using STFT
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            
            # Estimate noise floor
            noise_profile = torch.mean(magnitude[:, :int(magnitude.shape[1]*0.1)], dim=1, keepdim=True)
            return noise_profile
        except Exception as e:
            print(f"Error in noise profile estimation: {str(e)}")
            return None

    def spectral_subtraction(self, audio_tensor, noise_profile):
        """Perform spectral subtraction using GPU-accelerated processing"""
        try:
            # Compute STFT
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Subtract noise profile
            magnitude_clean = torch.clamp(magnitude - noise_profile * self.noise_reduce_strength, min=0)
            
            # Reconstruct signal
            stft_clean = magnitude_clean * torch.exp(1j * phase)
            audio_clean = torch.istft(
                stft_clean,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=audio_tensor.shape[0]
            )
            
            return audio_clean
        except Exception as e:
            print(f"Error in spectral subtraction: {str(e)}")
            return audio_tensor

    def reduce_noise(self, audio, sr):
        """Apply advanced noise reduction using GPU-accelerated processing"""
        try:
            # Convert to PyTorch tensor and move to GPU
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # Estimate noise profile
            noise_profile = self.estimate_noise_profile(audio_tensor)
            if noise_profile is None:
                return audio
            
            # Apply spectral subtraction
            audio_clean = self.spectral_subtraction(audio_tensor, noise_profile)
            
            # Apply adaptive thresholding
            stft = torch.stft(
                audio_clean,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            threshold = torch.mean(magnitude) * self.noise_threshold
            mask = (magnitude > threshold).float()
            stft_filtered = stft * mask
            
            # Convert back to time domain
            audio_filtered = torch.istft(
                stft_filtered,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=audio_tensor.shape[0]
            )
            
            # Convert back to numpy
            audio_enhanced = audio_filtered.cpu().numpy()
            
            return audio_enhanced
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return audio

    def enhance_audio(self, audio, sr):
        """Enhance audio quality using GPU-accelerated processing"""
        try:
            # Convert to PyTorch tensor and move to GPU
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # Normalize audio
            audio_tensor = F.normalize(audio_tensor, dim=0)
            
            # Apply STFT
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Enhance high frequencies
            freqs = torch.fft.fftfreq(self.n_fft, d=1/sr)
            gain = torch.ones_like(magnitude)
            gain[torch.abs(freqs) > 1000] = 1.2  # Boost high frequencies
            
            # Apply frequency masking for noise reduction
            freq_mask = (torch.rand_like(magnitude) > self.noise_reduce_strength).float()
            magnitude_enhanced = magnitude * gain * freq_mask
            
            # Reconstruct signal
            stft_enhanced = magnitude_enhanced * torch.exp(1j * phase)
            audio_enhanced = torch.istft(
                stft_enhanced,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=audio_tensor.shape[0]
            )
            
            # Convert back to numpy
            audio_enhanced = audio_enhanced.cpu().numpy()
            
            return audio_enhanced
        except Exception as e:
            print(f"Error in audio enhancement: {str(e)}")
            return audio

    def save_audio(self, audio, sr, output_path):
        """Save the processed audio"""
        try:
            sf.write(output_path, audio, sr)
            print(f"Enhanced audio saved to: {output_path}")
        except Exception as e:
            print(f"Error saving audio: {str(e)}")

    def process_audio(self, input_path, output_path):
        """Main processing pipeline"""
        print("Loading audio...")
        audio, sr = self.load_audio(input_path)
        if audio is None:
            return False

        print("Reducing background noise using GPU-accelerated processing...")
        audio = self.reduce_noise(audio, sr)

        print("Enhancing audio using GPU acceleration...")
        audio = self.enhance_audio(audio, sr)

        print("Saving enhanced audio...")
        self.save_audio(audio, sr, output_path)
        return True

def main():
    enhancer = AudioEnhancer()
    
    # Get input file path from user
    input_path = input("Enter the path to your audio file: ")
    
    # Create output path
    output_path = os.path.splitext(input_path)[0] + "_enhanced.wav"
    
    # Process the audio
    success = enhancer.process_audio(input_path, output_path)
    
    if success:
        print("Audio processing completed successfully!")
    else:
        print("Audio processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
