import torch
import torchaudio
import numpy as np
import librosa
#import cupy as cp
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

class AdvancedAudioProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_fft = 2048
        self.hop_length = 512
        
    def _get_freqs(self, sr):
        """Get frequency array with correct size"""
        return torch.fft.rfftfreq(self.n_fft, d=1/sr)
        
    def apply_adaptive_noise_reduction(self, audio, sr, strength=0.5):
        """Advanced adaptive noise reduction using spectral gating"""
        try:
            if len(audio) == 0:
                return audio, {'magnitude': np.array([]), 'mask': np.array([]), 'noise_profile': np.array([])}
                
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            # Estimate noise profile
            magnitude = torch.abs(stft)
            noise_profile = torch.mean(magnitude[:, :int(magnitude.shape[1]*0.1)], dim=1, keepdim=True)
            
            # Apply adaptive thresholding
            threshold = noise_profile * (1 + strength)
            mask = (magnitude > threshold).float()
            
            # Apply mask and reconstruct
            stft_clean = stft * mask
            audio_clean = torch.istft(
                stft_clean,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_clean.cpu().numpy(), {
                'magnitude': magnitude.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'noise_profile': noise_profile.cpu().numpy()
            }
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return audio, {
                'magnitude': np.abs(np.fft.rfft(audio)),
                'mask': np.ones(len(audio) // 2 + 1),
                'noise_profile': np.zeros(len(audio) // 2 + 1)
            }

    def apply_spectral_enhancement(self, audio, sr, enhancement_type='speech'):
        """Apply spectral enhancement based on content type"""
        try:
            if len(audio) == 0:
                return audio, {'magnitude': np.array([]), 'gain': np.array([]), 'magnitude_enhanced': np.array([])}
                
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Get frequency array with correct size
            freqs = self._get_freqs(sr)
            
            # Apply different enhancement based on type
            if enhancement_type == 'speech':
                # Boost speech frequencies
                gain = torch.ones_like(magnitude)
                gain[torch.abs(freqs) > 1000] = 1.3  # Boost high frequencies
                gain[torch.abs(freqs) < 300] = 1.2   # Boost low frequencies
            elif enhancement_type == 'music':
                # Enhance musical content
                gain = torch.ones_like(magnitude)
                gain[torch.abs(freqs) > 2000] = 1.4  # Boost high frequencies
                gain[torch.abs(freqs) < 200] = 1.3   # Boost low frequencies
            else:
                # General enhancement
                gain = torch.ones_like(magnitude)
            
            # Apply enhancement
            magnitude_enhanced = magnitude * gain
            stft_enhanced = magnitude_enhanced * torch.exp(1j * phase)
            
            # Reconstruct signal
            audio_enhanced = torch.istft(
                stft_enhanced,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_enhanced.cpu().numpy(), {
                'magnitude': magnitude.cpu().numpy(),
                'gain': gain.cpu().numpy(),
                'magnitude_enhanced': magnitude_enhanced.cpu().numpy()
            }
        except Exception as e:
            print(f"Error in spectral enhancement: {str(e)}")
            return audio, {
                'magnitude': np.abs(np.fft.rfft(audio)),
                'gain': np.ones(len(audio) // 2 + 1),
                'magnitude_enhanced': np.abs(np.fft.rfft(audio))
            }

    def apply_dynamic_range_compression(self, audio, threshold=-20, ratio=4):
        """Apply dynamic range compression"""
        try:
            if len(audio) == 0:
                return audio, {'audio_db': np.array([]), 'compressed_db': np.array([])}
                
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
            
            # Apply compression
            mask = audio_db > threshold
            audio_db[mask] = threshold + (audio_db[mask] - threshold) / ratio
            
            # Convert back to linear scale
            audio_compressed = 10 ** (audio_db / 20)
            audio_compressed *= np.sign(audio)
            
            return audio_compressed, {
                'audio_db': audio_db,
                'compressed_db': 20 * np.log10(np.abs(audio_compressed) + 1e-10)
            }
        except Exception as e:
            print(f"Error in dynamic range compression: {str(e)}")
            return audio, {
                'audio_db': 20 * np.log10(np.abs(audio) + 1e-10),
                'compressed_db': 20 * np.log10(np.abs(audio) + 1e-10)
            }

    def apply_phase_correction(self, audio, sr):
        """Apply phase correction to improve clarity"""
        try:
            if len(audio) == 0:
                return audio, {'phase': np.array([]), 'phase_smooth': np.array([])}
                
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            # Apply phase correction
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Smooth phase
            phase_smooth = torch.nn.functional.avg_pool2d(
                phase.unsqueeze(0).unsqueeze(0),
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ).squeeze()
            
            # Reconstruct signal
            stft_corrected = magnitude * torch.exp(1j * phase_smooth)
            audio_corrected = torch.istft(
                stft_corrected,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_corrected.cpu().numpy(), {
                'phase': phase.cpu().numpy(),
                'phase_smooth': phase_smooth.cpu().numpy()
            }
        except Exception as e:
            print(f"Error in phase correction: {str(e)}")
            return audio, {
                'phase': np.angle(np.fft.rfft(audio)),
                'phase_smooth': np.angle(np.fft.rfft(audio))
            }

    def apply_room_correction(self, audio, sr):
        """Apply room correction to reduce room effects"""
        try:
            if len(audio) == 0:
                return audio, {'magnitude': np.array([]), 'gain': np.array([]), 'magnitude_corrected': np.array([])}
                
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Get frequency array with correct size
            freqs = self._get_freqs(sr)
            
            # Apply room correction
            gain = torch.ones_like(magnitude)
            
            # Reduce room modes
            gain[torch.abs(freqs) < 100] = 0.8  # Reduce low frequencies
            gain[torch.abs(freqs) > 8000] = 0.9  # Reduce high frequencies
            
            # Apply correction
            magnitude_corrected = magnitude * gain
            stft_corrected = magnitude_corrected * torch.exp(1j * phase)
            
            # Reconstruct signal
            audio_corrected = torch.istft(
                stft_corrected,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_corrected.cpu().numpy(), {
                'magnitude': magnitude.cpu().numpy(),
                'gain': gain.cpu().numpy(),
                'magnitude_corrected': magnitude_corrected.cpu().numpy()
            }
        except Exception as e:
            print(f"Error in room correction: {str(e)}")
            return audio, {
                'magnitude': np.abs(np.fft.rfft(audio)),
                'gain': np.ones(len(audio) // 2 + 1),
                'magnitude_corrected': np.abs(np.fft.rfft(audio))
            }

    def apply_adaptive_eq(self, audio, sr, target_curve=None):
        """Apply adaptive equalization based on target curve"""
        try:
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Get frequency array
            freqs = self._get_freqs(sr)
            
            # Calculate current spectrum
            current_spectrum = torch.mean(magnitude, dim=1)
            
            # Calculate EQ curve
            if target_curve is None or len(target_curve) == 0:
                # Generate a default target curve (slight smile curve)
                target_spectrum = torch.ones_like(current_spectrum)
                # Boost low and high frequencies slightly
                low_boost = torch.exp(-(freqs - 100)**2 / (2 * 50**2))
                high_boost = torch.exp(-(freqs - 8000)**2 / (2 * 2000**2))
                target_spectrum = target_spectrum * (1 + 0.3 * (low_boost + high_boost))
            else:
                # Ensure target curve is the right size
                if len(target_curve) != len(freqs):
                    # Resample target curve to match frequency array
                    target_curve = np.interp(freqs.cpu().numpy(), 
                                           np.linspace(0, 1, len(target_curve)), 
                                           target_curve)
                target_spectrum = torch.from_numpy(target_curve).to(self.device)
            
            # Calculate EQ gain with safety limits
            eq_gain = target_spectrum / (current_spectrum + 1e-10)
            eq_gain = torch.clamp(eq_gain, 0.1, 10.0)  # Limit gain range
            
            # Apply EQ
            magnitude_eq = magnitude * eq_gain.unsqueeze(1)
            stft_eq = magnitude_eq * torch.exp(1j * phase)
            
            # Reconstruct signal
            audio_eq = torch.istft(
                stft_eq,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_eq.cpu().numpy(), {
                'magnitude': magnitude.cpu().numpy(),
                'eq_gain': eq_gain.cpu().numpy(),
                'magnitude_eq': magnitude_eq.cpu().numpy()
            }
            
        except Exception as e:
            print(f"Error in adaptive EQ: {str(e)}")
            # Return original audio if processing fails
            return audio, {
                'magnitude': np.abs(np.fft.rfft(audio)),
                'eq_gain': np.ones(len(audio) // 2 + 1),
                'magnitude_eq': np.abs(np.fft.rfft(audio))
            }

    def apply_adaptive_limiter(self, audio, threshold=-1.0, release_time=0.1):
        """Apply adaptive limiter with variable release time"""
        try:
            if len(audio) == 0:
                return audio, {'audio_db': np.array([]), 'gain_reduction': np.array([]), 'limited_db': np.array([])}
                
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
            
            # Calculate gain reduction
            gain_reduction = np.maximum(0, audio_db - threshold)
            
            # Apply release time
            release_samples = int(release_time * 16000)  # Assuming 16kHz
            gain_reduction = np.convolve(gain_reduction, np.ones(release_samples)/release_samples, mode='same')
            
            # Apply limiting
            audio_limited = audio * 10 ** (-gain_reduction / 20)
            
            return audio_limited, {
                'audio_db': audio_db,
                'gain_reduction': gain_reduction,
                'limited_db': 20 * np.log10(np.abs(audio_limited) + 1e-10)
            }
        except Exception as e:
            print(f"Error in adaptive limiter: {str(e)}")
            return audio, {
                'audio_db': 20 * np.log10(np.abs(audio) + 1e-10),
                'gain_reduction': np.zeros_like(audio),
                'limited_db': 20 * np.log10(np.abs(audio) + 1e-10)
            }

    def apply_spectral_gating(self, audio, sr, threshold=-60, smoothing=0.1):
        """Apply spectral gating for noise reduction"""
        try:
            if len(audio) == 0:
                return audio, {'magnitude': np.array([]), 'gate_mask': np.array([]), 'magnitude_gated': np.array([])}
                
            # Convert to frequency domain
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Calculate threshold in linear scale
            threshold_linear = 10 ** (threshold / 20)
            
            # Create gate mask
            gate_mask = (magnitude > threshold_linear).float()
            
            # Apply smoothing
            gate_mask = torch.nn.functional.avg_pool2d(
                gate_mask.unsqueeze(0).unsqueeze(0),
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ).squeeze()
            
            # Apply gate
            stft_gated = stft * gate_mask
            audio_gated = torch.istft(
                stft_gated,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            return audio_gated.cpu().numpy(), {
                'magnitude': magnitude.cpu().numpy(),
                'gate_mask': gate_mask.cpu().numpy(),
                'magnitude_gated': torch.abs(stft_gated).cpu().numpy()
            }
        except Exception as e:
            print(f"Error in spectral gating: {str(e)}")
            return audio, {
                'magnitude': np.abs(np.fft.rfft(audio)),
                'gate_mask': np.ones(len(audio) // 2 + 1),
                'magnitude_gated': np.abs(np.fft.rfft(audio))
            }

    def analyze_audio(self, audio, sr):
        """Analyze audio and return various metrics"""
        try:
            if len(audio) == 0:
                return {
                    'rms': 0,
                    'peak': 0,
                    'crest_factor': 0,
                    'spectral_centroid': 0,
                    'spectral_spread': 0,
                    'spectral_rolloff': 0,
                    'spectral_flatness': 0,
                    'zero_crossing_rate': 0,
                    'magnitude': np.array([]),
                    'freqs': np.array([]),
                    'duration': 0,
                    'sample_rate': sr,
                    'bit_depth': 16,
                    'spectral_bandwidth': 0,
                    'spectral_contrast': 0,
                    'spectral_flux': 0,
                    'spectral_entropy': 0,
                    'audio': audio,
                    'spectral_features': {
                        'centroid': np.array([]),
                        'rolloff': np.array([]),
                        'flux': np.array([])
                    }
                }
                
            # Calculate basic metrics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / rms if rms > 0 else 0
            
            # Calculate spectral features
            stft = torch.stft(
                torch.from_numpy(audio).to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            
            # Get frequency array with correct size
            freqs = self._get_freqs(sr)
            
            # Calculate spectral centroid
            spectral_centroid = torch.sum(magnitude * freqs.unsqueeze(1)) / torch.sum(magnitude)
            
            # Calculate spectral spread
            spectral_spread = torch.sqrt(
                torch.sum(magnitude * (freqs.unsqueeze(1) - spectral_centroid)**2) / torch.sum(magnitude)
            )
            
            # Calculate spectral rolloff
            total_energy = torch.sum(magnitude)
            rolloff_threshold = 0.85 * total_energy
            cumulative_energy = torch.cumsum(magnitude, dim=0)
            rolloff_idx = torch.where(cumulative_energy >= rolloff_threshold)[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            
            # Calculate spectral flatness
            geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10)))
            arithmetic_mean = torch.mean(magnitude)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio)))
            zero_crossing_rate = zero_crossings / len(audio)
            
            # Calculate spectral bandwidth
            spectral_bandwidth = torch.sqrt(
                torch.sum(magnitude * (freqs.unsqueeze(1) - spectral_centroid)**2) / torch.sum(magnitude)
            )
            
            # Calculate spectral contrast
            magnitude_db = 20 * torch.log10(magnitude + 1e-10)
            peak_db = torch.max(magnitude_db, dim=0)[0]
            valley_db = torch.min(magnitude_db, dim=0)[0]
            spectral_contrast = torch.mean(peak_db - valley_db)
            
            # Calculate spectral flux
            magnitude_diff = torch.diff(magnitude, dim=1)
            spectral_flux = torch.mean(torch.abs(magnitude_diff))
            
            # Calculate spectral entropy
            magnitude_norm = magnitude / (torch.sum(magnitude, dim=0) + 1e-10)
            spectral_entropy = -torch.sum(magnitude_norm * torch.log2(magnitude_norm + 1e-10))
            
            # Calculate time-varying spectral features
            n_frames = magnitude.shape[1]
            spectral_features = {
                'centroid': np.zeros(n_frames),
                'rolloff': np.zeros(n_frames),
                'flux': np.zeros(n_frames)
            }
            
            for i in range(n_frames):
                # Spectral centroid
                frame_magnitude = magnitude[:, i]
                if torch.sum(frame_magnitude) > 0:
                    spectral_features['centroid'][i] = torch.sum(frame_magnitude * freqs) / torch.sum(frame_magnitude)
                
                # Spectral rolloff
                frame_energy = torch.sum(frame_magnitude)
                if frame_energy > 0:
                    rolloff_threshold = 0.85 * frame_energy
                    cumulative_energy = torch.cumsum(frame_magnitude, dim=0)
                    rolloff_idx = torch.where(cumulative_energy >= rolloff_threshold)[0][0]
                    spectral_features['rolloff'][i] = freqs[rolloff_idx]
                
                # Spectral flux
                if i > 0:
                    spectral_features['flux'][i] = torch.sum(torch.abs(frame_magnitude - magnitude[:, i-1]))
            
            return {
                'rms': rms,
                'peak': peak,
                'crest_factor': crest_factor,
                'spectral_centroid': spectral_centroid.cpu().numpy(),
                'spectral_spread': spectral_spread.cpu().numpy(),
                'spectral_rolloff': spectral_rolloff.cpu().numpy(),
                'spectral_flatness': spectral_flatness.cpu().numpy(),
                'zero_crossing_rate': zero_crossing_rate,
                'magnitude': magnitude.cpu().numpy(),
                'freqs': freqs.cpu().numpy(),
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'bit_depth': 16,  # Assuming 16-bit audio
                'spectral_bandwidth': spectral_bandwidth.cpu().numpy(),
                'spectral_contrast': spectral_contrast.cpu().numpy(),
                'spectral_flux': spectral_flux.cpu().numpy(),
                'spectral_entropy': spectral_entropy.cpu().numpy(),
                'audio': audio,
                'spectral_features': spectral_features
            }
        except Exception as e:
            print(f"Error in audio analysis: {str(e)}")
            return {
                'rms': 0,
                'peak': 0,
                'crest_factor': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_rolloff': 0,
                'spectral_flatness': 0,
                'zero_crossing_rate': 0,
                'magnitude': np.array([]),
                'freqs': np.array([]),
                'duration': 0,
                'sample_rate': sr,
                'bit_depth': 16,
                'spectral_bandwidth': 0,
                'spectral_contrast': 0,
                'spectral_flux': 0,
                'spectral_entropy': 0,
                'audio': audio,
                'spectral_features': {
                    'centroid': np.array([]),
                    'rolloff': np.array([]),
                    'flux': np.array([])
                }
            } 