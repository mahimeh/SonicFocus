import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QSlider, QComboBox, QMessageBox,
                            QTabWidget, QGroupBox, QGridLayout, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QSplitter, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont
import pyqtgraph as pg
import numpy as np
from main import AudioEnhancer
from advanced_processing import AdvancedAudioProcessor
import librosa

class AudioProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    analysis_complete = pyqtSignal(dict)
    processing_update = pyqtSignal(str, dict)
    
    def __init__(self, input_path, output_path, processing_options):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.processing_options = processing_options
        self.enhancer = AudioEnhancer()
        self.advanced_processor = AdvancedAudioProcessor()
        
    def run(self):
        try:
            # Load audio
            audio, sr = self.enhancer.load_audio(self.input_path)
            if audio is None:
                self.error.emit("Failed to load audio file")
                return
                
            # Analyze audio
            analysis = self.advanced_processor.analyze_audio(audio, sr)
            self.analysis_complete.emit(analysis)
            
            # Apply processing based on options
            self.progress.emit(10)
            
            if self.processing_options['noise_reduction']:
                audio, viz_data = self.advanced_processor.apply_adaptive_noise_reduction(
                    audio, sr, self.processing_options['noise_strength']
                )
                self.processing_update.emit('noise_reduction', viz_data)
                self.progress.emit(20)
            
            if self.processing_options['spectral_enhancement']:
                audio, viz_data = self.advanced_processor.apply_spectral_enhancement(
                    audio, sr, self.processing_options['enhancement_type']
                )
                self.processing_update.emit('spectral_enhancement', viz_data)
                self.progress.emit(30)
            
            if self.processing_options['dynamic_range']:
                audio, viz_data = self.advanced_processor.apply_dynamic_range_compression(
                    audio, 
                    self.processing_options['compression_threshold'],
                    self.processing_options['compression_ratio']
                )
                self.processing_update.emit('dynamic_range', viz_data)
                self.progress.emit(40)
            
            if self.processing_options['phase_correction']:
                audio, viz_data = self.advanced_processor.apply_phase_correction(audio, sr)
                self.processing_update.emit('phase_correction', viz_data)
                self.progress.emit(50)
            
            if self.processing_options['room_correction']:
                audio, viz_data = self.advanced_processor.apply_room_correction(audio, sr)
                self.processing_update.emit('room_correction', viz_data)
                self.progress.emit(60)
            
            if self.processing_options['adaptive_eq']:
                audio, viz_data = self.advanced_processor.apply_adaptive_eq(
                    audio, sr, self.processing_options['target_curve']
                )
                self.processing_update.emit('adaptive_eq', viz_data)
                self.progress.emit(70)
            
            if self.processing_options['limiter']:
                audio, viz_data = self.advanced_processor.apply_adaptive_limiter(
                    audio,
                    self.processing_options['limiter_threshold'],
                    self.processing_options['limiter_release']
                )
                self.processing_update.emit('limiter', viz_data)
                self.progress.emit(80)
            
            if self.processing_options['spectral_gating']:
                audio, viz_data = self.advanced_processor.apply_spectral_gating(
                    audio, sr,
                    self.processing_options['gate_threshold'],
                    self.processing_options['gate_smoothing']
                )
                self.processing_update.emit('spectral_gating', viz_data)
                self.progress.emit(90)
            
            # Save processed audio
            self.enhancer.save_audio(audio, sr, self.output_path)
            self.progress.emit(100)
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(str(e))

class AudioEnhancerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Audio Enhancement Studio")
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create header
        header = QLabel("Advanced Audio Enhancement Studio")
        header.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Create main processing tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Create file selection section
        file_section = QWidget()
        file_layout = QHBoxLayout(file_section)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px;")
        
        select_button = QPushButton("Select Audio File")
        select_button.clicked.connect(self.select_file)
        select_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        main_layout.addWidget(file_section)
        
        # Create visualization section
        viz_section = QWidget()
        viz_layout = QHBoxLayout(viz_section)
        
        # Waveform plot
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setBackground('w')
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Time (s)')
        viz_layout.addWidget(self.waveform_plot)
        
        # Spectrum plot
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('w')
        self.spectrum_plot.setLabel('left', 'Magnitude (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        viz_layout.addWidget(self.spectrum_plot)
        
        main_layout.addWidget(viz_section)
        
        # Create processing visualization section
        self.processing_plot = pg.PlotWidget()
        self.processing_plot.setBackground('w')
        self.processing_plot.setLabel('left', 'Magnitude')
        self.processing_plot.setLabel('bottom', 'Frequency (Hz)')
        main_layout.addWidget(self.processing_plot)
        
        # Create processing options section
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout()
        
        # Noise reduction options
        self.noise_reduction_cb = QCheckBox("Noise Reduction")
        self.noise_reduction_cb.setChecked(True)
        options_layout.addWidget(self.noise_reduction_cb, 0, 0)
        
        self.noise_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_strength_slider.setRange(0, 100)
        self.noise_strength_slider.setValue(50)
        options_layout.addWidget(self.noise_strength_slider, 0, 1)
        
        # Enhancement type
        self.enhancement_type_combo = QComboBox()
        self.enhancement_type_combo.addItems(["Speech", "Music", "General"])
        options_layout.addWidget(QLabel("Enhancement Type:"), 1, 0)
        options_layout.addWidget(self.enhancement_type_combo, 1, 1)
        
        # Dynamic range compression
        self.dynamic_range_cb = QCheckBox("Dynamic Range Compression")
        options_layout.addWidget(self.dynamic_range_cb, 2, 0)
        
        self.compression_threshold = QDoubleSpinBox()
        self.compression_threshold.setRange(-60, 0)
        self.compression_threshold.setValue(-20)
        options_layout.addWidget(QLabel("Threshold (dB):"), 2, 1)
        options_layout.addWidget(self.compression_threshold, 2, 2)
        
        self.compression_ratio = QDoubleSpinBox()
        self.compression_ratio.setRange(1, 20)
        self.compression_ratio.setValue(4)
        options_layout.addWidget(QLabel("Ratio:"), 2, 3)
        options_layout.addWidget(self.compression_ratio, 2, 4)
        
        # Phase correction
        self.phase_correction_cb = QCheckBox("Phase Correction")
        options_layout.addWidget(self.phase_correction_cb, 3, 0)
        
        # Room correction
        self.room_correction_cb = QCheckBox("Room Correction")
        options_layout.addWidget(self.room_correction_cb, 3, 1)
        
        # Adaptive EQ
        self.adaptive_eq_cb = QCheckBox("Adaptive EQ")
        options_layout.addWidget(self.adaptive_eq_cb, 4, 0)
        
        # Limiter
        self.limiter_cb = QCheckBox("Adaptive Limiter")
        options_layout.addWidget(self.limiter_cb, 4, 1)
        
        self.limiter_threshold = QDoubleSpinBox()
        self.limiter_threshold.setRange(-60, 0)
        self.limiter_threshold.setValue(-1)
        options_layout.addWidget(QLabel("Limiter Threshold (dB):"), 4, 2)
        options_layout.addWidget(self.limiter_threshold, 4, 3)
        
        self.limiter_release = QDoubleSpinBox()
        self.limiter_release.setRange(0.01, 1.0)
        self.limiter_release.setValue(0.1)
        options_layout.addWidget(QLabel("Release Time (s):"), 4, 4)
        options_layout.addWidget(self.limiter_release, 4, 5)
        
        # Spectral gating
        self.spectral_gating_cb = QCheckBox("Spectral Gating")
        options_layout.addWidget(self.spectral_gating_cb, 5, 0)
        
        self.gate_threshold = QDoubleSpinBox()
        self.gate_threshold.setRange(-80, -20)
        self.gate_threshold.setValue(-60)
        options_layout.addWidget(QLabel("Gate Threshold (dB):"), 5, 1)
        options_layout.addWidget(self.gate_threshold, 5, 2)
        
        self.gate_smoothing = QDoubleSpinBox()
        self.gate_smoothing.setRange(0.01, 0.5)
        self.gate_smoothing.setValue(0.1)
        options_layout.addWidget(QLabel("Smoothing:"), 5, 3)
        options_layout.addWidget(self.gate_smoothing, 5, 4)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Create progress section
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Create process button
        self.process_button = QPushButton("Process Audio")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setEnabled(False)
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        main_layout.addWidget(self.process_button)
        
        # Add main tab
        tabs.addTab(main_tab, "Processing")
        
        # Create analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # Analysis results
        self.analysis_group = QGroupBox("Audio Analysis")
        analysis_results_layout = QGridLayout()
        
        # Basic metrics
        basic_metrics_group = QGroupBox("Basic Metrics")
        basic_metrics_layout = QGridLayout()
        
        # Add more detailed basic metrics
        self.rms_label = QLabel("RMS: --")
        self.peak_label = QLabel("Peak: --")
        self.crest_label = QLabel("Crest Factor: --")
        self.dynamic_range_label = QLabel("Dynamic Range: --")
        self.zero_crossing_label = QLabel("Zero Crossing Rate: --")
        self.duration_label = QLabel("Duration: --")
        self.sample_rate_label = QLabel("Sample Rate: --")
        self.bit_depth_label = QLabel("Bit Depth: --")
        
        basic_metrics_layout.addWidget(self.rms_label, 0, 0)
        basic_metrics_layout.addWidget(self.peak_label, 0, 1)
        basic_metrics_layout.addWidget(self.crest_label, 1, 0)
        basic_metrics_layout.addWidget(self.dynamic_range_label, 1, 1)
        basic_metrics_layout.addWidget(self.zero_crossing_label, 2, 0)
        basic_metrics_layout.addWidget(self.duration_label, 2, 1)
        basic_metrics_layout.addWidget(self.sample_rate_label, 3, 0)
        basic_metrics_layout.addWidget(self.bit_depth_label, 3, 1)
        
        basic_metrics_group.setLayout(basic_metrics_layout)
        analysis_results_layout.addWidget(basic_metrics_group, 0, 0)
        
        # Spectral metrics
        spectral_metrics_group = QGroupBox("Spectral Metrics")
        spectral_metrics_layout = QGridLayout()
        
        self.centroid_label = QLabel("Spectral Centroid: --")
        self.spread_label = QLabel("Spectral Spread: --")
        self.rolloff_label = QLabel("Spectral Rolloff: --")
        self.flatness_label = QLabel("Spectral Flatness: --")
        self.bandwidth_label = QLabel("Spectral Bandwidth: --")
        self.contrast_label = QLabel("Spectral Contrast: --")
        self.flux_label = QLabel("Spectral Flux: --")
        self.entropy_label = QLabel("Spectral Entropy: --")
        
        spectral_metrics_layout.addWidget(self.centroid_label, 0, 0)
        spectral_metrics_layout.addWidget(self.spread_label, 0, 1)
        spectral_metrics_layout.addWidget(self.rolloff_label, 1, 0)
        spectral_metrics_layout.addWidget(self.flatness_label, 1, 1)
        spectral_metrics_layout.addWidget(self.bandwidth_label, 2, 0)
        spectral_metrics_layout.addWidget(self.contrast_label, 2, 1)
        spectral_metrics_layout.addWidget(self.flux_label, 3, 0)
        spectral_metrics_layout.addWidget(self.entropy_label, 3, 1)
        
        spectral_metrics_group.setLayout(spectral_metrics_layout)
        analysis_results_layout.addWidget(spectral_metrics_group, 1, 0)
        
        # Analysis plots
        analysis_plots_group = QGroupBox("Analysis Plots")
        analysis_plots_layout = QVBoxLayout()
        
        # Create a splitter for plots
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top row of plots
        top_plots = QWidget()
        top_layout = QHBoxLayout(top_plots)
        
        # Spectrum plot
        self.analysis_spectrum_plot = pg.PlotWidget()
        self.analysis_spectrum_plot.setBackground('w')
        self.analysis_spectrum_plot.setLabel('left', 'Magnitude (dB)')
        self.analysis_spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.analysis_spectrum_plot.setTitle("Frequency Spectrum")
        top_layout.addWidget(self.analysis_spectrum_plot)
        
        # Waveform plot
        self.analysis_waveform_plot = pg.PlotWidget()
        self.analysis_waveform_plot.setBackground('w')
        self.analysis_waveform_plot.setLabel('left', 'Amplitude')
        self.analysis_waveform_plot.setLabel('bottom', 'Time (s)')
        self.analysis_waveform_plot.setTitle("Waveform")
        top_layout.addWidget(self.analysis_waveform_plot)
        
        plot_splitter.addWidget(top_plots)
        
        # Bottom row of plots
        bottom_plots = QWidget()
        bottom_layout = QHBoxLayout(bottom_plots)
        
        # Spectrogram plot
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setBackground('w')
        self.spectrogram_plot.setLabel('left', 'Frequency (Hz)')
        self.spectrogram_plot.setLabel('bottom', 'Time (s)')
        self.spectrogram_plot.setTitle("Spectrogram")
        bottom_layout.addWidget(self.spectrogram_plot)
        
        # Spectral features plot
        self.spectral_features_plot = pg.PlotWidget()
        self.spectral_features_plot.setBackground('w')
        self.spectral_features_plot.setLabel('left', 'Value')
        self.spectral_features_plot.setLabel('bottom', 'Time (s)')
        self.spectral_features_plot.setTitle("Spectral Features")
        bottom_layout.addWidget(self.spectral_features_plot)
        
        plot_splitter.addWidget(bottom_plots)
        
        analysis_plots_layout.addWidget(plot_splitter)
        analysis_plots_group.setLayout(analysis_plots_layout)
        analysis_results_layout.addWidget(analysis_plots_group, 2, 0)
        
        self.analysis_group.setLayout(analysis_results_layout)
        analysis_layout.addWidget(self.analysis_group)
        
        # Add analysis tab
        tabs.addTab(analysis_tab, "Analysis")
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac)"
        )
        
        if file_name:
            self.file_label.setText(os.path.basename(file_name))
            self.input_file = file_name
            self.process_button.setEnabled(True)
            self.load_audio_preview(file_name)
            
    def load_audio_preview(self, file_path):
        try:
            # Load audio and plot waveform
            enhancer = AudioEnhancer()
            audio, sr = enhancer.load_audio(file_path)
            if audio is not None:
                # Create advanced processor for analysis
                advanced_processor = AdvancedAudioProcessor()
                
                # Perform analysis
                analysis = advanced_processor.analyze_audio(audio, sr)
                
                # Update analysis display
                self.update_analysis(analysis)
                
                # Plot waveform
                self.waveform_plot.clear()
                time = np.arange(len(audio)) / sr
                self.waveform_plot.plot(time, audio, pen='b')
                
                # Plot spectrum
                self.spectrum_plot.clear()
                spectrum = np.abs(np.fft.rfft(audio))
                freqs = np.fft.rfftfreq(len(audio), 1/sr)
                self.spectrum_plot.plot(freqs, 20 * np.log10(spectrum + 1e-10), pen='r')
                
                # Plot spectrogram
                self.spectrogram_plot.clear()
                # Use librosa's mel spectrogram for better visualization
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                # Create image item for spectrogram
                img = pg.ImageItem()
                self.spectrogram_plot.addItem(img)
                # Set the image data
                img.setImage(mel_spec_db, autoLevels=False)
                # Set the correct orientation
                img.setTransform(pg.QtGui.QTransform().scale(1, -1))
                # Set the correct position
                img.setPos(0, 0)
                
                # Update spectral features plot
                self.spectral_features_plot.clear()
                if 'spectral_features' in analysis:
                    features = analysis['spectral_features']
                    time = np.arange(len(features['centroid'])) / sr
                    self.spectral_features_plot.plot(time, features['centroid'], pen='r', name='Centroid')
                    self.spectral_features_plot.plot(time, features['rolloff'], pen='g', name='Rolloff')
                    self.spectral_features_plot.plot(time, features['flux'], pen='b', name='Flux')
                    self.spectral_features_plot.addLegend()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load audio preview: {str(e)}")
            print(f"Detailed error: {str(e)}")  # Add detailed error logging
            
    def process_audio(self):
        if not hasattr(self, 'input_file'):
            return
            
        # Create output path
        output_path = os.path.splitext(self.input_file)[0] + "_enhanced.wav"
        
        # Get processing options
        processing_options = {
            'noise_reduction': self.noise_reduction_cb.isChecked(),
            'noise_strength': self.noise_strength_slider.value() / 100,
            'spectral_enhancement': True,
            'enhancement_type': self.enhancement_type_combo.currentText().lower(),
            'dynamic_range': self.dynamic_range_cb.isChecked(),
            'compression_threshold': self.compression_threshold.value(),
            'compression_ratio': self.compression_ratio.value(),
            'phase_correction': self.phase_correction_cb.isChecked(),
            'room_correction': self.room_correction_cb.isChecked(),
            'adaptive_eq': self.adaptive_eq_cb.isChecked(),
            'limiter': self.limiter_cb.isChecked(),
            'limiter_threshold': self.limiter_threshold.value(),
            'limiter_release': self.limiter_release.value(),
            'spectral_gating': self.spectral_gating_cb.isChecked(),
            'gate_threshold': self.gate_threshold.value(),
            'gate_smoothing': self.gate_smoothing.value()
        }
        
        # Disable UI elements during processing
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start processing thread
        self.thread = AudioProcessingThread(self.input_file, output_path, processing_options)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.analysis_complete.connect(self.update_analysis)
        self.thread.processing_update.connect(self.update_processing_visualization)
        self.thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_analysis(self, analysis):
        try:
            # Update basic metrics
            self.rms_label.setText(f"RMS: {analysis['rms']:.3f}")
            self.peak_label.setText(f"Peak: {analysis['peak']:.3f}")
            self.crest_label.setText(f"Crest Factor: {analysis['crest_factor']:.3f}")
            
            # Calculate dynamic range
            dynamic_range = 20 * np.log10(analysis['peak'] / analysis['rms']) if analysis['rms'] > 0 else 0
            self.dynamic_range_label.setText(f"Dynamic Range: {dynamic_range:.1f} dB")
            
            # Update zero crossing rate
            self.zero_crossing_label.setText(f"Zero Crossing Rate: {analysis['zero_crossing_rate']:.3f}")
            
            # Update duration and sample rate
            if 'duration' in analysis:
                self.duration_label.setText(f"Duration: {analysis['duration']:.2f} s")
            if 'sample_rate' in analysis:
                self.sample_rate_label.setText(f"Sample Rate: {analysis['sample_rate']} Hz")
            if 'bit_depth' in analysis:
                self.bit_depth_label.setText(f"Bit Depth: {analysis['bit_depth']} bits")
            
            # Update spectral metrics
            self.centroid_label.setText(f"Spectral Centroid: {analysis['spectral_centroid']:.1f} Hz")
            self.spread_label.setText(f"Spectral Spread: {analysis['spectral_spread']:.1f} Hz")
            self.rolloff_label.setText(f"Spectral Rolloff: {analysis['spectral_rolloff']:.1f} Hz")
            self.flatness_label.setText(f"Spectral Flatness: {analysis['spectral_flatness']:.3f}")
            
            # Update additional spectral metrics if available
            if 'spectral_bandwidth' in analysis:
                self.bandwidth_label.setText(f"Spectral Bandwidth: {analysis['spectral_bandwidth']:.1f} Hz")
            if 'spectral_contrast' in analysis:
                self.contrast_label.setText(f"Spectral Contrast: {analysis['spectral_contrast']:.3f}")
            if 'spectral_flux' in analysis:
                self.flux_label.setText(f"Spectral Flux: {analysis['spectral_flux']:.3f}")
            if 'spectral_entropy' in analysis:
                self.entropy_label.setText(f"Spectral Entropy: {analysis['spectral_entropy']:.3f}")
            
            # Update spectrum plot
            self.analysis_spectrum_plot.clear()
            if len(analysis['magnitude']) > 0:
                magnitude = analysis['magnitude'].mean(axis=1)
                freqs = analysis['freqs']
                self.analysis_spectrum_plot.plot(freqs, 20 * np.log10(magnitude + 1e-10), pen='b')
                
                # Add spectral centroid marker
                centroid_line = pg.InfiniteLine(
                    pos=analysis['spectral_centroid'],
                    angle=90,
                    pen=pg.mkPen('r', width=2)
                )
                self.analysis_spectrum_plot.addItem(centroid_line)
                
                # Add spectral rolloff marker
                rolloff_line = pg.InfiniteLine(
                    pos=analysis['spectral_rolloff'],
                    angle=90,
                    pen=pg.mkPen('g', width=2)
                )
                self.analysis_spectrum_plot.addItem(rolloff_line)
            
            # Update waveform plot
            self.analysis_waveform_plot.clear()
            if len(analysis['magnitude']) > 0:
                time = np.arange(len(analysis['magnitude'])) / analysis['sample_rate']
                self.analysis_waveform_plot.plot(time, analysis['magnitude'].mean(axis=0), pen='b')
                
                # Add RMS level line
                rms_line = pg.InfiniteLine(
                    pos=analysis['rms'],
                    angle=0,
                    pen=pg.mkPen('r', width=2)
                )
                self.analysis_waveform_plot.addItem(rms_line)
                
                # Add peak level line
                peak_line = pg.InfiniteLine(
                    pos=analysis['peak'],
                    angle=0,
                    pen=pg.mkPen('g', width=2)
                )
                self.analysis_waveform_plot.addItem(peak_line)
            
            # Update spectrogram
            self.spectrogram_plot.clear()
            if 'audio' in analysis and len(analysis['audio']) > 0:
                # Use librosa's mel spectrogram for better visualization
                mel_spec = librosa.feature.melspectrogram(y=analysis['audio'], sr=analysis['sample_rate'])
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                # Create image item for spectrogram
                img = pg.ImageItem()
                self.spectrogram_plot.addItem(img)
                # Set the image data
                img.setImage(mel_spec_db, autoLevels=False)
                # Set the correct orientation
                img.setTransform(pg.QtGui.QTransform().scale(1, -1))
                # Set the correct position
                img.setPos(0, 0)
            
            # Update spectral features plot
            self.spectral_features_plot.clear()
            if 'spectral_features' in analysis and len(analysis['spectral_features']['centroid']) > 0:
                features = analysis['spectral_features']
                time = np.arange(len(features['centroid'])) * self.hop_length / analysis['sample_rate']
                self.spectral_features_plot.plot(time, features['centroid'], pen='r', name='Centroid')
                self.spectral_features_plot.plot(time, features['rolloff'], pen='g', name='Rolloff')
                self.spectral_features_plot.plot(time, features['flux'], pen='b', name='Flux')
                self.spectral_features_plot.addLegend()
                
        except Exception as e:
            print(f"Error updating analysis display: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to update analysis display: {str(e)}")
        
    def update_processing_visualization(self, step, data):
        self.processing_plot.clear()
        
        if step == 'noise_reduction':
            # Plot noise reduction visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude'].mean(axis=1) + 1e-10), pen='b', name='Original')
            self.processing_plot.plot(freqs, 20 * np.log10(data['noise_profile'] + 1e-10), pen='r', name='Noise Profile')
            
        elif step == 'spectral_enhancement':
            # Plot spectral enhancement visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude'].mean(axis=1) + 1e-10), pen='b', name='Original')
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude_enhanced'].mean(axis=1) + 1e-10), pen='g', name='Enhanced')
            
        elif step == 'dynamic_range':
            # Plot dynamic range compression visualization
            self.processing_plot.plot(data['audio_db'], pen='b', name='Original')
            self.processing_plot.plot(data['compressed_db'], pen='r', name='Compressed')
            
        elif step == 'phase_correction':
            # Plot phase correction visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, data['phase'].mean(axis=1), pen='b', name='Original Phase')
            self.processing_plot.plot(freqs, data['phase_smooth'].mean(axis=1), pen='r', name='Corrected Phase')
            
        elif step == 'room_correction':
            # Plot room correction visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude'].mean(axis=1) + 1e-10), pen='b', name='Original')
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude_corrected'].mean(axis=1) + 1e-10), pen='g', name='Corrected')
            
        elif step == 'adaptive_eq':
            # Plot adaptive EQ visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude'].mean(axis=1) + 1e-10), pen='b', name='Original')
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude_eq'].mean(axis=1) + 1e-10), pen='g', name='EQ Applied')
            
        elif step == 'limiter':
            # Plot limiter visualization
            self.processing_plot.plot(data['audio_db'], pen='b', name='Original')
            self.processing_plot.plot(data['limited_db'], pen='r', name='Limited')
            
        elif step == 'spectral_gating':
            # Plot spectral gating visualization
            freqs = np.fft.rfftfreq(2048, d=1/16000)
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude'].mean(axis=1) + 1e-10), pen='b', name='Original')
            self.processing_plot.plot(freqs, 20 * np.log10(data['magnitude_gated'].mean(axis=1) + 1e-10), pen='g', name='Gated')
        
    def processing_finished(self, output_path):
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self.statusBar().showMessage(f"Processing complete. Saved to: {output_path}")
        
        # Update visualization with processed audio
        self.load_audio_preview(output_path)
        
        QMessageBox.information(
            self,
            "Success",
            f"Audio processing complete!\nSaved to: {output_path}"
        )
        
    def processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self.statusBar().showMessage("Error occurred during processing")
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during processing: {error_msg}"
        )

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for modern look
    
    # Set application-wide font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = AudioEnhancerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 