import customtkinter as ctk
import threading
from call_monitor import CallMonitor
import time
from datetime import datetime
import queue
from PIL import Image, ImageDraw
from textblob import TextBlob
import traceback
from sentiment_analyzer import SentimentAnalyzer, SentimentVisualizer
import os
from pydub import AudioSegment
import speech_recognition as sr
import librosa
import tkinter as tk
from tkinter import filedialog

class SentimentGauge(ctk.CTkCanvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(width=200, height=100)
        self.value = 0
        self.draw_gauge()

    def draw_gauge(self):
        self.delete("all")
        # Draw background arc
        self.create_arc(10, 10, 190, 190, 
                       start=180, extent=180, 
                       fill="#2B2B2B")
        
        # Draw value arc
        extent = 180 * (self.value + 1) / 2
        color = self.get_sentiment_color(self.value)
        self.create_arc(10, 10, 190, 190, 
                       start=180, extent=extent, 
                       fill=color)
        
        # Draw center text
        text = f"{self.value:.2f}"
        self.create_text(100, 80, text=text, 
                        fill="white", font=("Helvetica", 20))

    def get_sentiment_color(self, value):
        if value > 0.5:
            return "#4CAF50"  # Strong positive - Green
        elif value > 0:
            return "#8BC34A"  # Mild positive - Light green
        elif value < -0.5:
            return "#FF5252"  # Strong negative - Red
        elif value < 0:
            return "#FF7043"  # Mild negative - Orange
        return "#FFC107"      # Neutral - Yellow

    def set_value(self, value):
        self.value = max(-1, min(1, value))
        self.draw_gauge()

class ScrollableFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

class CallMonitorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Call Conversation Monitor")
        self.geometry("1200x800")
        self.minsize(1000, 600)  # Set minimum window size
        ctk.set_appearance_mode("dark")
        
        # Initialize analyzers
        self.call_monitor = CallMonitor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_visualizer = SentimentVisualizer(self)
        self.call_monitor.analyze_text = self.analyze_text_wrapper
        self.update_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
        self.create_widgets()
        self.is_monitoring = False
        
        # Start update checker
        self.check_updates()
        
        # Configure update interval
        self.update_interval = 1000  # Update visualizations every 1 second
        self.schedule_visualization_update()

    def create_widgets(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=4)  # Left panel takes 40% of width
        self.grid_columnconfigure(1, weight=6)  # Right panel takes 60% of width
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Left frame contents
        self.create_left_frame()
        
        # Right frame contents
        self.create_right_frame()

    def create_left_frame(self):
        """Create and configure the left frame"""
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Add file upload button and status
        self.upload_frame = ctk.CTkFrame(self.left_frame)
        self.upload_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.upload_frame.grid_columnconfigure(1, weight=1)
        
        self.upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Upload Audio File",
            command=self.upload_audio_file
        )
        self.upload_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.file_label = ctk.CTkLabel(
            self.upload_frame,
            text="No file selected",
            anchor="w"
        )
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Create text area for conversation
        self.text_area = ctk.CTkTextbox(self.left_frame, height=300)
        self.text_area.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create sentiment gauge
        self.gauge_frame = ctk.CTkFrame(self.left_frame)
        self.gauge_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.gauge_frame.grid_columnconfigure(0, weight=1)
        
        self.gauge = SentimentGauge(self.gauge_frame)
        self.gauge.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create button frame
        self.button_frame = ctk.CTkFrame(self.left_frame)
        self.button_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.button_frame.grid_columnconfigure(0, weight=1)
        
        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Configure row weights
        self.left_frame.grid_rowconfigure(1, weight=3)  # Text area
        self.left_frame.grid_rowconfigure(2, weight=1)  # Gauge

    def create_right_frame(self):
        # Configure right frame grid with fixed proportions
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        
        # Create a main container frame with fixed size
        self.viz_container = ctk.CTkFrame(self.right_frame)
        self.viz_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viz_container.grid_columnconfigure(0, weight=1)
        
        # Set fixed total height for visualization container
        total_height = 800  # Total height in pixels
        
        # Create frames with fixed sizes
        self.mood_frame = ctk.CTkFrame(self.viz_container)
        self.mood_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.mood_frame.configure(height=int(total_height * 0.2))  # 20% of height
        self.mood_frame.grid_propagate(False)
        
        self.plot_frame = ctk.CTkFrame(self.viz_container)
        self.plot_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.plot_frame.configure(height=int(total_height * 0.4))  # 40% of height
        self.plot_frame.grid_propagate(False)
        
        self.emotion_frame = ctk.CTkFrame(self.viz_container)
        self.emotion_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.emotion_frame.configure(height=int(total_height * 0.4))  # 40% of height
        self.emotion_frame.grid_propagate(False)
        
        # Configure row weights
        self.viz_container.grid_rowconfigure(0, weight=2)  # Summary
        self.viz_container.grid_rowconfigure(1, weight=4)  # Trend plot
        self.viz_container.grid_rowconfigure(2, weight=4)  # Distribution

    def schedule_visualization_update(self):
        """Schedule periodic updates for visualizations"""
        if not hasattr(self, 'last_update_time'):
            self.last_update_time = time.time()
        
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:  # Update every second
            try:
                if self.is_monitoring and hasattr(self, 'call_monitor'):
                    self.update_visualizations()
                self.last_update_time = current_time
            except Exception as e:
                self.handle_error(f"Error in visualization schedule: {str(e)}")
        
        # Schedule next check
        self.after(100, self.schedule_visualization_update)

    def update_visualizations(self):
        """Update all visualizations"""
        if not hasattr(self, 'call_monitor') or not self.call_monitor.conversation_history:
            return
            
        try:
            # Update visualizations one by one with error handling
            try:
                self.sentiment_visualizer.create_tone_summary(
                    self.call_monitor.conversation_history,
                    self.mood_frame
                )
            except Exception as e:
                self.handle_error(f"Error updating tone summary: {str(e)}")
            
            try:
                self.sentiment_visualizer.create_sentiment_plot(
                    self.call_monitor.conversation_history,
                    self.plot_frame
                )
            except Exception as e:
                self.handle_error(f"Error updating sentiment plot: {str(e)}")
            
            try:
                self.sentiment_visualizer.create_emotion_distribution(
                    self.call_monitor.conversation_history,
                    self.emotion_frame
                )
            except Exception as e:
                self.handle_error(f"Error updating emotion distribution: {str(e)}")
            
            # Force update of frames
            for frame in [self.mood_frame, self.plot_frame, self.emotion_frame]:
                frame.update_idletasks()
            self.viz_container.update_idletasks()
            
        except Exception as e:
            self.handle_error(f"Error in visualization update: {str(e)}")

    def analyze_text_wrapper(self, text):
        """Wrapper for the original analyze_text method to update GUI"""
        try:
            # Perform sentiment analysis
            analysis = self.sentiment_analyzer.analyze_text(text)
            
            # Store in conversation history
            self.call_monitor.conversation_history.append(analysis)
            
            # Queue the update for the GUI
            self.update_queue.put(analysis)
            
        except Exception as e:
            self.handle_error(f"Error analyzing text: {str(e)}")

    def update_gui(self, analysis):
        try:
            # Update gauge with mood score
            self.gauge.set_value(analysis['mood_score'])
            
            # Update conversation text
            timestamp = datetime.strptime(analysis['timestamp'], '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S')
            text = f"[{timestamp}] {analysis['text']}\n"
            mood_text = f"Mood: {analysis['tone']}\n"
            emotion_text = f"Emotion: {analysis['dominant_emotion'].title()}\n"
            
            self.text_area.insert("end", text)
            self.text_area.insert("end", mood_text)
            self.text_area.insert("end", emotion_text)
            self.text_area.insert("end", "-" * 50 + "\n")
            self.text_area.see("end")
            
        except Exception as e:
            self.handle_error(f"Error updating GUI: {str(e)}")

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()

    def start_monitoring(self):
        try:
            self.is_monitoring = True
            self.start_button.configure(text="Stop Monitoring")
            self.text_area.delete("1.0", "end")
            self.call_monitor.conversation_history = []
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.run_monitoring)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
        except Exception as e:
            self.handle_error(f"Failed to start monitoring: {str(e)}")
            self.stop_monitoring()

    def run_monitoring(self):
        try:
            self.log_message("Starting audio monitoring...")
            self.call_monitor.start_monitoring()
        except Exception as e:
            self.error_queue.put(e)

    def stop_monitoring(self):
        try:
            self.is_monitoring = False
            self.call_monitor.stop_monitoring()
            self.start_button.configure(text="Start Monitoring")
            self.log_message("Monitoring stopped")
            self.update_visualizations()
        except Exception as e:
            self.handle_error(f"Error stopping monitoring: {str(e)}")

    def log_message(self, message, error=False):
        timestamp = datetime.now().strftime('%H:%M:%S')
        prefix = "[ERROR]" if error else "[INFO]"
        self.text_area.insert("end", f"[{timestamp}] {prefix} {message}\n")
        self.text_area.see("end")

    def handle_error(self, error_message):
        self.log_message(error_message, error=True)
        
    def check_updates(self):
        try:
            # Check for errors
            try:
                while True:
                    error = self.error_queue.get_nowait()
                    self.handle_error(f"Monitoring error: {str(error)}")
            except queue.Empty:
                pass
            
            # Check for updates
            try:
                while True:
                    analysis = self.update_queue.get_nowait()
                    self.update_gui(analysis)
            except queue.Empty:
                pass
                
        except Exception as e:
            self.handle_error(f"Error updating GUI: {str(e)}")
        finally:
            self.after(100, self.check_updates)

    def upload_audio_file(self):
        """Handle audio file upload"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[
                    ("Audio Files", "*.wav *.mp3 *.m4a *.ogg"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_path:
                self.file_label.configure(text=os.path.basename(file_path))
                self.process_audio_file(file_path)
        except Exception as e:
            self.handle_error(f"Error uploading file: {str(e)}")
    
    def process_audio_file(self, file_path):
        """Process uploaded audio file"""
        try:
            # Clear existing conversation history
            self.call_monitor.conversation_history = []
            self.text_area.delete(1.0, tk.END)
            
            # Convert audio to wav if needed
            if not file_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(file_path)
                wav_path = file_path + '.wav'
                audio.export(wav_path, format='wav')
                file_path = wav_path
            
            # Load the audio file
            audio = sr.AudioFile(file_path)
            
            with audio as source:
                # Analyze audio in chunks
                chunk_duration = 10  # seconds
                offset = 0
                duration = librosa.get_duration(path=file_path)
                
                while offset < duration:
                    try:
                        # Get audio chunk
                        audio_data = self.call_monitor.recognizer.record(
                            source,
                            duration=min(chunk_duration, duration - offset),
                            offset=offset
                        )
                        
                        # Recognize speech in chunk
                        text = self.call_monitor.recognizer.recognize_google(audio_data)
                        
                        if text:
                            # Process the recognized text
                            self.call_monitor.process_recognized_text(text)
                            
                            # Update display
                            self.text_area.insert(tk.END, f"[{offset:.1f}s]: {text}\n")
                            self.text_area.see(tk.END)
                            
                            # Force update visualizations
                            self.update_visualizations()
                            self.update_idletasks()
                            
                            # Small delay to show progress
                            time.sleep(0.1)
                        
                    except sr.UnknownValueError:
                        print(f"Could not understand audio at {offset}s")
                    except Exception as e:
                        print(f"Error processing chunk at {offset}s: {str(e)}")
                    
                    offset += chunk_duration
                
                # Final update of visualizations
                self.update_visualizations()
                
                # Show completion message
                self.text_area.insert(tk.END, "\n=== Audio File Analysis Complete ===\n\n")
                self.text_area.see(tk.END)
                
        except Exception as e:
            self.handle_error(f"Error processing audio file: {str(e)}")

if __name__ == "__main__":
    try:
        app = CallMonitorGUI()
        app.mainloop()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc() 