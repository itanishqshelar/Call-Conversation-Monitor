import sounddevice as sd
import numpy as np
import speech_recognition as sr
from textblob import TextBlob
import queue
import threading
import time
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

class CallMonitor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.is_running = False
        self.sample_rate = 16000
        self.block_duration = 5  # seconds
        self.channels = 1  # Changed to mono for better compatibility
        
        # Sentiment analysis thresholds
        self.POSITIVE_THRESHOLD = 0.3
        self.NEGATIVE_THRESHOLD = -0.3
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Print available audio devices for debugging
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        print(f"\nDefault Input Device: {sd.query_devices(kind='input')}")
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function to handle audio data"""
        try:
            if status:
                print(f"Status: {status}")
            self.audio_queue.put(indata.copy())
        except Exception as e:
            print(f"Error in audio callback: {e}")

    def process_audio(self):
        """Process audio data from the queue"""
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                
                # Ensure audio data is not empty and contains valid values
                if audio_data is None or len(audio_data) == 0:
                    continue
                    
                # Convert to mono if necessary
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_mono = np.mean(audio_data, axis=1)
                else:
                    audio_mono = audio_data.flatten()
                
                # Normalize audio data
                audio_mono = np.clip(audio_mono, -1, 1)
                
                # Convert numpy array to audio data for speech recognition
                audio_bytes = (audio_mono * 32767).astype(np.int16).tobytes()
                audio_source = sr.AudioData(audio_bytes, self.sample_rate, 2)
                
                try:
                    print("Recognizing speech...")
                    text = self.recognizer.recognize_google(audio_source)
                    print(f"Recognized text: {text}")
                    if text.strip():
                        self.analyze_text(text)
                except sr.UnknownValueError:
                    print("Speech not recognized")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")

    def analyze_text(self, text):
        """Analyze the sentiment and behavior in the text"""
        try:
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            subjectivity_score = blob.sentiment.subjectivity
            
            # Determine sentiment category
            if sentiment_score >= self.POSITIVE_THRESHOLD:
                sentiment_category = "Positive"
            elif sentiment_score <= self.NEGATIVE_THRESHOLD:
                sentiment_category = "Negative"
            else:
                sentiment_category = "Neutral"
                
            # Store in conversation history
            analysis_result = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'text': text,
                'sentiment': sentiment_category,
                'sentiment_score': round(sentiment_score, 2),
                'subjectivity': round(subjectivity_score, 2)
            }
            
            self.conversation_history.append(analysis_result)
            
            # Print analysis results
            print("\n=== Speech Analysis ===")
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment_category} (Score: {sentiment_score:.2f})")
            print(f"Subjectivity: {subjectivity_score:.2f}")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error analyzing text: {e}")

    def start_monitoring(self):
        """Start monitoring the audio"""
        try:
            self.is_running = True
            
            # Start audio processing thread
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.daemon = True
            processing_thread.start()
            
            print("Initializing audio stream...")
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * self.block_duration),
                device=None  # Use default input device
            ) as stream:
                print("Audio stream initialized successfully")
                print("Call monitoring started. Press Ctrl+C to stop.")
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            self.stop_monitoring()
            self.display_summary()
            
        except Exception as e:
            print(f"\nError starting monitoring: {e}")
            print("Audio device details:")
            print(sd.query_devices())
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the monitoring process"""
        print("Stopping monitoring...")
        self.is_running = False

    def display_summary(self):
        """Display a summary of the conversation analysis"""
        if not self.conversation_history:
            print("No conversation data recorded.")
            return
            
        total_entries = len(self.conversation_history)
        sentiment_counts = {
            'Positive': sum(1 for entry in self.conversation_history if entry['sentiment'] == 'Positive'),
            'Neutral': sum(1 for entry in self.conversation_history if entry['sentiment'] == 'Neutral'),
            'Negative': sum(1 for entry in self.conversation_history if entry['sentiment'] == 'Negative')
        }
        
        print("\n=== Conversation Summary ===")
        print(f"Total speech segments analyzed: {total_entries}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_entries) * 100
            print(f"{sentiment}: {count} ({percentage:.1f}%)")
        
        # Calculate average sentiment and subjectivity
        avg_sentiment = sum(entry['sentiment_score'] for entry in self.conversation_history) / total_entries
        avg_subjectivity = sum(entry['subjectivity'] for entry in self.conversation_history) / total_entries
        
        print(f"\nAverage Sentiment Score: {avg_sentiment:.2f}")
        print(f"Average Subjectivity: {avg_subjectivity:.2f}")
        print("=" * 50)

if __name__ == "__main__":
    monitor = CallMonitor()
    monitor.start_monitoring() 