# Real-time Call Conversation Monitor



https://github.com/user-attachments/assets/8dd97544-1483-4eda-bf29-b1043c525697


This Python application monitors and analyzes call conversations in real-time, providing sentiment analysis and behavioral insights from the audio input and output of your device.

## Features

- Real-time audio capture from system audio
- Speech-to-text conversion
- Sentiment analysis of conversations
- Real-time feedback on conversation tone and behavior
- Conversation summary with sentiment distribution
- Subjectivity analysis
- Timestamp tracking for all analyzed segments

## Requirements

- Python 3.7+
- Working microphone and audio output
- Internet connection (for speech recognition)

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python call_monitor.py
```

2. The application will start monitoring your audio input and output
3. Speak or play audio through your device
4. The application will display real-time analysis of the conversation
5. Press Ctrl+C to stop monitoring and display a summary

## Analysis Metrics

- **Sentiment Score**: Ranges from -1 (very negative) to 1 (very positive)
- **Subjectivity**: Ranges from 0 (very objective) to 1 (very subjective)
- **Sentiment Categories**: Positive, Neutral, or Negative
##Screenshots
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/3cd13f76-d3b6-4764-afd0-37eb819eb6c9" />

## Notes

- The application requires a stable internet connection for speech recognition
- Audio quality affects the accuracy of speech recognition
- Make sure your system's audio input/output devices are properly configured
- The application processes audio in 5-second blocks by default 
