import pyaudio
import wave
import time
import io

# Settings
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate
THRESHOLD = 500  # Silence threshold (adjust as needed)
SILENCE_DURATION = 2  # Seconds of silence to stop recording

def is_silent(data):
    """Check if the audio data is below the silence threshold."""
    return max(data) < THRESHOLD

def text2text(audio_stream):
    """Process the audio stream. Replace this with your desired functionality."""
    print("Audio stream received for processing.")
    # Example: Just print the length of the audio stream
    audio_stream.seek(0)  # Reset the buffer to the beginning
    print(f"Audio length: {len(audio_stream.getvalue())} bytes")
    # Add your audio processing logic here

def record_until_silent():
    """Record audio until the speaker stops speaking and pass it to a function."""
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Speak now!")
    frames = []
    silent_chunks = 0
    start_time = time.time()

    while True:
        # Read audio data
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = wave.struct.unpack(f"{CHUNK}h", data)
        frames.append(data)

        # Check for silence
        if is_silent(audio_data):
            silent_chunks += 1
        else:
            silent_chunks = 0

        # Stop recording after prolonged silence
        if silent_chunks > (SILENCE_DURATION * RATE // CHUNK):
            print("Silence detected. Stopping recording.")
            break

    # Close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to an in-memory stream
    audio_stream = io.BytesIO()
    with wave.open(audio_stream, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    # Pass the audio stream to the function
    text2text(audio_stream)

if __name__ == "__main__":
    record_until_silent()
