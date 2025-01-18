import os
from text_to_speech import text_to_speech
import pyaudio
import wave
import time
import io
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import os
import numpy as np
import noisereduce as nr
from text_to_text import generate_text_from_prompt
# Settings
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate
THRESHOLD = 500  # Silence threshold (adjust as needed)
SILENCE_DURATION = 2  # Seconds of silence to stop recording

    
# Initialize the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def is_silent(data):
    """Check if the audio data is below the silence threshold."""
    return max(data) < THRESHOLD

def reduce_noise(audio_data, rate):
    """Reduce noise from the audio data."""
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate)
    return reduced_noise

def audio_to_text(audio_stream):
    """Process the audio stream. Replace this with your desired functionality."""
    print("Audio stream received for processing.")
    # Example: Just print the length of the audio stream
    audio_stream.seek(0)  # Reset the buffer to the beginning
    print(f"Audio length: {len(audio_stream.getvalue())} bytes")
    # Convert audio stream to numpy array
    audio_data = np.frombuffer(audio_stream.getvalue(), dtype=np.int16)
    
    # Reduce noise
    audio_data = reduce_noise(audio_data, RATE)
    # save audio data to a file post.mp3
    wf = wave.open("post.mp3", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(audio_data.tobytes())
    wf.close()
    # Perform speech-to-text
    result = pipe("post.mp3", generate_kwargs={"language": "en"})
    print('Speech to text result:')
    print(result["text"])
    return result["text"]

def record_until_silent():
    """Record audio until the speaker stops speaking or 10 seconds have passed."""
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

        # Check for silence after the first second
        if time.time() - start_time > 1:
            if is_silent(audio_data):
                silent_chunks += 1
            else:
                silent_chunks = 0

            # Stop recording after prolonged silence or 10 seconds
            if silent_chunks > (SILENCE_DURATION * RATE // CHUNK):
                print("Silence detected. Stopping recording.")
                break
        if time.time() - start_time > 10:
            print("Maximum recording time reached. Stopping recording.")
            break

    # Close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Convert frames to BytesIO
    audio_stream = io.BytesIO(b''.join(frames))
    return audio_stream
import time
from playsound import playsound

# Example usage
if __name__ == "__main__":
    counter = 0
    while True:
        start_time = time.time()
        audio_stream = record_until_silent()
        record_time = time.time()
        print(f"record_until_silent took {record_time - start_time:.2f} seconds")

        input_text = audio_to_text(audio_stream)
        audio_to_text_time = time.time()
        print(f"audio_to_text took {audio_to_text_time - record_time:.2f} seconds")

        output_text = generate_text_from_prompt(input_text)
        generate_text_time = time.time()
        print(f"generate_text_from_prompt took {generate_text_time - audio_to_text_time:.2f} seconds")
        output_file = f"output_{counter}.mp3"
        output_path = text_to_speech(output_text,output_file)
        if counter >=3:
            counter = 0
        text_to_speech_time = time.time()
        print(f"text_to_speech took {text_to_speech_time - generate_text_time:.2f} seconds")
        # play the audio file with python
        playsound(output_path)
        # wait until the audio is finished playing
        print(f"Output audio played from {output_path}")