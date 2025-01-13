import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import os
import numpy as np

# Initialize the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to list available input devices
def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Input Device id {i} - {device_info.get('name')}")
            return i
    p.terminate()
    return num_devices

# Function to record audio from the microphone
def record_audio(filename, duration=5, sample_rate=16000, device_index=None):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, input_device_index=device_index, frames_per_buffer=1024)
    frames = []

    print("Recording...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to check if the audio file is silent
def is_silent(filename, threshold=500):
    wf = wave.open(filename, 'rb')
    frames = wf.readframes(wf.getnframes())
    wf.close()

    audio_data = np.frombuffer(frames, dtype=np.int16)
    return np.max(np.abs(audio_data)) < threshold

# List available input devices
device_index = list_input_devices()

# Record audio and save it to a file
audio_filename = "output.wav"
record_audio(audio_filename, device_index=device_index)  # Change device_index to the correct input device id

# Check if the recorded audio file is empty
if os.path.getsize(audio_filename) == 44:  # 44 bytes is the size of an empty WAV header
    print("The recorded audio file is empty.")
elif is_silent(audio_filename):
    print("The recorded audio file is all silence.")
else:
    # Load the recorded audio and perform speech recognition
    with open(audio_filename, "rb") as f:
        audio_data = f.read()

    result = pipe(audio_data, generate_kwargs={"language": "english"})
    print(result["text"])