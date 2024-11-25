import pyaudio
import numpy as np
import pvrhino
CHUNK = 512 #number of frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1 #number of audio channel
RATE = 16000 #sample rate

p = pyaudio.PyAudio()
audio_buffers = [] #store the audio buffer

#Rhino Speech-to-Intent Engine
#github: https://github.com/Picovoice/rhino
#picovoice: https://console.picovoice.ai/
rhino = pvrhino.create(
    access_key="42hU5EeyIr72qiaGQ930V12ruOqaZhmPLyh9gDxPIZ0oK39h5czFlw==", #my key for the picovoice
    context_path="Robot_en_windows_v3_0_0.rhn", #model to convert speech to intent, **change the path**
    sensitivity = 0.5,
    endpoint_duration_sec = 1)

#audio input is being open
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

#store the audio frame in frames[]
try:
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_buffers.append(audio_data)
except KeyboardInterrupt:
    print("Stopped by user")
#audio input is being closed
stream.stop_stream()
stream.close()
p.terminate()

#process the audio buffers and get inferences
inferences = []
num_buffer = len(audio_buffers)
for i in range(num_buffer):
    frames = audio_buffers[i * rhino.frame_length:(i + 1) * rhino.frame_length]
    frames = audio_buffers[i]
    is_finalized = rhino.process(frames)
    if is_finalized:
        inference = rhino.get_inference() #get inference
        print(inference)
        if inference.is_understood:
            inferences.append(inference)
rhino.delete()
#print the inference
for i in range(len(inferences)):
    print("%d" %(i+1))
    for slot, value in inferences[i].slots.items():
        print("    %s : '%s'" % (slot, value))