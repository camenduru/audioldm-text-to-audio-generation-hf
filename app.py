import gradio as gr
import numpy as np
from audioldm import text_to_audio

def greet(text):
    waveform = text_to_audio(text, n_gen=1) # [bs, 1, samples]
    waveform = [(16000, wave[0]) for wave in waveform]
    return waveform

iface = gr.Interface(fn=greet, inputs="text", outputs=["audio", "audio"])
iface.launch()

# if __name__ == "__main__":
#     greet("hello world")
