import gradio as gr
import numpy as np

def greet(name):
    return (16000, np.random.randn(16000))

iface = gr.Interface(fn=greet, inputs="text", outputs="audio")
iface.launch()