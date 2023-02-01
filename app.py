import gradio as gr
import numpy as np
# from audioldm import text_to_audio

def text2audio(text, length):
    # waveform = text_to_audio(text, n_gen=1) # [bs, 1, samples]
    # waveform = [(16000, wave[0]) for wave in waveform]
    waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    return waveform

# iface = gr.Interface(fn=greet, inputs="text", outputs=["audio", "audio"])
# iface.launch()


block = gr.Blocks()

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Text-to-Audio Generation with AudioLDM
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                <a href="https://arxiv.org/abs/2301.12503">[Paper]</a>  <a href="https://audioldm.github.io/">[Project page]</a>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            textbox = gr.Textbox(value="A man is speaking in a huge room")
            length = gr.Slider(1.0, 30.0, value=5.0, step=0.5, label="Audio length in seconds")
            # model = gr.Dropdown(choices=["harmonai/maestro-150k"], value="harmonai/maestro-150k",type="value", label="Model")
            out = [gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")]
            btn = gr.Button("Submit").style(full_width=True)
        
        btn.click(text2audio, inputs=[textbox, length], outputs=out) 
        gr.HTML('''
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Model by <a href="https://haoheliu.github.io/" style="text-decoration: underline;" target="_blank">Haohe Liu</a>
                    </p>
        </div>
        ''')

block.launch(debug=True)