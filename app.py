import gradio as gr
import numpy as np
from audioldm import text_to_audio, seed_everything, build_model

audioldm = build_model()

def text2audio(text, duration, guidance_scale, random_seed):
    # print(text, length, guidance_scale)
    waveform = text_to_audio(audioldm, text, random_seed, duration=duration, guidance_scale=guidance_scale, n_candidate_gen_per_text=1) # [bs, 1, samples]
    waveform = [(16000, wave[0]) for wave in waveform]
    # waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    return waveform

# iface = gr.Interface(fn=text2audio, inputs=[
#         gr.Textbox(value="A man is speaking in a huge room", max_lines=1),
#         gr.Slider(2.5, 10, value=5, step=2.5),
#         gr.Slider(0, 5, value=2.5, step=0.5),
#         gr.Number(value=42)
#     ], outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")],
#                 allow_flagging="never"
#                      )
# iface.launch(share=True)

iface = gr.Blocks()

with iface:
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
            ############# Input
            textbox = gr.Textbox(value="A hammer is hitting a wooden surface", max_lines=1)
            duration = gr.Slider(2.5, 10, value=5, step=2.5)
            guidance_scale = gr.Slider(0, 5, value=2.5, step=0.5)
            seed = gr.Number(value=42)
            ############# Output
            outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")]
            
            btn = gr.Button("Submit").style(full_width=True)
        btn.click(text2audio, inputs=[textbox, duration, guidance_scale, seed], outputs=outputs) 
        gr.HTML('''
        <hr>
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Model by <a href="https://haoheliu.github.io/" style="text-decoration: underline;" target="_blank">Haohe Liu</a>
                    </p>
        </div>
        ''')

iface.queue(concurrency_count=2)
iface.launch(debug=True, share=True)