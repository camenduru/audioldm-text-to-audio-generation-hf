import gradio as gr
import numpy as np
from audioldm import text_to_audio, build_model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

audioldm = build_model()
# audioldm=None

# def predict(input, history=[]):
#     # tokenize the new input sentence
#     new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

#     # generate a response 
#     history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

#     # convert the tokens to text, and then split the responses into lines
#     response = tokenizer.decode(history[0]).split("<|endoftext|>")
#     response = [(response[i], response[i+1]) for i in range(0, len(response)-1, 2)]  # convert to tuples of list
#     return response, history
  
def text2audio(text, duration, guidance_scale, random_seed, n_candidates):
    # print(text, length, guidance_scale)
    waveform = text_to_audio(audioldm, text, random_seed, duration=duration, guidance_scale=guidance_scale, n_candidate_gen_per_text=int(n_candidates)) # [bs, 1, samples]
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

            with gr.Accordion("Click to change detailed configurations", open=False):
              seed = gr.Number(value=42, label="Change this value (any integer number) will lead to a different generation result.")
              duration = gr.Slider(2.5, 10, value=5, step=2.5, label="Duration (seconds)")
              guidance_scale = gr.Slider(0, 5, value=2.5, step=0.5, label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)")
              n_candidates = gr.Slider(1, 5, value=3, step=1, label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation")
            ############# Output
            outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")]
            
            btn = gr.Button("Submit").style(full_width=True)
        btn.click(text2audio, inputs=[textbox, duration, guidance_scale, seed, n_candidates], outputs=outputs) 
        gr.HTML('''
        <hr>
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Model by <a href="https://haoheliu.github.io/" style="text-decoration: underline;" target="_blank">Haohe Liu</a>
                    </p>
        </div>
        ''')

iface.queue(concurrency_count=2)
iface.launch(debug=True)
# iface.launch(debug=True, share=True)