import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import whisper
from gtts import gTTS
import tempfile

# ---------- Cargar el modelo de corrección gramatical ----------
base_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
adapter_name = "LuigiGuzman/english-spanish-grammar-tutor"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, adapter_name)

# ---------- Cargar Whisper (voz -> texto), modelo pequeño para CPU ----------
whisper_model = whisper.load_model("tiny")

# ---------- Función: corregir texto ----------
def correct_text(text):
    inputs = tokenizer(f"fix grammar: {text}", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# ---------- Función: texto -> voz ----------
def text_to_speech(text, lang="es"):
    tts = gTTS(text=text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# ---------- Flujo con texto ----------
def process_text(text):
    corrected = correct_text(text)
    audio_path = text_to_speech(corrected)
    return corrected, audio_path

# ---------- Flujo con voz ----------
def process_voice(audio_path):
    transcription = whisper_model.transcribe(audio_path)["text"]
    corrected = correct_text(transcription)
    audio_out = text_to_speech(corrected)
    return transcription, corrected, audio_out

# ---------- Interfaz Gradio ----------
with gr.Blocks(title="Tutor Bilingüe Español-Inglés") as demo:
    gr.Markdown("# 🎓 Tutor de Gramática Español-Inglés")
    gr.Markdown("Escribe o habla una frase, y te ayudo a corregirla — con voz incluida.")

    with gr.Tab("✍️ Texto"):
        text_input = gr.Textbox(label="Escribe tu frase")
        text_button = gr.Button("Corregir")
        text_output = gr.Textbox(label="Frase corregida")
        text_audio = gr.Audio(label="Escúchala")
        text_button.click(process_text, inputs=text_input, outputs=[text_output, text_audio])

    with gr.Tab("🎙️ Voz"):
        voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Habla aquí")
        voice_button = gr.Button("Corregir")
        voice_transcription = gr.Textbox(label="Lo que escuché")
        voice_output = gr.Textbox(label="Frase corregida")
        voice_audio = gr.Audio(label="Escúchala")
        voice_button.click(process_voice, inputs=voice_input, outputs=[voice_transcription, voice_output, voice_audio])

if __name__ == "__main__":
    demo.launch()
