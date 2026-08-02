# 🎓 Bilingual Grammar Tutor (Spanish-English)

A voice-and-text grammar correction tool built as a personal AI/ML portfolio project.

## What it does
Users type or speak a sentence in English, and the app corrects grammar
mistakes, returning both the corrected text and audio playback — designed
to help Spanish-speaking English learners practice naturally.

## How it works
- **Base model:** SmolLM2-1.7B-Instruct (open-weight, HuggingFace)
- **Fine-tuning:** LoRA/PEFT on the juancavallotti/multilingual-gec dataset,
  trained on Google Colab's free GPU
- **Speech-to-text:** OpenAI Whisper
- **Text-to-speech:** gTTS
- **Interface:** Gradio
- **Hosting:** ModelScope Studios (free CPU/GPU hosting)

## Try it live
👉 [modelscope.ai/studios/Luigiloco1/english-spanish-tutor](https://modelscope.ai/studios/Luigiloco1/english-spanish-tutor)

## Why I built this
My wife has former students and colleagues who still teach Spanish/English
learners. I wanted to build something they could actually use to practice —
and use it as a way to learn the full pipeline of fine-tuning, deployment,
and iterating based on real user feedback.

## Current limitations / next steps
- Struggles with longer, more complex sentences (small base model + light fine-tuning)
- Working on improving accuracy and adding stronger Spanish-language support

## About this project
This was built through hands-on collaboration with Claude (Anthropic), where
I directed the approach, made the design decisions, and reviewed every step —
similar to how a project lead works with a technical team.
