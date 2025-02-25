import pyaudio
import numpy as np
import wave
import time
import threading

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MBartForConditionalGeneration, MBart50Tokenizer  # インポートを追加
import torch
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import shutil
import os

import noisereduce as nr  # ノイズ除去のライブラリをインポート

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAITSECONDS = 1
INPUT_DEVICE_INDEX = 0

recode_finish_flag = False
model = None
pipe = None
translation_model = None
translation_model_name = 'facebook/mbart-large-50-many-to-many-mmt'
translation_tokenizer = None
output_folder_path = "./output_folder"
frames_queue = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # AngularアプリのURLを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_whisper_model(model, save_path: str):
    model.save_pretrained(save_path)
    print(f"Whisper モデルを {save_path} に保存しました。")

def save_whisper_processor(processor, save_path: str):
    processor.save_pretrained(save_path)
    print(f"Whisper プロセッサーを {save_path} に保存しました。")

def load_whisper_model(model_path: str, device: str = 'cpu'):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def init_whisper_model():
    global model, pipe
    # model_id = "kotoba-tech/kotoba-whisper-v1.0"
    model_id = "kotoba-tech/kotoba-whisper-v2.0"
    cache_dir = 'model/whisper'
    os.makedirs(cache_dir, exist_ok=True)
    custom_model_path = os.path.join(cache_dir, model_id.replace('/', '_'))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if os.path.isdir(custom_model_path):
        # ディレクトリ内に保存されたモデルがあるか確認
        files = [f for f in os.listdir(custom_model_path) if os.path.isfile(os.path.join(custom_model_path, f))]
        if len(files) > 0:
            # モデルをロード
            model, processor = load_whisper_model(custom_model_path, device)
        else:
            # 新しいモデルをロードして保存
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            save_whisper_model(model, custom_model_path)
            save_whisper_processor(processor, custom_model_path)
    else:
        # ディレクトリが存在しない場合、新しいモデルをロードして保存
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        save_whisper_model(model, custom_model_path)
        save_whisper_processor(processor, custom_model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

def load_translation_model(model_path: str, device: str = 'cpu'):
    translation_model = MBartForConditionalGeneration.from_pretrained(model_path)
    translation_model.to(device)
    translation_tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    return translation_model, translation_tokenizer

def save_translation_model(model, save_path: str):
    translation_model.save_pretrained(save_path)
    print(f"翻訳モデルを {save_path} に保存しました。")

def save_translation_tokenizer(tokenizer, save_path: str):
    translation_tokenizer.save_pretrained(save_path)
    print(f"トークナイザーを {save_path} に保存しました。")

def init_translation_model():
    global translation_model
    global translation_tokenizer
    cache_dir = 'model/translation'
    os.makedirs(cache_dir, exist_ok=True)
    custom_model_path = os.path.join(cache_dir, translation_model_name)
    custom_tokenizer_path = os.path.join(cache_dir, translation_model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.isdir(custom_model_path):
        # ディレクトリ内のファイルリストを取得
        files = [f for f in os.listdir(custom_model_path) if os.path.isfile(os.path.join(custom_model_path, f))]
        if len(files) > 0:
            translation_model, translation_tokenizer = load_translation_model(custom_model_path, device)
        else:
            translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_name)
            translation_model.to(device)
            translation_tokenizer = MBart50Tokenizer.from_pretrained(translation_model_name)
            save_translation_model(translation_model_name, custom_model_path)
            save_translation_tokenizer(translation_model_name, custom_tokenizer_path)
    else:
        translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_name)
        translation_model.to(device)
        translation_tokenizer = MBart50Tokenizer.from_pretrained(translation_model_name)
        save_translation_model(translation_model_name, custom_model_path)
        save_translation_tokenizer(translation_model_name, custom_tokenizer_path)

def model_init():
    init_whisper_model()
    init_translation_model()
    
def transcription_record(uploadFile):
    global pipe
    result = pipe(uploadFile)
    japanese_text = result["text"]
    english_text = translation(japanese_text)
    return japanese_text, english_text
def translation(japanese_text: str):
    global translation_model
    global translation_tokenizer
    translation_tokenizer.src_lang = "ja_XX"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_text = translation_tokenizer(japanese_text, return_tensors="pt").to(device)
    generated_tokens = translation_model.generate(**encoded_text, forced_bos_token_id=translation_tokenizer.lang_code_to_id["en_XX"])
    english_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return english_text

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# ノイズ除去を追加した音声処理関数
def process_audio(file_path: str):
    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(file_path)
    
    # 音声データをnumpy配列に変換（ノイズ除去にはnumpyが必要）
    samples = np.array(audio.get_array_of_samples())

    # ノイズ除去を適用
    reduced_noise_samples = nr.reduce_noise(y=samples, sr=audio.frame_rate)

    # ノイズ除去後のデータをAudioSegmentに変換
    reduced_noise_audio = AudioSegment(
        reduced_noise_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels,
    )
    
    # ノイズ除去後の音声を分割
    segments = split_on_silence(reduced_noise_audio, silence_thresh=-40)  # しきい値の調整が必要

    # 各セグメントを一時ファイルとして保存
    segment_paths = []
    for i, segment in enumerate(segments):
        segment_path = f"segment_{i}.wav"
        segment.export(segment_path, format="wav")
        segment_paths.append(segment_path)

    return segment_paths

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # 一時ファイルのパスを指定
    temp_file_path = f"temp_{file.filename}"

    try:
        # UploadFile の内容を一時ファイルとして保存
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ファイルが保存できたことを確認するために、ファイルパスを表示
        print(f"File saved to {temp_file_path}")

        # 音声ファイルを処理
        segment_paths = process_audio(temp_file_path)
        transcription_results = []
        translation_results = []

        # 各セグメントを処理
        for segment_path in segment_paths:
            japanese_text, english_text = transcription_record(segment_path)
            transcription_results.append(japanese_text)
            translation_results.append(english_text)

        # 結果をまとめる
        transcription_text = " ".join(transcription_results)
        translation_text = " ".join(translation_results)

        return JSONResponse(content={"transcription": transcription_text, "translation": translation_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        for segment_path in segment_paths:
            if os.path.exists(segment_path):
                os.remove(segment_path)

# if __name__ == '__main__':
model_init()
