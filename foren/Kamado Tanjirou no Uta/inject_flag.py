#!/usr/bin/env python3
"""
inject_flag.py
Menyisipkan teks flag ke dalam audio sehingga terlihat di spectrogram.

Requirements:
  pip install numpy librosa pillow soundfile pydub tqdm
  ffmpeg harus terinstal (untuk pydub membuka mp3)
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

FLAG_TEXT = "LYCORIS{1n1_4d4L4h_B4s1C_4Ud10}"
INPUT_FILE = "chall.mp3"
OUTPUT_FILE = "chall.wav"
START_TIME = 5.0   # detik mulai inject
DURATION = 6.0     # panjang flag dalam detik
SR = 22050         # sample rate
GAIN_DB = -12.0    # makin negatif makin samar

def text_to_image(text, width, height, font_size=40):
    """Render teks ke gambar hitam putih"""
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - w) // 2, (height - h) // 2), text, fill=255, font=font)
    return np.array(img)

def image_to_mag(img, n_fft=2048, hop_length=512, max_amp=8.0):
    """Konversi gambar ke magnitude spectrogram"""
    freq_bins = n_fft // 2 + 1
    pil = Image.fromarray(img)
    pil = pil.resize((img.shape[1], freq_bins))
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = np.flipud(arr)
    return arr * max_amp

def mag_to_audio(mag, n_fft=2048, hop_length=512, n_iter=80):
    """Rekonstruksi audio dari magnitude dengan Griffin-Lim"""
    S_power = mag**2
    # versi baru librosa tidak pakai argumen power lagi
    y = librosa.griffinlim(S_power, 
                           n_iter=n_iter,
                           hop_length=hop_length,
                           win_length=n_fft,
                           window="hann")
    return y

def make_flag_audio(flag_text, sr=22050, duration=6.0):
    """Buat potongan audio berisi flag"""
    n_fft = 2048
    hop_length = 512
    time_frames = int(np.ceil(duration * sr / hop_length))
    img = text_to_image(flag_text, time_frames, 256, font_size=32)
    mag = image_to_mag(img, n_fft=n_fft, hop_length=hop_length)
    y = mag_to_audio(mag, n_fft=n_fft, hop_length=hop_length, n_iter=80)
    y = y / (np.max(np.abs(y)) + 1e-9)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def main():
    print("[+] Membuat stego audio dari flag...")
    stego, sr = make_flag_audio(FLAG_TEXT, sr=SR, duration=DURATION)

    print("[+] Membuka host audio...")
    host = AudioSegment.from_file(INPUT_FILE)

    # simpan flag sementara
    tmp_wav = "._tmp_flag.wav"
    sf.write(tmp_wav, stego, sr, subtype="PCM_16")
    stego_seg = AudioSegment.from_wav(tmp_wav) + GAIN_DB
    os.remove(tmp_wav)

    print("[+] Overlay stego ke host...")
    mixed = host.overlay(stego_seg, position=int(START_TIME * 1000))

    print("[+] Menyimpan ke chall.wav ...")
    mixed.export(OUTPUT_FILE, format="wav")

    print("[+] Selesai! Cek spectrogram di", OUTPUT_FILE)

if __name__ == "__main__":
    main()

