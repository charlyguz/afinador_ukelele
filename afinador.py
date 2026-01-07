"""
Afinador de Ukelele con Interfaz Gráfica
Usa FFT + Harmonic Product Spectrum (HPS)

Requirements:
  pip install numpy scipy sounddevice
"""

import copy
import threading
import numpy as np
import scipy.fftpack
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime

SAMPLE_FREQ = 48000
WINDOW_SIZE = 32768
WINDOW_STEP = 8192
NUM_HPS = 5
POWER_THRESH = 1e-6
WHITE_NOISE_THRESH = 0.20
CONCERT_PITCH = 440.0

SMOOTH_ALPHA = 0.25
STABLE_FRAMES = 3

LOW_G = False

UKULELE_TARGETS = {
    ("G4" if not LOW_G else "G3"): (392.00 if not LOW_G else 196.00),
    "C4": 261.63,
    "E4": 329.63,
    "A4": 440.00,
}

DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
HANN_WINDOW = np.hanning(WINDOW_SIZE)


class UkuleleTuner:
    def __init__(self, log_callback=None):
        self.window_samples = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.stable_buffer = []
        self.smooth_freq = None
        self.is_running = False
        self.stream = None
        self.log_callback = log_callback
        
        self.current_string = "---"
        self.detected_freq = 0.0
        self.target_freq = 0.0
        self.cents = 0.0
        self.status = "ESPERANDO"
        self.is_stable = False
        self.signal_level = 0.0
        
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        
    def cents_error(self, f_detected, f_target):
        if f_target <= 0:
            return 0.0
        return 1200.0 * np.log2(f_detected / f_target)
    
    def find_closest_string(self, f_detected):
        best = None
        for s, f_t in UKULELE_TARGETS.items():
            c = self.cents_error(f_detected, f_t)
            score = abs(c)
            if best is None or score < best[0]:
                best = (score, s, f_t, c)
        _, s, f_t, c = best
        return s, f_t, c
    
    def format_status(self, cents):
        if abs(cents) <= 5:
            return "AFINADO"
        return "AGUDO" if cents > 0 else "GRAVE"
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"⚠️ Audio status: {status}")
        
        x = indata[:, 0]
        if not np.any(x):
            return
        
        self.window_samples = np.concatenate((self.window_samples, x)).astype(np.float32)
        self.window_samples = self.window_samples[len(x):]
        
        signal_power = (np.linalg.norm(self.window_samples, ord=2) ** 2) / len(self.window_samples)
        self.signal_level = signal_power
        
        if signal_power < POWER_THRESH:
            if self.status != "SEÑAL BAJA":
                self.log("🔇 Señal de audio muy baja")
            self.status = "SEÑAL BAJA"
            self.current_string = "---"
            self.is_stable = False
            return
        
        hann_samples = self.window_samples * HANN_WINDOW
        magnitude_spec = np.abs(scipy.fftpack.fft(hann_samples)[:WINDOW_SIZE // 2])
        
        cutoff_bins = int(62 / DELTA_FREQ)
        magnitude_spec[:cutoff_bins] = 0.0
        
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            ind_end = min(ind_end, len(magnitude_spec))
            if ind_end <= ind_start + 1:
                continue
            
            band = magnitude_spec[ind_start:ind_end]
            avg_energy = (np.linalg.norm(band, ord=2) ** 2) / (ind_end - ind_start)
            avg_energy = np.sqrt(avg_energy)
            
            thresh = WHITE_NOISE_THRESH * avg_energy
            band[band < thresh] = 0.0
            magnitude_spec[ind_start:ind_end] = band
        
        ipol_x = np.arange(0, len(magnitude_spec))
        ipol_x2 = np.arange(0, len(magnitude_spec), 1 / NUM_HPS)
        mag_spec_ipol = np.interp(ipol_x2, ipol_x, magnitude_spec)
        
        norm = np.linalg.norm(mag_spec_ipol, ord=2)
        if norm > 0:
            mag_spec_ipol = mag_spec_ipol / norm
        
        hps_spec = copy.deepcopy(mag_spec_ipol)
        
        for factor in range(2, NUM_HPS + 1):
            decimated = mag_spec_ipol[::factor]
            limit = min(len(hps_spec), len(decimated))
            tmp = hps_spec[:limit] * decimated[:limit]
            if not np.any(tmp):
                break
            hps_spec = tmp
        
        max_ind = int(np.argmax(hps_spec))
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
        
        if not (150.0 <= max_freq <= 600.0):
            if self.status != "FUERA DE RANGO":
                self.log(f"⚠️ Frecuencia fuera de rango: {max_freq:.2f} Hz")
            self.status = "FUERA DE RANGO"
            self.current_string = "---"
            self.is_stable = False
            return
        
        if self.smooth_freq is None:
            self.smooth_freq = max_freq
        else:
            self.smooth_freq = (1 - SMOOTH_ALPHA) * max_freq + SMOOTH_ALPHA * self.smooth_freq
        
        f = float(self.smooth_freq)
        string_name, f_target, cents = self.find_closest_string(f)
        status_txt = self.format_status(cents)
        
        self.stable_buffer.insert(0, string_name)
        self.stable_buffer = self.stable_buffer[:STABLE_FRAMES]
        stable = (self.stable_buffer.count(self.stable_buffer[0]) == len(self.stable_buffer))
        
        if stable and string_name != self.current_string:
            self.log(f"🎵 Cuerda detectada: {string_name} ({f:.2f} Hz)")
        
        if stable and status_txt == "AFINADO" and self.status != "AFINADO":
            self.log(f"✅ ¡{string_name} está afinado! ({cents:+.1f} cents)")
        
        self.current_string = string_name
        self.detected_freq = f
        self.target_freq = f_target
        self.cents = cents
        self.status = status_txt
        self.is_stable = stable
    
    def start(self):
        if not self.is_running:
            try:
                self.log("🎤 Iniciando captura de audio...")
                self.is_running = True
                self.stream = sd.InputStream(
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=WINDOW_STEP,
                    samplerate=SAMPLE_FREQ,
                    dtype="float32",
                )
                self.stream.start()
                self.log(f"✅ Audio iniciado (Sample Rate: {SAMPLE_FREQ} Hz)")
                return True
            except Exception as e:
                self.is_running = False
                self.log(f"❌ Error al iniciar audio: {e}")
                return False
        return True
    
    def stop(self):
        if self.is_running:
            self.log("⏹️ Deteniendo afinador...")
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.reset()
            self.log("✅ Afinador detenido")
    
    def reset(self):
        self.window_samples = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.stable_buffer = []
        self.smooth_freq = None
        self.current_string = "---"
        self.detected_freq = 0.0
        self.target_freq = 0.0
        self.cents = 0.0
        self.status = "ESPERANDO"
        self.is_stable = False
        self.signal_level = 0.0


class TunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎸 Afinador de Ukelele FFT+HPS")
        self.root.geometry("600x900")
        self.root.resizable(False, False)
        self.root.configure(bg="#667eea")
        
        self.tuner = UkuleleTuner(log_callback=self.add_log)
        
        self.create_widgets()
        self.update_display()
        self.add_log("📱 Afinador iniciado. Presiona INICIAR para comenzar.")
        
    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#667eea")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title = tk.Label(main_frame, text="🎸 Afinador de Ukelele", 
                        font=("Helvetica", 28, "bold"), 
                        bg="#667eea", fg="white")
        title.pack(pady=(0, 5))
        
        subtitle = tk.Label(main_frame, text="FFT + Harmonic Product Spectrum", 
                           font=("Helvetica", 10), 
                           bg="#667eea", fg="white")
        subtitle.pack(pady=(0, 20))
        
        display_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=3)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.string_label = tk.Label(display_frame, text="---", 
                                     font=("Helvetica", 80, "bold"), 
                                     bg="white", fg="#667eea")
        self.string_label.pack(pady=20)
        
        freq_frame = tk.Frame(display_frame, bg="white")
        freq_frame.pack(pady=10)
        
        self.freq_label = tk.Label(freq_frame, text="Detectada: --- Hz", 
                                   font=("Helvetica", 14), 
                                   bg="white", fg="#333")
        self.freq_label.pack()
        
        self.target_label = tk.Label(freq_frame, text="Objetivo: --- Hz", 
                                     font=("Helvetica", 14), 
                                     bg="white", fg="#333")
        self.target_label.pack()
        
        meter_frame = tk.Frame(display_frame, bg="white")
        meter_frame.pack(pady=20, padx=30, fill=tk.X)
        
        self.meter_canvas = tk.Canvas(meter_frame, height=100, bg="#f0f0f0", 
                                     highlightthickness=0)
        self.meter_canvas.pack(fill=tk.X)
        
        self.cents_label = tk.Label(display_frame, text="0 cents", 
                                   font=("Helvetica", 24, "bold"), 
                                   bg="white", fg="#666")
        self.cents_label.pack(pady=10)
        
        self.status_label = tk.Label(display_frame, text="ESPERANDO", 
                                     font=("Helvetica", 18, "bold"), 
                                     bg="#f0f0f0", fg="#888",
                                     padx=20, pady=10, relief=tk.RAISED)
        self.status_label.pack(pady=20)
        
        ref_frame = tk.Frame(main_frame, bg="#667eea")
        ref_frame.pack(pady=10)
        
        tk.Label(ref_frame, text="Cuerdas del Ukelele:", 
                font=("Helvetica", 12, "bold"), 
                bg="#667eea", fg="white").pack()
        
        strings_text = " | ".join([f"{k}: {v:.2f}Hz" for k, v in UKULELE_TARGETS.items()])
        tk.Label(ref_frame, text=strings_text, 
                font=("Helvetica", 10), 
                bg="#667eea", fg="white").pack()
        
        self.control_button = tk.Button(main_frame, text="▶ INICIAR", 
                                       command=self.toggle_tuner,
                                       font=("Helvetica", 16, "bold"),
                                       bg="#51cf66", fg="white",
                                       activebackground="#40c057",
                                       relief=tk.RAISED, bd=3,
                                       padx=40, pady=15,
                                       cursor="hand2")
        self.control_button.pack(pady=20)
        
        log_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(log_frame, text="📋 Registro de Eventos", 
                font=("Helvetica", 12, "bold"), 
                bg="white", fg="#333").pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                  height=8,
                                                  font=("Courier", 9),
                                                  bg="#f9f9f9",
                                                  fg="#333",
                                                  state=tk.DISABLED,
                                                  wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def draw_meter(self):
        self.meter_canvas.delete("all")
        width = self.meter_canvas.winfo_width()
        height = self.meter_canvas.winfo_height()
        
        if width <= 1:
            return
        
        center_x = width / 2
        
        for i in range(-50, 51, 10):
            x = center_x + (i / 50) * (width / 2 - 20)
            mark_height = 30 if i == 0 else 15
            self.meter_canvas.create_line(x, height/2 - mark_height/2, 
                                         x, height/2 + mark_height/2,
                                         fill="#333" if i == 0 else "#ccc",
                                         width=3 if i == 0 else 1)
            
            if i % 20 == 0:
                self.meter_canvas.create_text(x, height - 15, 
                                             text=str(i), 
                                             font=("Helvetica", 8))
        
        if self.tuner.is_stable and self.tuner.status != "ESPERANDO":
            cents = max(-50, min(50, self.tuner.cents))
            needle_x = center_x + (cents / 50) * (width / 2 - 20)
            
            color = "#51cf66" if abs(cents) <= 5 else "#ff6b6b"
            
            self.meter_canvas.create_line(needle_x, 20, needle_x, height - 30,
                                         fill=color, width=4)
            self.meter_canvas.create_oval(needle_x - 8, height/2 - 8,
                                         needle_x + 8, height/2 + 8,
                                         fill=color, outline="white", width=2)
    
    def update_display(self):
        if self.tuner.is_running:
            self.string_label.config(text=self.tuner.current_string if self.tuner.is_stable else "...")
            
            if self.tuner.detected_freq > 0:
                self.freq_label.config(text=f"Detectada: {self.tuner.detected_freq:.2f} Hz")
            else:
                self.freq_label.config(text="Detectada: --- Hz")
            
            if self.tuner.target_freq > 0:
                self.target_label.config(text=f"Objetivo: {self.tuner.target_freq:.2f} Hz")
            else:
                self.target_label.config(text="Objetivo: --- Hz")
            
            if self.tuner.is_stable and self.tuner.status != "ESPERANDO":
                self.cents_label.config(text=f"{self.tuner.cents:+.1f} cents")
                
                if abs(self.tuner.cents) <= 5:
                    self.cents_label.config(fg="#51cf66")
                elif self.tuner.cents > 0:
                    self.cents_label.config(fg="#ffa94d")
                else:
                    self.cents_label.config(fg="#ff6b6b")
            else:
                self.cents_label.config(text="--- cents", fg="#666")
            
            self.status_label.config(text=self.tuner.status)
            
            status_colors = {
                "AFINADO": ("#e0ffe0", "#51cf66"),
                "AGUDO": ("#fff3e0", "#ffa94d"),
                "GRAVE": ("#ffe0e0", "#ff6b6b"),
                "ESPERANDO": ("#f0f0f0", "#888"),
                "SEÑAL BAJA": ("#f0f0f0", "#888"),
                "FUERA DE RANGO": ("#ffe0e0", "#ff6b6b"),
            }
            
            bg, fg = status_colors.get(self.tuner.status, ("#f0f0f0", "#888"))
            self.status_label.config(bg=bg, fg=fg)
            
            self.draw_meter()
        
        self.root.after(50, self.update_display)
    
    def toggle_tuner(self):
        if not self.tuner.is_running:
            self.tuner.start()
            self.control_button.config(text="⏹ DETENER", bg="#ff6b6b", 
                                      activebackground="#ee5a6f")
        else:
            self.tuner.stop()
            self.control_button.config(text="▶ INICIAR", bg="#51cf66",
                                      activebackground="#40c057")
            
            self.string_label.config(text="---")
            self.freq_label.config(text="Detectada: --- Hz")
            self.target_label.config(text="Objetivo: --- Hz")
            self.cents_label.config(text="--- cents", fg="#666")
            self.status_label.config(text="ESPERANDO", bg="#f0f0f0", fg="#888")
            self.meter_canvas.delete("all")


def main():
    root = tk.Tk()
    app = TunerGUI(root)
    
    def on_closing():
        app.tuner.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()