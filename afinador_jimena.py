"""
Afinador de Ukelele con Interfaz Gráfica
Usa FFT + Harmonic Product Spectrum (HPS)

Requirements:
  pip install numpy scipy sounddevice customtkinter pillow
"""

import copy
import threading
import numpy as np
import scipy.fftpack
import sounddevice as sd
import customtkinter as ctk
from tkinter import Canvas
from PIL import Image
from datetime import datetime
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

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


class TunerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        
        self.title("Afinador de Ukelele - Proyecto Final DSP")
        self.geometry("1100x750")
        self.minsize(1100, 750)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.tuner = UkuleleTuner(log_callback=self.add_log)
        
        self.string_labels = {}

        
        header = ctk.CTkFrame(self, height=90, corner_radius=0, fg_color="#1a1a2e")
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        header.grid_columnconfigure((0, 1, 2), weight=1)

        
        logo_path = "/Users/jimm/Documents/Septimo Semestre/Señales/P.FINAL/Logo-ESCOM.png"
        if os.path.exists(logo_path):
            logo = ctk.CTkImage(Image.open(logo_path), size=(70, 70))
            ctk.CTkLabel(header, image=logo, text="").grid(row=0, column=0, padx=20)
            ctk.CTkLabel(header, image=logo, text="").grid(row=0, column=2, padx=20)
        else:
            ctk.CTkLabel(header, text="🎵", font=ctk.CTkFont(size=40)).grid(row=0, column=0, padx=20)
            ctk.CTkLabel(header, text="🎵", font=ctk.CTkFont(size=40)).grid(row=0, column=2, padx=20)

        ctk.CTkLabel(
            header,
            text=(
                "INSTITUTO POLITÉCNICO NACIONAL\n"
                "ESCUELA SUPERIOR DE CÓMPUTO – ESCOM\n"
                "Procesamiento Digital de Señales | Proyecto Final"
            ),
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="center",
        ).grid(row=0, column=1)

        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#16213e")
        self.sidebar.grid(row=1, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar, 
            text="🎸 AFINADOR\nDE UKELELE",
            font=ctk.CTkFont(size=24, weight="bold"),
            justify="center"
        ).pack(pady=(25, 5))
        
        ctk.CTkLabel(
            self.sidebar,
            text="FFT + Harmonic Product Spectrum",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(pady=(0, 15))


        self.btn_toggle = ctk.CTkButton(
            self.sidebar, 
            text="▶  INICIAR",
            height=55,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#00d9a5",
            hover_color="#00b894",
            text_color="#1a1a2e",
            corner_radius=12,
            command=self.toggle_tuner
        )
        self.btn_toggle.pack(padx=20, pady=15, fill="x")

        level_frame = ctk.CTkFrame(self.sidebar, fg_color="#1b1f24", corner_radius=10)
        level_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(
            level_frame,
            text="📊 NIVEL DE SEÑAL",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray"
        ).pack(pady=(8, 5))
        
        self.level_canvas = Canvas(
            level_frame, 
            height=25, 
            bg="#0a0a14",
            highlightthickness=0
        )
        self.level_canvas.pack(padx=10, pady=(0, 10), fill="x")


        ctk.CTkLabel(self.sidebar, text="─" * 22, text_color="#333").pack(pady=8)

        ctk.CTkLabel(
            self.sidebar,
            text="🎶 CUERDAS DEL UKELELE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        ).pack(pady=(5, 10))


        strings_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        strings_frame.pack(padx=15, fill="x")

        string_colors = ["#ff6b6b", "#ffa502", "#2ed573", "#1e90ff"]
        string_names_display = ["G", "C", "E", "A"]
        
        for i, (name, freq) in enumerate(UKULELE_TARGETS.items()):
            string_row = ctk.CTkFrame(strings_frame, fg_color="#1b1f24", corner_radius=10)
            string_row.pack(fill="x", pady=3)
            

            color_indicator = ctk.CTkFrame(
                string_row, width=6, height=40, 
                fg_color=string_colors[i], corner_radius=3
            )
            color_indicator.pack(side="left", padx=(8, 12), pady=6)
            

            ctk.CTkLabel(
                string_row, 
                text=f"{4-i}ª",
                font=ctk.CTkFont(size=11),
                text_color="gray",
                width=25
            ).pack(side="left")

            string_label = ctk.CTkLabel(
                string_row, 
                text=string_names_display[i],
                font=ctk.CTkFont(size=18, weight="bold"),
                width=35
            )
            string_label.pack(side="left", padx=5)
            self.string_labels[name] = string_label
            

            ctk.CTkLabel(
                string_row,
                text=f"{freq:.1f} Hz",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            ).pack(side="right", padx=12)


        ctk.CTkLabel(self.sidebar, text="─" * 22, text_color="#333").pack(pady=10)


        status_frame = ctk.CTkFrame(self.sidebar, fg_color="#1b1f24", corner_radius=12)
        status_frame.pack(padx=20, pady=5, fill="x")
        
        self.status_icon = ctk.CTkLabel(
            status_frame,
            text="⏸",
            font=ctk.CTkFont(size=30)
        )
        self.status_icon.pack(pady=(10, 0))
        
        self.status_sidebar = ctk.CTkLabel(
            status_frame,
            text="ESPERANDO",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray"
        )
        self.status_sidebar.pack(pady=(0, 10))


        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#1a1a2e")
        self.main.grid(row=1, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)


        top_panel = ctk.CTkFrame(self.main, fg_color="#0f3460", corner_radius=15)
        top_panel.pack(padx=20, pady=(20, 10), fill="x")
        top_panel.grid_columnconfigure((0, 1), weight=1)


        note_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        note_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.nota_display = ctk.CTkLabel(
            note_frame, 
            text="--",
            font=ctk.CTkFont(size=100, weight="bold"),
            text_color="#e94560"
        )
        self.nota_display.pack()

        self.frecuencia_label = ctk.CTkLabel(
            note_frame, 
            text="--- Hz",
            font=ctk.CTkFont(size=20), 
            text_color="gray"
        )
        self.frecuencia_label.pack()

        self.target_label = ctk.CTkLabel(
            note_frame,
            text="Objetivo: --- Hz",
            font=ctk.CTkFont(size=12),
            text_color="#666"
        )
        self.target_label.pack(pady=(5, 0))


        meter_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        meter_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")


        self.cents_label = ctk.CTkLabel(
            meter_frame,
            text="0 cents",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="gray"
        )
        self.cents_label.pack(pady=(10, 5))


        self.status_badge = ctk.CTkLabel(
            meter_frame,
            text="  ESPERANDO  ",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2a2f36",
            corner_radius=20,
            text_color="gray"
        )
        self.status_badge.pack(pady=10)


        meter_container = ctk.CTkFrame(meter_frame, fg_color="#111418", corner_radius=10)
        meter_container.pack(fill="x", pady=10)


        meter_labels = ctk.CTkFrame(meter_container, fg_color="transparent")
        meter_labels.pack(fill="x", padx=15, pady=(8, 0))
        
        ctk.CTkLabel(
            meter_labels, text="◀ GRAVE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#ff4757"
        ).pack(side="left")
        
        ctk.CTkLabel(
            meter_labels, text="AGUDO ▶",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#ffc107"
        ).pack(side="right")

        self.meter_canvas = Canvas(
            meter_container, height=50, bg="#0a0a14",
            highlightthickness=0, bd=0
        )
        self.meter_canvas.pack(padx=15, pady=(5, 12), fill="x")


        log_container = ctk.CTkFrame(self.main, fg_color="#1b1f24", corner_radius=12)
        log_container.pack(padx=20, pady=(15, 20), fill="both", expand=True)

        log_header = ctk.CTkFrame(log_container, fg_color="transparent")
        log_header.pack(fill="x", padx=15, pady=(8, 0))
        
        ctk.CTkLabel(
            log_header,
            text="📋 REGISTRO DE EVENTOS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        ).pack(side="left")

        self.log_textbox = ctk.CTkTextbox(
            log_container,
            height=60,
            font=ctk.CTkFont(family="Monaco", size=10),
            fg_color="#0a0a14",
            text_color="#00d9a5",
            corner_radius=8
        )
        self.log_textbox.pack(padx=10, pady=(5, 10), fill="both", expand=True)
        self.log_textbox.configure(state="disabled")


        self.add_log("📱 Afinador iniciado. Presiona INICIAR para comenzar.")
        self.update_display()

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", log_message)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def draw_level_meter(self):
        """Dibuja el indicador de nivel de señal"""
        self.level_canvas.delete("all")
        width = self.level_canvas.winfo_width()
        height = self.level_canvas.winfo_height()
        
        if width <= 1:
            return
        

        level = min(1.0, self.tuner.signal_level * 10000)
        bar_width = int((width - 20) * level)
        
        # Color basado en nivel
        if level < 0.3:
            color = "#00d9a5"  # Verde - bajo
        elif level < 0.7:
            color = "#ffc107"  # Amarillo - medio
        else:
            color = "#ff4757"  # Rojo - alto
        

        self.level_canvas.create_rectangle(10, 5, width-10, height-5, fill="#1a1a2e", outline="#333")
        

        if bar_width > 0:
            self.level_canvas.create_rectangle(10, 5, 10 + bar_width, height-5, fill=color, outline="")
        

        for i in range(0, 11, 2):
            x = 10 + (width - 20) * i / 10
            self.level_canvas.create_line(x, height-3, x, height-7, fill="#444")

    def draw_meter(self):
        """Dibuja el medidor de afinación"""
        self.meter_canvas.delete("all")
        width = self.meter_canvas.winfo_width()
        height = self.meter_canvas.winfo_height()

        if width <= 1:
            return

        center_x = width / 2
        margin = 30
        usable_width = width - (2 * margin)


        self.meter_canvas.create_rectangle(
            margin, 8, center_x - 35, height - 8,
            fill='#2d1f1f', outline=''
        )
        self.meter_canvas.create_rectangle(
            center_x - 35, 8, center_x + 35, height - 8,
            fill='#1f2d1f', outline=''
        )
        self.meter_canvas.create_rectangle(
            center_x + 35, 8, width - margin, height - 8,
            fill='#2d2d1f', outline=''
        )


        for i in range(-50, 51, 10):
            x = center_x + (i / 50) * (usable_width / 2)
            if i == 0:
                self.meter_canvas.create_line(
                    x, 3, x, height - 3,
                    fill='#00d9a5', width=3
                )
            else:
                mark_h = 16 if i % 20 == 0 else 8
                self.meter_canvas.create_line(
                    x, height/2 - mark_h/2, x, height/2 + mark_h/2,
                    fill='#444', width=1
                )


        if self.tuner.is_stable and self.tuner.status not in ["ESPERANDO", "SEÑAL BAJA"]:
            cents = max(-50, min(50, self.tuner.cents))
            needle_x = center_x + (cents / 50) * (usable_width / 2)

            if abs(cents) <= 5:
                color = '#00d9a5'
                glow = '#00ffaa'
            elif cents > 0:
                color = '#ffc107'
                glow = '#ffdd00'
            else:
                color = '#ff4757'
                glow = '#ff6666'

            self.meter_canvas.create_oval(
                needle_x - 14, height/2 - 14,
                needle_x + 14, height/2 + 14,
                fill='', outline=glow, width=2
            )


            self.meter_canvas.create_line(
                needle_x, 10, needle_x, height - 10,
                fill=color, width=4
            )


            self.meter_canvas.create_oval(
                needle_x - 7, height/2 - 7,
                needle_x + 7, height/2 + 7,
                fill=color, outline='white', width=2
            )
        else:

            self.meter_canvas.create_oval(
                center_x - 5, height/2 - 5,
                center_x + 5, height/2 + 5,
                fill='#333', outline='#444', width=2
            )

    def update_display(self):
        if self.tuner.is_running:

            if self.tuner.is_stable:

                note_name = self.tuner.current_string.replace("4", "").replace("3", "")
                self.nota_display.configure(text=note_name)

                if self.tuner.status == "AFINADO":
                    self.nota_display.configure(text_color="#00d9a5")
                elif self.tuner.status == "AGUDO":
                    self.nota_display.configure(text_color="#ffc107")
                elif self.tuner.status == "GRAVE":
                    self.nota_display.configure(text_color="#ff4757")
                else:
                    self.nota_display.configure(text_color="#e94560")

                for name, label in self.string_labels.items():
                    if name == self.tuner.current_string:
                        label.configure(text_color="#00d9a5")
                    else:
                        label.configure(text_color="white")
            else:
                self.nota_display.configure(text="...", text_color="gray")
                for label in self.string_labels.values():
                    label.configure(text_color="white")

            if self.tuner.detected_freq > 0:
                self.frecuencia_label.configure(text=f"{self.tuner.detected_freq:.2f} Hz")
            else:
                self.frecuencia_label.configure(text="--- Hz")

            if self.tuner.target_freq > 0:
                self.target_label.configure(text=f"Objetivo: {self.tuner.target_freq:.2f} Hz")
            else:
                self.target_label.configure(text="Objetivo: --- Hz")


            if self.tuner.is_stable and self.tuner.status not in ["ESPERANDO", "SEÑAL BAJA"]:
                self.cents_label.configure(text=f"{self.tuner.cents:+.1f} cents")
                if abs(self.tuner.cents) <= 5:
                    self.cents_label.configure(text_color="#00d9a5")
                elif self.tuner.cents > 0:
                    self.cents_label.configure(text_color="#ffc107")
                else:
                    self.cents_label.configure(text_color="#ff4757")
            else:
                self.cents_label.configure(text="--- cents", text_color="gray")


            status_configs = {
                "AFINADO": ("#00d9a5", "#1a1a2e"),
                "AGUDO": ("#ffc107", "#1a1a2e"),
                "GRAVE": ("#ff4757", "white"),
                "ESPERANDO": ("#2a2f36", "gray"),
                "SEÑAL BAJA": ("#2a2f36", "gray"),
                "FUERA DE RANGO": ("#4a1f3d", "#ff4757"),
            }
            bg, fg = status_configs.get(self.tuner.status, ("#2a2f36", "gray"))
            self.status_badge.configure(text=f"  {self.tuner.status}  ", fg_color=bg, text_color=fg)


            status_icons = {
                "AFINADO": ("✅", "#00d9a5"),
                "AGUDO": ("🔺", "#ffc107"),
                "GRAVE": ("🔻", "#ff4757"),
                "ESPERANDO": ("⏸", "gray"),
                "SEÑAL BAJA": ("🔇", "gray"),
                "FUERA DE RANGO": ("⚠️", "#ff4757"),
            }
            icon, color = status_icons.get(self.tuner.status, ("⏸", "gray"))
            self.status_icon.configure(text=icon)
            self.status_sidebar.configure(text=self.tuner.status, text_color=color)


            self.draw_meter()
            self.draw_level_meter()

        self.after(50, self.update_display)

    def toggle_tuner(self):
        if not self.tuner.is_running:
            self.tuner.start()
            self.btn_toggle.configure(
                text="⏹  DETENER",
                fg_color="#ff4757",
                hover_color="#ee5a6f",
                text_color="white"
            )
        else:
            self.tuner.stop()
            self.btn_toggle.configure(
                text="▶  INICIAR",
                fg_color="#00d9a5",
                hover_color="#00b894",
                text_color="#1a1a2e"
            )


            self.nota_display.configure(text="--", text_color="#e94560")
            self.frecuencia_label.configure(text="--- Hz")
            self.target_label.configure(text="Objetivo: --- Hz")
            self.cents_label.configure(text="--- cents", text_color="gray")
            self.status_badge.configure(text="  ESPERANDO  ", fg_color="#2a2f36", text_color="gray")
            self.status_icon.configure(text="⏸")
            self.status_sidebar.configure(text="ESPERANDO", text_color="gray")
            
            for label in self.string_labels.values():
                label.configure(text_color="white")
            
            self.meter_canvas.delete("all")
            self.draw_meter()


def main():
    app = TunerGUI()
    
    def on_closing():
        app.tuner.stop()
        app.destroy()
    
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()