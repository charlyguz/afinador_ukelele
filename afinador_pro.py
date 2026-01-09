
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
import math
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

SAMPLE_FREQ = 48000
WINDOW_SIZE = 32768
WINDOW_STEP = 8192
NUM_HPS = 5
POWER_THRESH = 1e-6
WHITE_NOISE_THRESH = 0.20
CONCERT_PITCH = 440.0

# Parámetros de estabilidad mejorados
SMOOTH_ALPHA = 0.35  # Aumentado para más suavizado
STABLE_FRAMES = 5  # Más frames para confirmar estabilidad
SIGNAL_DECAY_THRESHOLD = 0.3  # Si señal cae >30%, congelar aguja
MIN_SIGNAL_FOR_UPDATE = 2e-6  # Señal mínima para actualizar
FREQ_BUFFER_SIZE = 5  # Tamaño del buffer de promedio móvil

LOW_G = False

UKULELE_TARGETS = {
    ("G4" if not LOW_G else "G3"): (392.00 if not LOW_G else 196.00),
    "C4": 262.63,
    "E4": 329.63,
    "A4": 440.00,
}

GUITAR_TARGETS = {
    "E2": 82.00,
    "A2": 110.00,
    "D3": 146.80,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 330.00,
}

CURRENT_INSTRUMENT = "ukulele"

def get_current_targets():
    return UKULELE_TARGETS if CURRENT_INSTRUMENT == "ukulele" else GUITAR_TARGETS

DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
HANN_WINDOW = np.hanning(WINDOW_SIZE)


class UkuleleTuner:
    def __init__(self, log_callback=None, debug=True):
        self.window_samples = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.stable_buffer = []
        self.smooth_freq = None
        self.is_running = False
        self.stream = None
        self.log_callback = log_callback
        self.debug = debug
        self.debug_counter = 0  # Para imprimir cada N frames
        
        self.current_string = "---"
        self.detected_freq = 0.0
        self.target_freq = 0.0
        self.cents = 0.0
        self.status = "ESPERANDO"
        self.is_stable = False
        self.signal_level = 0.0
        
        # Variables para estabilidad mejorada
        self.last_valid_freq = None  # Última frecuencia válida
        self.last_signal_level = 0.0  # Nivel anterior de señal
        self.freq_buffer = []  # Buffer de frecuencias para promedio móvil
        
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        
    def cents_error(self, f_detected, f_target):
        if f_target <= 0:
            return 0.0
        return 1200.0 * np.log2(f_detected / f_target)
    
    def find_closest_string(self, f_detected):
        best = None
        targets = get_current_targets()
        for s, f_t in targets.items():
            c = self.cents_error(f_detected, f_t)
            score = abs(c)
            if best is None or score < best[0]:
                best = (score, s, f_t, c)
        _, s, f_t, c = best
        return s, f_t, c
    
    def format_status(self, cents):
        if abs(cents) <= 10:
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
        
        # Detectar caída brusca de señal (nota decayendo)
        if self.last_signal_level > 0:
            signal_ratio = signal_power / self.last_signal_level
            if signal_ratio < SIGNAL_DECAY_THRESHOLD and self.last_valid_freq:
                # La señal está decayendo rápidamente, mantener última frecuencia válida
                if self.debug and self.debug_counter % 10 == 0:
                    print(f"\r[HOLD] Manteniendo última nota válida - Señal decayendo ({signal_ratio:.2%})", end="", flush=True)
                self.last_signal_level = signal_power
                return  # No actualizar nada, mantener estado actual
        
        self.last_signal_level = signal_power
        
        if signal_power < POWER_THRESH:
            if self.status != "SEÑAL BAJA":
                self.log("Señal de audio muy baja")
            self.status = "SEÑAL BAJA"
            self.current_string = "---"
            self.is_stable = False
            self.freq_buffer.clear()
            return
        
        # Verificar señal mínima para actualizar
        if signal_power < MIN_SIGNAL_FOR_UPDATE:
            return  # Mantener estado actual
        
        hann_samples = self.window_samples * HANN_WINDOW
        magnitude_spec = np.abs(scipy.fftpack.fft(hann_samples)[:WINDOW_SIZE // 2])
        
        # Filtro de corte bajo para eliminar ruido, pero permitir E2 de guitarra (82 Hz)
        cutoff_bins = int(50 / DELTA_FREQ)
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
        
        # Debug: Imprimir frecuencia detectada cada 10 frames (~0.5 segundos)
        if self.debug and self.debug_counter % 10 == 0:
            print(f"\r[DEBUG] Freq: {max_freq:7.2f} Hz | Señal: {signal_power:8.2e} | Rango: 75-650 Hz", end="", flush=True)
        self.debug_counter += 1
        
        # Rango ampliado para guitarra: E2 (82 Hz) hasta más allá de E4 (329 Hz)
        if not (75.0 <= max_freq <= 650.0):
            if self.status != "FUERA DE RANGO":
                self.log(f"\n Frecuencia fuera de rango: {max_freq:.2f} Hz")
            self.status = "FUERA DE RANGO"
            self.current_string = "---"
            self.is_stable = False
            self.freq_buffer.clear()
            return
        
        # Agregar frecuencia al buffer para promedio móvil
        self.freq_buffer.append(max_freq)
        if len(self.freq_buffer) > FREQ_BUFFER_SIZE:
            self.freq_buffer.pop(0)
        
        # Usar promedio del buffer en lugar de solo la última lectura
        avg_freq = sum(self.freq_buffer) / len(self.freq_buffer)
        
        if self.smooth_freq is None:
            self.smooth_freq = avg_freq
        else:
            self.smooth_freq = (1 - SMOOTH_ALPHA) * avg_freq + SMOOTH_ALPHA * self.smooth_freq
        
        f = float(self.smooth_freq)
        self.last_valid_freq = f  # Guardar última frecuencia válida
        string_name, f_target, cents = self.find_closest_string(f)
        status_txt = self.format_status(cents)
        
        # Calcular estabilidad primero
        self.stable_buffer.insert(0, string_name)
        self.stable_buffer = self.stable_buffer[:STABLE_FRAMES]
        stable = (len(self.stable_buffer) == STABLE_FRAMES and 
                 self.stable_buffer.count(self.stable_buffer[0]) == STABLE_FRAMES)
        
        # Debug: Imprimir información detallada de la nota detectada
        if self.debug and self.debug_counter % 10 == 0:
            stable_indicator = "✔" if stable else "⌛"
            status_color = "VERDE" if status_txt == "AFINADO" else ("NARANJA" if status_txt == "AGUDO" else "ROJO")
            print(f" | Nota: {string_name:3} ({f_target:6.2f}Hz) | Desv: {cents:+6.2f}c | Estado: {status_txt:7} | {stable_indicator}", end="", flush=True)
        
        if stable and string_name != self.current_string:
            if self.debug:
                print()  # Nueva línea para separar del debug
            self.log(f"🎵 Cuerda detectada: {string_name} ({f:.2f} Hz)")
        
        if stable and status_txt == "AFINADO" and self.status != "AFINADO":
            if self.debug:
                print()  # Nueva línea para separar del debug
            self.log(f" ¡{string_name} está afinado! ({cents:+.1f} cents)")
        
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
                self.log(f" Audio iniciado (Sample Rate: {SAMPLE_FREQ} Hz)")
                return True
            except Exception as e:
                self.is_running = False
                self.log(f" Error al iniciar audio: {e}")
                return False
        return True
    
    def stop(self):
        if self.is_running:
            self.log(" Deteniendo afinador...")
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
        self.last_valid_freq = None
        self.last_signal_level = 0.0
        self.freq_buffer = []


class SemiCircleGauge(ctk.CTkCanvas):
    """Widget de gauge semicircular para mostrar afinación en cents"""
    def __init__(self, parent, width=550, height=350, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg="#f5f5f5", highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.current_angle = 0  # Ángulo actual de la aguja
        self.target_angle = 0   # Ángulo objetivo de la aguja
        self.cents = 0
        self.is_tuned = False
        
        self.draw_gauge()
    
    def draw_gauge(self):
        """Dibuja el gauge base (sin aguja)"""
        self.delete("all")
        
        # Centro y radio
        cx = self.width / 2
        cy = self.height - 40
        radius = min(self.width, self.height) - 60
        
        # Zona sombreada verde para rango permitido (±10 cents)
        # Calcular ángulos para -10 y +10 cents
        angle_minus_10 = 180 - (-10 + 80) * 180 / 160
        angle_plus_10 = 180 - (10 + 80) * 180 / 160
        extent_angle = angle_minus_10 - angle_plus_10
        
        # Dibujar arco sombreado
        self.create_arc(
            cx - radius + 10, cy - radius + 10, cx + radius - 10, cy + radius - 10,
            start=angle_plus_10, extent=extent_angle, 
            fill="#d4f4dd", outline="", style="pieslice"
        )
        
        # Arco semicircular (sin fondo)
        self.create_arc(
            cx - radius, cy - radius, cx + radius, cy + radius,
            start=0, extent=180, fill="", outline="#e0e0e0", width=2,
            style="arc"
        )
        
        # Dibujar marcas radiales
        for i in range(-80, 81, 10):
            angle_deg = 180 - (i + 80) * 180 / 160  # Mapear -80 a +80 cents
            angle_rad = math.radians(angle_deg)
            
            # Determinar longitud de marca
            if i == 0:
                mark_length = 35
                width = 4
                color = "#00d9a5"
            elif i % 20 == 0:
                mark_length = 25
                width = 3
                color = "#666"
            else:
                mark_length = 12
                width = 2
                color = "#bbb"
            
            # Punto inicial (exterior)
            x1 = cx + (radius - 5) * math.cos(angle_rad)
            y1 = cy - (radius - 5) * math.sin(angle_rad)
            
            # Punto final (interior)
            x2 = cx + (radius - 5 - mark_length) * math.cos(angle_rad)
            y2 = cy - (radius - 5 - mark_length) * math.sin(angle_rad)
            
            self.create_line(x1, y1, x2, y2, fill=color, width=width)
            
            # Etiquetas cada 20 cents
            if i % 20 == 0 and abs(i) <= 60:
                label_x = cx + (radius - 50) * math.cos(angle_rad)
                label_y = cy - (radius - 50) * math.sin(angle_rad)
                self.create_text(label_x, label_y, text=str(i), 
                               font=("Helvetica", 12, "bold"), fill="#555")
        
        # Círculo central decorativo
        self.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, 
                        fill="#f5f5f5", outline="#ccc", width=2)
    
    def update_needle(self, cents, is_tuned=False):
        """Actualiza la posición de la aguja con animación suave"""
        self.cents = max(-80, min(80, cents))
        self.is_tuned = is_tuned
        
        # Calcular ángulo objetivo
        self.target_angle = 180 - (self.cents + 80) * 180 / 160
        
    def animate(self):
        """Anima la aguja hacia el ángulo objetivo"""
        # Interpolación suave
        diff = self.target_angle - self.current_angle
        self.current_angle += diff * 0.15
        
        # Redibujar gauge y aguja
        self.draw_gauge()
        self.draw_needle()
        
    def draw_needle(self):
        """Dibuja la aguja en la posición actual"""
        cx = self.width / 2
        cy = self.height - 40
        radius = min(self.width, self.height) - 60
        
        angle_rad = math.radians(self.current_angle)
        
        # Determinar color basado en afinación
        if abs(self.cents) <= 10:
            color = "#00d9a5"  # Verde - afinado
            glow_color = "#00ffaa"
        elif self.cents > 0:
            color = "#ffa502"  # Naranja - agudo
            glow_color = "#ffbb33"
        else:
            color = "#ff4757"  # Rojo - grave
            glow_color = "#ff6666"
        
        # Punto final de la aguja
        needle_x = cx + (radius - 45) * math.cos(angle_rad)
        needle_y = cy - (radius - 45) * math.sin(angle_rad)
        
        # Efecto glow (más grande)
        self.create_line(cx, cy, needle_x, needle_y, 
                        fill=glow_color, width=10, capstyle="round")
        
        # Aguja principal (más gruesa)
        self.create_line(cx, cy, needle_x, needle_y, 
                        fill=color, width=6, capstyle="round")
        
        # Círculo central sobre la aguja (más grande)
        self.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, 
                        fill=color, outline="white", width=2)


class CircularStringButton(ctk.CTkFrame):
    """Botón circular para seleccionar cuerdas"""
    def __init__(self, parent, string_name, diameter=130, command=None, **kwargs):
        super().__init__(parent, width=diameter, height=diameter, 
                        fg_color="transparent", **kwargs)
        
        self.string_name = string_name
        self.diameter = diameter
        self.command = command
        self.is_active = False
        self.is_tuned = False  # Nueva propiedad para marcar como afinada
        
        # Canvas para dibujar el círculo
        self.canvas = Canvas(self, width=diameter, height=diameter,
                           bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack()
        
        self.draw()
        
        # Bind click
        self.canvas.bind("<Button-1>", self.on_click)
        
    def draw(self):
        """Dibuja el botón circular"""
        self.canvas.delete("all")
        
        cx = self.diameter / 2
        cy = self.diameter / 2
        radius = self.diameter / 2 - 6
        
        if self.is_tuned:
            # Estado afinado: fondo verde
            # Sombra
            self.canvas.create_oval(
                cx - radius + 2, cy - radius + 2, cx + radius + 2, cy + radius + 2,
                fill="#b3e6c4", outline=""
            )
            # Botón principal
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#00d9a5", outline="#00b894", width=3
            )
            text_color = "white"
        elif self.is_active:
            # Estado activo: fondo negro con sombra
            # Sombra
            self.canvas.create_oval(
                cx - radius + 2, cy - radius + 2, cx + radius + 2, cy + radius + 2,
                fill="#cccccc", outline=""
            )
            # Botón principal
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#000000", outline="#00d9a5", width=3
            )
            text_color = "white"
        else:
            # Estado normal: fondo blanco con borde suave
            # Sombra sutil
            self.canvas.create_oval(
                cx - radius + 2, cy - radius + 2, cx + radius + 2, cy + radius + 2,
                fill="#e8e8e8", outline=""
            )
            # Botón principal
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#ffffff", outline="#d0d0d0", width=2
            )
            text_color = "#2c3e50"
        
        # Letra de la cuerda (ajustar tamaño según diámetro)
        font_size = 44 if self.diameter >= 130 else 32
        self.canvas.create_text(
            cx, cy, text=self.string_name,
            font=("Helvetica", font_size, "bold"), fill=text_color
        )
    
    def on_click(self, event):
        """Maneja el clic en el botón"""
        if self.command:
            self.command(self.string_name)
    
    def set_active(self, active):
        """Establece el estado activo del botón"""
        self.is_active = active
        self.draw()
    
    def set_tuned(self, tuned):
        """Establece el estado de afinado del botón"""
        self.is_tuned = tuned
        self.draw()


class TunerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de ventana
        self.title("Afinador de Ukelele - Proyecto Final DSP")
        self.geometry("1000x850")
        self.resizable(False, False)
        self.configure(fg_color="#f5f5f5")

        self.tuner = UkuleleTuner(log_callback=self.add_log)
        
        self.auto_mode = True
        self.selected_string = None
        self.string_buttons = {}
        self.tuned_strings = set()  # Conjunto de cuerdas ya afinadas
        
        # Sistema de confirmación con delay
        self.tuning_confirmation = {}  # {string_name: consecutive_frames}
        self.CONFIRMATION_FRAMES = 30  # ~1.5 segundos a 20 FPS
        
        # Sistema de resplandor de fondo
        self.glow_frame = 0
        self.is_glowing = False
        
        # Cargar sonido de éxito
        self.success_sound = None
        if SOUND_AVAILABLE:
            try:
                sound_path = os.path.join(os.path.dirname(__file__), "assets", "success.mp3")
                if os.path.exists(sound_path):
                    self.success_sound = pygame.mixer.Sound(sound_path)
                    self.success_sound.set_volume(0.5)
            except Exception as e:
                self.add_log(f" No se pudo cargar el sonido: {e}")
        
        # Frame principal
        self.main_container = ctk.CTkFrame(self, fg_color="#f5f5f5")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ==== TOP BAR ====
        top_bar = ctk.CTkFrame(self.main_container, fg_color="transparent", height=50)
        top_bar.pack(fill="x", pady=(0, 20))
        
        # Selector de instrumento
        self.instrument_selector = ctk.CTkOptionMenu(
            top_bar,
            values=["Ukelele", "Guitarra"],
            command=self.change_instrument,
            width=150,
            height=36,
            fg_color="#ffffff",
            button_color="#00d9a5",
            button_hover_color="#00b894",
            text_color="#2c3e50",
            font=ctk.CTkFont(size=14, weight="bold"),
            dropdown_fg_color="#ffffff",
            dropdown_hover_color="#e8f5f0",
            dropdown_text_color="#2c3e50",
            corner_radius=15
        )
        self.instrument_selector.set("Ukelele")
        self.instrument_selector.pack(side="left", padx=(0, 15))
        
        # Toggle Auto
        auto_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        auto_frame.pack(side="left")
        
        ctk.CTkLabel(
            auto_frame, 
            text="Auto",
            font=ctk.CTkFont(size=14),
            text_color="#2c3e50"
        ).pack(side="left", padx=(0, 8))
        
        self.auto_switch = ctk.CTkSwitch(
            auto_frame,
            text="",
            command=self.toggle_auto_mode,
            progress_color="#00d9a5",
            button_color="#00d9a5",
            button_hover_color="#00b894",
            width=50,
            height=24
        )
        self.auto_switch.select()
        self.auto_switch.pack(side="left")
        
        # ==== GAUGE SECTION ====
        gauge_container = ctk.CTkFrame(self.main_container, fg_color="transparent", 
                                      corner_radius=20, height=380)
        gauge_container.pack(fill="x", pady=(0, 20))
        gauge_container.pack_propagate(False)
        
        # Gauge semicircular
        self.gauge = SemiCircleGauge(gauge_container, width=550, height=350)
        self.gauge.pack(pady=15)
        
        # ==== STATUS SECTION ====
        status_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        status_container.pack(fill="x", pady=(0, 25))
        
        # Indicador grande Bajo/Perfecto/Alto
        self.status_label = ctk.CTkLabel(
            status_container,
            text="Bajo",
            font=ctk.CTkFont(size=110, weight="bold"),
            text_color="#ff4757"
        )
        self.status_label.pack()
        
        # Frecuencia detectada
        self.freq_label = ctk.CTkLabel(
            status_container,
            text="328Hz",
            font=ctk.CTkFont(size=28),
            text_color="#999"
        )
        self.freq_label.pack(pady=(10, 0))
        
        # ==== STRING BUTTONS SECTION ====
        strings_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        strings_container.pack(fill="both", expand=True)
        
        # Grid para botones de cuerdas (2x3 para guitarra, 2x2 para ukelele)
        self.strings_grid = ctk.CTkFrame(strings_container, fg_color="transparent")
        self.strings_grid.pack(expand=True)
        
        # Crear botones para ukelele inicialmente
        self.create_string_buttons()
        
        # Log inicial
        self.log_messages = []
        self.add_log("📱 Afinador iniciado. Escucha automática activada.")
        
        # Iniciar tuner automáticamente
        self.tuner.start()
        
        # Iniciar actualización de display
        self.update_display()
    
    def create_string_buttons(self):
        """Crea los botones de cuerdas según el instrumento actual"""
        # Limpiar botones existentes
        for widget in self.strings_grid.winfo_children():
            widget.destroy()
        self.string_buttons.clear()
        
        targets = get_current_targets()
        string_names = list(targets.keys())
        
        # Ajustar tamaño según instrumento
        if CURRENT_INSTRUMENT == "ukulele":
            # Layout 1x4 para ukelele en una sola fila: G, C, E, A
            layout = [
                ["G4" if not LOW_G else "G3", "C4", "E4", "A4"]
            ]
            button_diameter = 130
            button_padding = 20
        else:
            # Layout 1x6 para guitarra en una sola fila
            layout = [
                ["E2", "A2", "D3", "G3", "B3", "E4"]
            ]
            button_diameter = 110  # Más pequeños para que quepan 6
            button_padding = 12
        
        for row_idx, row in enumerate(layout):
            for col_idx, string_name in enumerate(row):
                if string_name in targets:
                    # Para guitarra, mostrar nota con número para distinguir E2 de E4
                    if CURRENT_INSTRUMENT == "guitar":
                        display_name = string_name  # E2, A2, D3, etc.
                    else:
                        # Para ukelele, solo la letra
                        display_name = string_name[0]
                    
                    btn = CircularStringButton(
                        self.strings_grid,
                        string_name=display_name,
                        diameter=button_diameter,
                        command=lambda s=string_name: self.select_string(s)
                    )
                    btn.grid(row=row_idx, column=col_idx, padx=button_padding, pady=20)
                    self.string_buttons[string_name] = btn
    
    def change_instrument(self, choice):
        """Cambia entre ukelele y guitarra"""
        global CURRENT_INSTRUMENT
        if choice == "Ukelele":
            CURRENT_INSTRUMENT = "ukulele"
            self.add_log("🎵 Cambiado a modo Ukelele (4 cuerdas)")
        else:
            CURRENT_INSTRUMENT = "guitar"
            self.add_log("🎸 Cambiado a modo Guitarra (6 cuerdas)")
        
        # Limpiar cuerdas afinadas al cambiar instrumento
        self.tuned_strings.clear()
        
        # Recrear botones de cuerdas
        self.create_string_buttons()
        
        # Resetear tuner
        self.tuner.reset()
    
    def toggle_auto_mode(self):
        """Alterna entre modo automático y manual"""
        self.auto_mode = self.auto_switch.get()
        if self.auto_mode:
            self.add_log("🔄 Modo automático activado")
            self.selected_string = None
            # Desactivar todos los botones
            for btn in self.string_buttons.values():
                btn.set_active(False)
        else:
            self.add_log("🎯 Modo manual activado - selecciona una cuerda")
    
    def select_string(self, string_name):
        """Selecciona manualmente una cuerda en modo manual"""
        if not self.auto_mode:
            self.selected_string = string_name
            self.add_log(f"🎯 Cuerda seleccionada: {string_name}")
            
            # Actualizar estado visual de botones
            for name, btn in self.string_buttons.items():
                btn.set_active(name == string_name)

    def add_log(self, message):
        """Agrega un mensaje al log (guardado en memoria)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_messages.append(log_message)
        # Mantener solo los últimos 50 mensajes
        if len(self.log_messages) > 50:
            self.log_messages.pop(0)

    def update_display(self):
        """Actualiza la interfaz gráfica"""
        if self.tuner.is_running:
            # Actualizar gauge con animación
            if self.tuner.is_stable and self.tuner.status not in ["ESPERANDO", "SEÑAL BAJA", "FUERA DE RANGO"]:
                self.gauge.update_needle(self.tuner.cents, self.tuner.status == "AFINADO")
            else:
                self.gauge.update_needle(0, False)
            
            self.gauge.animate()
            
            # Actualizar indicador de estado (Bajo/Perfecto/Alto)
            if self.tuner.is_stable and self.tuner.status not in ["ESPERANDO", "SEÑAL BAJA", "FUERA DE RANGO"]:
                if self.tuner.status == "AFINADO":
                    self.status_label.configure(text="Perfecto", text_color="#00d9a5")
                    
                    # Sistema de confirmación con delay
                    if self.auto_mode and self.tuner.current_string not in self.tuned_strings:
                        current_string = self.tuner.current_string
                        
                        # Incrementar contador de frames consecutivos
                        if current_string not in self.tuning_confirmation:
                            self.tuning_confirmation[current_string] = 0
                        
                        self.tuning_confirmation[current_string] += 1
                        
                        # Si alcanza el umbral, confirmar afinación
                        if self.tuning_confirmation[current_string] >= self.CONFIRMATION_FRAMES:
                            self.tuned_strings.add(current_string)
                            self.add_log(f"✅ ¡{current_string} afinada correctamente!")
                            
                            # Reproducir sonido de éxito
                            if self.success_sound:
                                try:
                                    self.success_sound.play()
                                except Exception as e:
                                    pass
                            
                            # Marcar el botón como afinado
                            if current_string in self.string_buttons:
                                self.string_buttons[current_string].set_tuned(True)
                            
                            # Iniciar resplandor de fondo
                            self.start_glow()
                            
                            # Limpiar contador
                            self.tuning_confirmation.pop(current_string, None)
                else:
                    # Si no está afinado, resetear contador
                    if self.tuner.current_string in self.tuning_confirmation:
                        self.tuning_confirmation.pop(self.tuner.current_string)
                    
                    if self.tuner.status == "AGUDO":
                        self.status_label.configure(text="Alto", text_color="#ffa502")
                    else:  # GRAVE
                        self.status_label.configure(text="Bajo", text_color="#ff4757")
            else:
                self.status_label.configure(text="---", text_color="#ccc")
                # Resetear todos los contadores si no hay señal estable
                self.tuning_confirmation.clear()
            
            # Actualizar frecuencia
            if self.tuner.detected_freq > 0:
                self.freq_label.configure(text=f"{int(self.tuner.detected_freq)}Hz", text_color="#666")
            else:
                self.freq_label.configure(text="---Hz", text_color="#ccc")
            
            # Actualizar botones de cuerdas
            if self.auto_mode:
                for name, btn in self.string_buttons.items():
                    # Si ya está afinada, mantener verde
                    if name in self.tuned_strings:
                        btn.set_tuned(True)
                    else:
                        # Resaltar la cuerda actual
                        is_current = (name == self.tuner.current_string and self.tuner.is_stable)
                        btn.set_active(is_current)
                        btn.set_tuned(False)
        else:
            # Tuner detenido
            self.gauge.update_needle(0, False)
            self.gauge.animate()
            self.status_label.configure(text="---", text_color="#ccc")
            self.freq_label.configure(text="---Hz")
        
        # Actualizar resplandor de fondo si está activo
        if self.is_glowing:
            self.update_glow()
        
        # Llamar nuevamente después de 50ms
        self.after(50, self.update_display)
    
    def start_glow(self):
        """Inicia el efecto de resplandor verde en el fondo"""
        self.is_glowing = True
        self.glow_frame = 0
    
    def update_glow(self):
        """Actualiza el color del fondo con efecto de resplandor suave"""
        if self.glow_frame < 20:  # ~1 segundo
            self.glow_frame += 1
            
            # Calcular intensidad del resplandor (fade in y fade out)
            if self.glow_frame <= 10:
                # Fade in
                intensity = self.glow_frame / 10.0
            else:
                # Fade out
                intensity = (20 - self.glow_frame) / 10.0
            
            # Interpolar entre gris claro (#f5f5f5) y verde muy suave (#e8f5f0)
            base_r, base_g, base_b = 245, 245, 245
            target_r, target_g, target_b = 232, 245, 240
            
            r = int(base_r + (target_r - base_r) * intensity)
            g = int(base_g + (target_g - base_g) * intensity)
            b = int(base_b + (target_b - base_b) * intensity)
            
            color_bg = f"#{r:02x}{g:02x}{b:02x}"
            self.main_container.configure(fg_color=color_bg)
        else:
            # Terminar resplandor
            self.is_glowing = False
            self.main_container.configure(fg_color="#f5f5f5")


def main():
    app = TunerGUI()
    
    def on_closing():
        app.tuner.stop()
        app.destroy()
    
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()