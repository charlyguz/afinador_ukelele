# Afinador de Ukelele y Guitarra

Afinador digital profesional con interfaz gráfica moderna, desarrollado como proyecto final para la clase de Procesamiento Digital de Señales en ESCOM-IPN.

## 🎯 Características

- **Detección de frecuencia precisa** usando FFT + Harmonic Product Spectrum (HPS)
- **Interfaz moderna** con gauge semicircular y animaciones suaves
- **Soporte dual**: Ukelele (4 cuerdas) y Guitarra (6 cuerdas)
- **Modo automático**: Detecta automáticamente la cuerda que estás tocando
- **Modo manual**: Selecciona una cuerda específica para afinar
- **Indicador visual intuitivo**: Low/Perfect/High con colores (rojo/verde/naranja)
- **Gauge semicircular**: Muestra la afinación de -80 a +80 cents con aguja animada

## 🚀 Instalación

```bash
pip install numpy scipy sounddevice customtkinter pillow pygame
```

## 📖 Uso

```bash
python afinador_pro.py
```

El afinador se inicia automáticamente al abrir la aplicación.

### Controles

- **Selector de instrumento** (esquina superior izquierda): Cambia entre 4-string (Ukelele) y 6-string (Guitarra)
- **Toggle Auto**: Activa/desactiva el modo de detección automática
- **Botones circulares**: En modo manual, haz clic en una cuerda para seleccionarla

### Afinación estándar

**Ukelele (4 cuerdas)**:
- G4: 392.00 Hz
- C4: 261.63 Hz
- E4: 329.63 Hz
- A4: 440.00 Hz

**Guitarra (6 cuerdas)** (preparada para implementación futura):
- E2: 82.41 Hz
- A2: 110.00 Hz
- D3: 146.83 Hz
- G3: 196.00 Hz
- B3: 246.94 Hz
- E4: 329.63 Hz

## 🎨 Diseño

La interfaz presenta:
- Fondo claro (#f5f5f5) con contraste moderno
- Gauge semicircular con marcas cada 10 cents
- Botones circulares minimalistas para las cuerdas
- Animaciones suaves en la aguja del gauge (factor de smoothing: 0.15)
- Indicador de estado grande y visible

## 🔧 Tecnologías

- **NumPy**: Operaciones matemáticas y manejo de arrays
- **SciPy**: Transformada rápida de Fourier (FFT)
- **SoundDevice**: Captura de audio en tiempo real del micrófono
- **CustomTkinter**: Framework moderno para interfaz gráfica
- **PIL/Pillow**: Manejo de imágenes (si se requieren assets)

## 📝 Parámetros de configuración

```python
SAMPLE_FREQ = 48000      # Frecuencia de muestreo
WINDOW_SIZE = 32768      # Tamaño de ventana FFT
WINDOW_STEP = 8192       # Paso de ventana
NUM_HPS = 5              # Armónicos para HPS
SMOOTH_ALPHA = 0.25      # Factor de suavizado de frecuencia
STABLE_FRAMES = 3        # Frames necesarios para detección estable
```

## 👥 Créditos

Proyecto Final - Procesamiento Digital de Señales  
ESCOM - Instituto Politécnico Nacional
