# Notas de Implementación - Nuevo Diseño

## Cambios Realizados

### 1. Estructura de Datos
- ✅ Agregado `GUITAR_TARGETS` con 6 cuerdas estándar
- ✅ Variable `CURRENT_INSTRUMENT` para cambiar entre ukelele/guitarra
- ✅ Función `get_current_targets()` para obtener diccionario activo
- ✅ Cambiado tema de dark a light

### 2. Nuevos Componentes

#### SemiCircleGauge
- Widget personalizado de gauge semicircular
- Rango: -80 a +80 cents
- Marcas cada 10 cents, etiquetas cada 20 cents
- Aguja animada con interpolación suave (factor 0.15)
- Colores dinámicos según afinación:
  - Verde (#00d9a5): Afinado (±5 cents)
  - Naranja (#ffa502): Agudo (> 5 cents)
  - Rojo (#ff4757): Grave (< -5 cents)

#### CircularStringButton
- Botones circulares de 100px diámetro
- Estado normal: fondo blanco, borde gris
- Estado activo: fondo negro, texto blanco
- Click handler para modo manual

### 3. Nueva Interfaz

#### Layout
- Ventana fija: 900x700px (relación ~4:3)
- Fondo claro: #f5f5f5
- Sin sidebar, diseño vertical centrado

#### Secciones
1. **Top Bar**: Selector de instrumento + Toggle Auto
2. **Gauge Section**: Gauge semicircular en contenedor blanco
3. **Status Section**: Indicador grande Low/Perfect/High + frecuencia
4. **String Buttons**: Grid 2x2 (ukelele) o 2x3 (guitarra)

### 4. Funcionalidades

#### Modo Auto vs Manual
- **Auto Mode** (default): Detecta automáticamente cualquier cuerda
- **Manual Mode**: Usuario selecciona cuerda específica, filtra detección

#### Selector de Instrumento
- "4-string": Modo ukelele con 4 botones (G, C, E, A)
- "6-string": Modo guitarra con 6 botones (E2, A2, D3, G3, B3, E4)

#### Inicio Automático
- El tuner se inicia automáticamente al abrir la app
- No requiere botón de inicio/detención

### 5. Animaciones
- Aguja del gauge: Interpolación lineal suave
- Formula: `current_angle += (target_angle - current_angle) * 0.15`
- Actualización cada 50ms (20 FPS)

### 6. Paleta de Colores

```python
# Fondo
BACKGROUND = "#f5f5f5"  # Gris claro
GAUGE_BG = "#ffffff"    # Blanco

# Estados de afinación
TUNED = "#00d9a5"       # Verde turquesa
SHARP = "#ffa502"       # Naranja
FLAT = "#ff4757"        # Rojo

# Texto
TEXT_PRIMARY = "#2c3e50"   # Gris oscuro
TEXT_SECONDARY = "#999"    # Gris medio
TEXT_DISABLED = "#ccc"     # Gris claro
```

## Pendientes / Mejoras Futuras

- [ ] Testing completo del modo guitarra (6 cuerdas)
- [ ] Agregar botón de inicio/pausa (opcional)
- [ ] Persistencia de configuración (última selección de instrumento)
- [ ] Modo de calibración (Concert Pitch personalizado)
- [ ] Soporte para afinaciones alternativas (Drop D, etc.)
- [ ] Historial de sesiones de afinación
- [ ] Export/Import de configuraciones
- [ ] Modo oscuro/claro toggle

## Notas Técnicas

### Performance
- Actualización cada 50ms es suficiente para respuesta fluida
- Factor de smoothing 0.15 balancea suavidad vs responsividad
- Canvas redraw es eficiente para gauge simple

### Portabilidad
- Todo dibujado con Canvas, sin imágenes externas
- CustomTkinter funciona en Windows/Mac/Linux
- SoundDevice detecta micrófono automáticamente

### Limitaciones Conocidas
- Ventana no redimensionable (by design)
- Rango de frecuencia: 150-600 Hz (suficiente para ukelele/guitarra)
- No hay visualización de waveform o espectro
