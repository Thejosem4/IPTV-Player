# 📺 IPTV-Player: AI-Powered Streaming Experience

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![AI](https://img.shields.io/badge/AI-Evolutionary_Brain-orange.svg)

**IPTV-Player v7.2** es una plataforma avanzada de streaming que combina la reproducción de listas IPTV con un potente motor de **Inteligencia Artificial**. Diseñado para ser modular y ultra-rápido, este sistema usa pipelines asíncronos para streaming y aprende de tu red para optimizar la estabilidad de los canales.

---

## 🚀 Características Principales (v7.2)

- **⚡ Motor VOD Asíncrono:** Nuevo pipeline usando `aiohttp` que transmite películas pesadas directamente al socket sin consumir memoria RAM extra.
- **🛡️ Privacidad Local y Auto-Entrenamiento:** El cerebro de la IA (`core/iptv_ai_core.py`) se distribuye **virgen**. No pre-cargamos datos. La red neuronal aprende localmente de tu proveedor de internet y tus listas, garantizando que tus tokens y hábitos de visualización nunca salgan de tu computadora.
- **🧠 Evolutionary Brain (AI Core):** Un motor de IA que engaña sistemas antibots modificando heurísticas (Jitter, Headers) y predice la calidad de los streams para evitar buffers.
- **🔄 Proxy Resiliente:** Compartición de flujos HLS. Si varios reproductores piden el mismo canal, el servidor reutiliza el stream FFmpeg activo ahorrando ancho de banda.
- **⚙️ Backend Automatizado:** Servidor robusto con un programador inteligente que actualiza listas comprimidas (gzip) sin interrumpir los streams en vivo.

---

## 📂 Estructura del Proyecto

```text
IPTV-Player/
├── frontend/           # Interfaces de usuario (Player, AI Dashboard)
├── backend/            # Servidor API y Programador de tareas
├── core/               # Motores de IA, Entrenamiento y Lógica Evolutiva
├── config/             # Configuración de búsqueda y Certificados SSL
├── cache/              # Almacenamiento local de modelos y bases de datos (Ignorado en Git)
├── logs/               # Registros de actividad y errores (Ignorado en Git)
└── scripts/            # Utilidades y herramientas de mantenimiento
```

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/IPTV-Player.git
   cd IPTV-Player
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Asegúrate de tener un entorno virtual activo)*

3. **Configuración Inicial:**
   - Coloca tus archivos de búsqueda en `config/`.
   - Verifica las rutas relativas en `backend/iptv-server.py`.

---

## 🖥️ Uso

Para iniciar el servidor principal:
```bash
python backend/iptv-server.py
```
Accede a la interfaz web a través de tu navegador en http://localhost:5000 (o el puerto configurado).

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el modelo de IA o la interfaz de usuario:
1. Haz un Fork del proyecto.
2. Crea una rama para tu característica (git checkout -b feature/NuevaMejora).
3. Haz un commit de tus cambios (git commit -m 'Añade nueva mejora').
4. Haz un Push a la rama (git push origin feature/NuevaMejora).
5. Abre un Pull Request.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

---
*Desarrollado con ❤️ para la comunidad de streaming e Inteligencia Artificial.*
