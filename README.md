# 📺 IPTV-Player: AI-Powered Streaming Experience

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![AI](https://img.shields.io/badge/AI-Evolutionary_Brain-orange.svg)

**IPTV-Player** es una plataforma avanzada de streaming que combina la reproducción de listas IPTV con un potente motor de **Inteligencia Artificial**. Diseñado para ser modular, eficiente y altamente personalizable, este sistema no solo reproduce contenido, sino que aprende y optimiza la estabilidad de los canales.

---

## 🚀 Características Principales

- **🧠 Evolutionary Brain (AI Core):** Un motor de IA que analiza metadatos, predice la calidad de los streams y gestiona el aprendizaje continuo mediante entrenamiento offline y online.
- **📊 AI Dashboard:** Visualización en tiempo real del estado del "cerebro", estadísticas de uso y métricas de rendimiento del sistema.
- **⚙️ Backend Automatizado:** Servidor robusto con un programador inteligente (i_scheduler) que realiza tareas de mantenimiento, actualización de listas y evolución del modelo sin intervención manual.
- **📁 Estructura Modular:** Separación clara entre frontend, backend y lógica de IA para facilitar la escalabilidad y el mantenimiento.
- **🛡️ Gestión Inteligente de Caché:** Sistema de persistencia dual para acceso ultrarrápido a datos críticos y recuperación de desastres.

---

## 📂 Estructura del Proyecto

`	ext
IPTV-Player/
├── frontend/           # Interfaces de usuario (Player, AI Dashboard)
├── backend/            # Servidor API y Programador de tareas
├── core/               # Motores de IA, Entrenamiento y Lógica Evolutiva
├── config/             # Configuración de búsqueda y Certificados SSL
├── cache/              # Almacenamiento local de modelos y bases de datos (Ignorado en Git)
├── logs/               # Registros de actividad y errores (Ignorado en Git)
└── scripts/            # Utilidades y herramientas de mantenimiento
`

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio:**
   `ash
   git clone https://github.com/tu-usuario/IPTV-Player.git
   cd IPTV-Player
   `

2. **Instalar dependencias:**
   `ash
   pip install -r requirements.txt
   `
   *(Asegúrate de tener un entorno virtual activo)*

3. **Configuración Inicial:**
   - Coloca tus archivos de búsqueda en config/.
   - Verifica las rutas relativas en ackend/iptv-server.py.

---

## 🖥️ Uso

Para iniciar el servidor principal:
`ash
python backend/iptv-server.py
`
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
