# 🌤️ Seattle Weather Calendar & Predictor

Este proyecto es una aplicación interactiva desarrollada en **Streamlit** que analiza registros climáticos históricos de la ciudad de Seattle y predice las condiciones meteorológicas del día a día para cualquier mes del año 2025, utilizando un modelo de Machine Learning supervisado. Además, presenta un visualizador anual estilizado de temperaturas promedio históricas.

El proyecto ha sido refactorizado por completo bajo una arquitectura de **4 capas independientes (estancas)** para garantizar alta mantenibilidad, modularidad y facilidad de pruebas.

---

## 🏗️ Arquitectura del Software (4 Capas)

La aplicación sigue el principio de separación de responsabilidades para evitar el acoplamiento:

```
[ Capa de UI (calendario.py) ]
             │
             ▼
[ Capa de Lógica de Negocio (src/logic/weather_forecaster.py) ]
             │
             ├───► [ Capa de Servicios/Adaptadores (src/services/ml_service.py) ]
             │                                              │
             │                                              ▼
             │                                     [ scikit-learn SDK ]
             │
             ▼
[ Capa de Datos (src/data/weather_repository.py) ]
             │
             ▼
[ Seattle Weather CSV Dataset ]
```

1. **Capa de Interfaz de Usuario (UI)** (`calendario.py`):
   Solo renderiza componentes visuales y estados de Streamlit. No procesa datos crudos ni realiza entrenamiento directo. Optimizado con memoización (`@st.cache_data` y `@st.cache_resource`) para lograr una ejecución instantánea.
2. **Capa de Lógica de Negocio (Logic)** (`src/logic/`):
   Funciones puras de cálculo (conversión de estaciones, cálculo de métricas mensuales y perturbación de simulaciones meteorológicas). Es agnóstica de frameworks visuales y de las librerías de ML.
3. **Capa de Servicios y Adaptadores (Services)** (`src/services/`):
   Envuelve las llamadas a la biblioteca externa `scikit-learn` (`RandomForestClassifier` e `SimpleImputer`) aplicando el **Adapter Pattern**. De este modo, si se cambia de modelo o librería en el futuro, la lógica de negocio y la UI permanecen intactas.
4. **Capa de Datos (Data)** (`src/data/`):
   Carga los registros del archivo CSV y valida su estructura interna, abstrayendo el acceso a archivos del resto del sistema.

---

## ✨ Características Principales

- **Modelo RandomForest Optimizado**: Ajustado con balanceo de pesos de clases para compensar el desequilibrio en climas raros (como nieve o llovizna).
- **Simulación Meteorológica Realista**: Al pronosticar un mes, simula condiciones diarias partiendo de las medianas históricas e integrando variabilidad física (perturbación aleatoria controlada) y acumulados de promedios móviles de 7 días (`temp_avg_7d`, `precip_7d`).
- **Control de Calendario Dinámico**: El software genera el número exacto de días según el mes del calendario (28 días para febrero, 31 para marzo, etc.).
- **Interfaz de Usuario Estilizada**: Panel temático con tarjetas de clima adaptativas basadas en la temperatura (azul para frío, naranja para templado, rojo para cálido) y gráficos de distribución interactivos.
- **Manejo de Estados Robusto**: Visualización de los 4 estados principales (Ideal con datos, Cargando con spinners interactivos, Alertas de datos vacíos informativos y de Errores controlados).

---

## 📁 Estructura del Proyecto

```text
├── .gitignore                      # Exclusión de archivos y entornos pesados
├── EJECUTAR.bat                    # Script ejecutable portable para Windows
├── ejecutar.txt                    # Instrucciones de ejecución
├── calendario.py                   # Script principal de la UI (Streamlit)
├── seattle_weather_clean.csv        # Dataset de entrenamiento
├── requirements.txt                # Dependencias del proyecto
└── src/                            # Código fuente modularizado
    ├── data/
    │   ├── __init__.py
    │   └── weather_repository.py   # Repositorio de lectura de CSV
    ├── logic/
    │   ├── __init__.py
    │   └── weather_forecaster.py   # Lógica pura del pronosticador
    ├── services/
    │   ├── __init__.py
    │   └── ml_service.py           # Adaptador de scikit-learn
    └── tests/
        ├── __init__.py
        └── test_weather_forecaster.py # Pruebas unitarias de cobertura lógica
```

---

## 🚀 Instrucciones de Ejecución

### Opción 1: Ejecución Rápida (Entorno Portable en Windows)
Si cuentas con la carpeta `pythonport` descargada localmente, simplemente haz doble clic en el archivo:
```bash
EJECUTAR.bat
```
El script detectará automáticamente el Python portable y levantará el servidor local de Streamlit. Si no lo encuentra, buscará automáticamente si tienes Python instalado globalmente en tu sistema.

### Opción 2: Ejecución Manual (Cualquier Plataforma)
1. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
2. Inicia la aplicación de Streamlit:
   ```bash
   streamlit run calendario.py
   ```

---

## 🧪 Pruebas Unitarias

La lógica de negocio cuenta con un 100% de cobertura en tests unitarios. Para ejecutarlas de forma automática, corre el siguiente comando en la raíz del proyecto:
```bash
python -m unittest discover -s src/tests
```
