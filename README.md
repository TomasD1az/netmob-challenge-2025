# NetMob 2025 – Ad-Placement Optimization with Mobility Data

Este repositorio contiene el código y los experimentos realizados para el proyecto presentado en el desafío de datos de NetMob 2025, enfocado en la optimización de campañas publicitarias geolocalizadas a partir de datos de movilidad y segmentación socio-demográfica.

## Resumen del proyecto

Se desarrolló un modelo predictivo que, usando datos anonimizados de movilidad en la región de Île-de-France (París y alrededores), estima la probabilidad de que individuos de ciertos perfiles socio-demográficos visiten diferentes zonas urbanas.

A partir de esto, se generan mapas de calor probabilísticos por segmento, útiles para campañas de marketing dirigidas y ubicaciones estratégicas de publicidad.

## Estructura de datos

- Fuente: Encuesta EMG 2023
- Tamaño: 3.337 participantes, una semana de seguimiento GNSS
- Datos utilizados:
  - Información socio-demográfica
  - Trazas GNSS (coordenadas GPS)
  - Detalles de viajes (no utilizados en esta etapa)

## Metodología

1. Preprocesamiento:
   - Filtrado de puntos GPS dentro de París
   - Limpieza de datos faltantes y codificación de variables categóricas

2. Construcción del target:
   - Ajuste de un modelo GMM (Gaussian Mixture Model) por usuario
   - División de París en una grilla 50x50
   - Probabilidad de visita a cada celda como variable objetivo

3. Ingeniería de features:
   - Aplicación de NMF (Non-negative Matrix Factorization) sobre datos demográficos
   - Generación de 10 nuevas variables latentes

4. Entrenamiento del modelo:
   - Modelo: XGBoost
   - Métrica: RMSE
   - Validación cruzada con K-Fold sin fuga de usuarios entre sets

5. Validación:
   - Comparación entre mapas predichos y distribuciones GMM reales
   - Evaluación a nivel de cobertura espacial (no de trayectorias exactas)

## Casos de estudio

Se realizaron simulaciones de campañas para dos perfiles distintos:

- Compradores potenciales de BMW: Hombres mayores de 50 con empleos ejecutivos
- Familias con adultos mayores: Personas de 30 o más años que viven con alguien de 65 o más

En ambos casos se generaron mapas agregados de movilidad para guiar decisiones publicitarias en zonas de mayor exposición.

## Trabajo futuro

- Extensión del análisis a toda la región Île-de-France
- Incorporación de modelos agregados por clústers de usuarios
- Posible integración de un modelo de movilidad tipo Deep Gravity Model extendido

## Contenido del repositorio

- `notebooks/`: notebooks con análisis exploratorio, construcción del target y visualizaciones
- `models/`: scripts de entrenamiento y validación del modelo
- `data/`: scripts de carga y preprocesamiento (sin datos originales por motivos de privacidad)
- `reports/`: informe PDF y figuras generadas
- `README.md`: este archivo

## Autores

- Santiago M. Barrón Bucolo  
- Tomás Díaz  
- Jeremías Figueiredo Paschmann  
- Juan Kaplan  
- Francisco Nattero  
- Mariano G. Beiró  

Proyecto desarrollado en el marco de la Universidad de San Andrés y CONICET (Argentina).
