# Pre-entrega · Modelado y Evaluación de Clasificación (Churn)

**Curso:** Data Science II — Coderhouse

## Problema

Predecir el **abandono de clientes (churn)** de una empresa de telecomunicaciones a partir
de sus datos de contrato y consumo. Es un problema de **clasificación binaria**
(`Churn` = Yes/No). El objetivo del negocio es **detectar a tiempo a los clientes en
riesgo** para accionar campañas de retención.

## Dataset

**Telco Customer Churn** (IBM Sample Data) — 7.043 clientes, 21 variables.
Fuente: Kaggle, https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Incluye datos demográficos (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), de
servicios contratados (`InternetService`, `TechSupport`, `StreamingTV`, etc.), de contrato
y facturación (`Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `tenure`) y
la variable objetivo `Churn`.

> **Cómo obtener los datos:** descargar el CSV de Kaggle y guardarlo como
> `telco_churn.csv`, junto al notebook (o ajustar la variable `CSV` en la primera celda).

## Estructura

```
preentrega10-modelado-metricas/
├── modelado_metricas.ipynb   # notebook con el modelado y las métricas
├── telco_churn.csv           # dataset de Kaggle (agregar al descargarlo)
├── requirements.txt
├── .gitignore
└── README.md
```

## Cómo ejecutar

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook modelado_metricas.ipynb   # Kernel -> Run All
```

## Metodología

1. **Limpieza:** `TotalCharges` a numérico (11 celdas en blanco → nulos), se descarta
   `customerID` y se convierte `Churn` a binario.
2. **División Train/Test 80/20** estratificada (antes de cualquier ajuste → evita leakage).
3. **Pipeline:** preprocesamiento (imputación + escalado + One-Hot) + **Random Forest**.
4. **Evaluación:** Accuracy, Precision, Recall, F1 y matriz de confusión (heatmap).
5. **Interpretación** de negocio centrada en el costo de los falsos negativos.

## Resultados

Las métricas se calculan en el notebook sobre el set de prueba (Accuracy, Precision,
Recall, F1 y matriz de confusión). El foco del análisis está en el **recall de la clase
churn** y en la cantidad de **falsos negativos** (clientes que se van sin ser detectados).

## Conclusión (impacto en el negocio)

Para churn, la métrica más relevante es el **Recall de la clase "churn"**, porque el error
más caro es el **falso negativo**: un cliente que se va sin que el modelo lo haya alertado
es un cliente perdido (y retener cuesta mucho menos que adquirir uno nuevo). Un falso
positivo, en cambio, solo implica ofrecer una acción de retención a alguien que se iba a
quedar: un costo bajo y acotado.

Por eso se prioriza **maximizar el recall** aun a costa de algo de precision, y el
**Accuracy por sí solo resulta engañoso** con clases desbalanceadas. El modelo se usa para
generar una lista de clientes en riesgo y dirigir la retención a los segmentos más
churn-propensos (contratos *month-to-month*, `tenure` bajo, fibra óptica, pago con
*electronic check*).
