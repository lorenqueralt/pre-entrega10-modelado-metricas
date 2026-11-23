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

## Resultados (set de prueba)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.779 |
| Precision (churn) | 0.608 |
| Recall (churn) | 0.476 |
| F1-Score (churn) | 0.534 |

**Matriz de confusión:** 920 verdaderos negativos, 115 falsos positivos, **196 falsos
negativos** (clientes que se van y el modelo no detecta) y 178 verdaderos positivos.

El modelo acierta en el ~78% de los casos, pero su **recall sobre la clase churn es 0.48**:
deja escapar casi la mitad de los clientes que efectivamente se van. Ese es el número
crítico para el negocio.

## Conclusión (impacto en el negocio)

Para churn, la métrica más relevante es el **Recall de la clase "churn"**, porque el error
más caro es el **falso negativo**: un cliente que se va sin que el modelo lo haya alertado
es un cliente perdido (y retener cuesta mucho menos que adquirir uno nuevo). Un falso
positivo, en cambio, solo implica ofrecer una acción de retención a alguien que se iba a
quedar: un costo bajo y acotado.

En este caso el modelo deja **196 falsos negativos**: 196 clientes que se fueron sin que
el modelo los alertara. Por eso se prioriza **maximizar el recall** aun a costa de algo de
precision, y el **Accuracy (0.78) por sí solo resulta engañoso** —con un 26.5% de churn,
un modelo que dijera "nadie se va" tendría ~73% de accuracy y recall 0—. Si el negocio
necesita capturar más churners, se puede **bajar el umbral de decisión** (por debajo de
0.5) para subir el recall aceptando más falsos positivos, un trade-off favorable dado el
bajo costo del FP. El modelo se usa para generar una lista de clientes en riesgo y dirigir
la retención a los segmentos más churn-propensos (contratos *month-to-month*, `tenure`
bajo, fibra óptica, pago con *electronic check*).
