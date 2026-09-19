# Optimizador Black-Litterman

Aplicación Streamlit para construir portafolios de acciones y ETFs con el modelo **Black-Litterman**. Toma los datos de **Yahoo Finance** y trabaja en **USD** como moneda base.

El flujo es el siguiente: los rendimientos de equilibrio del mercado se combinan con tus opiniones (views) para obtener rendimientos posteriores. Con ellos se optimiza un portafolio media-varianza bajo restricciones realistas, se mide su riesgo y se valida con un backtest walk-forward.

---

## Características

| Bloque | Opciones |
|---|---|
| **Universo** | De 2 a 10 activos. Se eligen de un catálogo con búsqueda (acciones de EE. UU., emisoras de la BMV y ETFs) o se escribe cualquier ticker de Yahoo Finance. Incluye cinco universos sugeridos que se cargan con un clic. Con 1 activo, la app muestra solo el análisis individual. |
| **Datos** | Precios de cierre ajustados. Conversión automática a USD con `{CCY}=X` y manejo de subunidades (GBp, ILA, ZAc). Alineación de calendarios entre mercados y filtro de calidad por datos faltantes. |
| **Periodo** | Botones de 1, 3, 5 (por defecto) y 10 años, o fechas personalizadas. Rendimientos diarios anualizados con 252 días. |
| **Tasa libre de riesgo** | T-Bill a 13 semanas (`^IRX`), con el último dato o el promedio del periodo, o un valor manual. |
| **Pesos de equilibrio w_mkt** | Capitalización de mercado (por defecto), 1/N o pesos manuales. Si falta la capitalización de algún ticker, se usa 1/N como respaldo automático. |
| **Aversión al riesgo δ** | 2.5 fijo (por defecto), implícito del índice de referencia (acotado y con advertencia) o manual. |
| **Covarianza Σ** | Ledoit-Wolf con correlación constante (por defecto), muestral o EWMA (RiskMetrics, λ ajustable). Incluye número de condición y reparación PSD. |
| **τ** | 0.05 (por defecto), 1/T con T en años de datos, o manual. |
| **Views** | Absolutas, relativas y de canasta (con ponderación igual o por capitalización), armadas con un constructor guiado: tipo, activos, rendimiento esperado y confianza. Se pueden activar, editar o eliminar desde una tabla. |
| **Ω** | Idzorek con confianza en % (por defecto), He-Litterman o intervalos de confianza. |
| **Optimización** | Máximo de μ_BLᵀw − (δ/2)·wᵀΣw, con Σ_BL posterior (por defecto) o la Σ histórica. |
| **Restricciones** | Presupuesto (totalmente invertido, sin apalancamiento o libre), solo largos, rango de peso por activo con un control deslizante (o personalizado por activo), límites por sector con grupos asignados automáticamente y rotación máxima (turnover). |
| **Riesgo** | Ex-ante: rendimiento, volatilidad, Sharpe, VaR y CVaR paramétricos, contribución al riesgo (Euler), ratio de diversificación y N efectivo. Ex-post: VaR y CVaR históricos, VaR Cornish-Fisher, beta, tracking error y drawdown. |
| **Backtest** | Walk-forward con rebalanceo mensual o trimestral, ventana de estimación y costos de transacción configurables. Compara cuatro estrategias: BL con views, equilibrio, 1/N e índice. |
| **Exportación** | Excel con todas las tablas y matrices, y reporte PDF con metodología, resultados y gráficas. |
| **Diseño** | Encabezado institucional, tarjetas de indicadores, barra lateral organizada por secciones y paleta azul. La app carga los datos automáticamente al abrirse. |

---

## Metodología

1. **Equilibrio (optimización inversa).** Π = δ Σ w_mkt, expresado en exceso de r_f.
2. **Views.** P μ = Q + ε, con ε ~ N(0, Ω).
   - En las views absolutas se captura el rendimiento total y el modelo usa Q − r_f.
   - En las relativas y en las canastas neutrales, r_f se cancela.
3. **Posterior (He & Litterman, 1999).**
   - μ_BL = Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ)
   - Σ_BL = Σ + τΣ − τΣPᵀ(PτΣPᵀ + Ω)⁻¹PτΣ

   Esta forma admite confianza del 100 % (Ω = 0) y permite descomponer de forma exacta el aporte de cada view a μ_BL y a los pesos.
4. **Ω.**
   - He-Litterman: ω_k = p_k τΣ p_kᵀ.
   - Idzorek (2005): ω_k = ((1 − c_k)/c_k) · p_k τΣ p_kᵀ. Es la solución cerrada equivalente a la calibración numérica original.
   - Intervalos: ω_k = ((U − L)/(2·z))².
5. **Optimización convexa** con `cvxpy` (CLARABEL, con OSQP y SCS como respaldo).

**Controles de consistencia incluidos:**
- Sin views y sin restricciones, el portafolio óptimo reproduce w_mkt (con Σ_BL, w* = w_mkt/(1+τ) antes de normalizar).
- La confianza implícita de cada view coincide con la capturada cuando se usa Idzorek.

---

## Estructura del repositorio

El modelo completo está en un solo archivo, así que el repositorio solo necesita tres archivos en la raíz:

```
├── app.py            # Modelo Black-Litterman + aplicación Streamlit (7 pestañas)
├── requirements.txt  # Dependencias
└── README.md         # Este documento
```

`app.py` está organizado en secciones: configuración, datos, covarianza, equilibrio, views, posterior, optimización, orquestador, riesgo, backtest, exportación, gráficas y la interfaz Streamlit.

---

## Publicar en GitHub (desde el navegador)

1. En GitHub, crea un repositorio nuevo (por ejemplo `estudio-black-litterman`).
2. Entra a **Add file → Upload files**, arrastra `app.py`, `requirements.txt` y `README.md`, y pulsa **Commit changes**.
   Otra opción es **Add file → Create new file**: escribes el nombre del archivo, pegas su contenido y confirmas.
3. Verifica que los tres archivos queden en la **raíz** del repositorio, no dentro de una carpeta.

## Publicar en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) y elige **Create app**.
2. Selecciona el repositorio, la rama `main` y el archivo principal `app.py`.
3. En **Advanced settings** elige **Python 3.11** y pulsa **Deploy**.
4. Si actualizas un archivo en GitHub, la app se vuelve a desplegar sola. Si no lo hace, usa **Manage app → Reboot app**.

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Para probar la app sin conexión a Yahoo Finance, define la variable de entorno `BL_DEMO_DATA=1`. Con ella, la app usa datos sintéticos reproducibles. En producción no se define.

---

## Flujo de trabajo en la app

1. **Barra lateral: universo.** Elige un universo sugerido o selecciona activos del catálogo. Después elige el índice de referencia y el periodo, y pulsa **Actualizar datos**. Al abrir la app se carga automáticamente el universo EE. UU. + México.
2. **Barra lateral: modelo.** Todos los parámetros se eligen con un clic: pesos de equilibrio, δ, Σ, τ, Ω, la covarianza para optimizar, el presupuesto y si se permiten solo posiciones largas.
3. **Tarjetas de resumen.** Muestran el rendimiento esperado, la volatilidad, el Sharpe, el VaR y los parámetros del portafolio óptimo, comparados con el mercado.
4. **Datos.** Divisas, capitalizaciones, calidad de datos, precios en USD y estadísticos históricos.
5. **Equilibrio.** δ, τ, r_f, Π frente al rendimiento histórico, diagnóstico de Σ y correlaciones.
6. **Views.** Constructor guiado y tabla de views registradas, con el impacto de cada una en μ_BL y en los pesos.
7. **Optimización.** Restricciones con interruptores y controles deslizantes, portafolio óptimo y fronteras eficientes.
8. **Riesgo.** VaR, CVaR, contribución al riesgo y métricas ex-post.
9. **Backtest.** Evaluación walk-forward con rebalanceo, ventana y costos configurables.
10. **Exportar.** Excel y reporte PDF.

## Cómo capturar una view

| Tipo | Qué eliges | Ejemplo |
|---|---|---|
| Absoluta | Un activo y su rendimiento total anual esperado | "AAPL rendirá 12 % anual" |
| Relativa | El activo que supera, el activo superado y el diferencial | "MSFT superará a NVDA por 3 %" |
| Canasta | Una canasta larga y, opcionalmente, una corta, con ponderación igual o por capitalización | "AAPL + MSFT superarán a JPM por 2 %" |

La **confianza (%)** se usa con el método de Idzorek: es la fracción del camino entre el equilibrio y tu view que recorre el modelo. Con el método de intervalos, el constructor pide además un margen ± alrededor del rendimiento esperado.

---

## Limitaciones y buenas prácticas

- **Yahoo Finance** es una fuente gratuita no oficial. Puede tener huecos, ajustes tardíos o límites de consultas; la app guarda las descargas en caché durante una hora.
- **Capitalización de mercado:** Yahoo solo ofrece la actual, por lo que el backtest con w_mkt por capitalización tiene sesgo de anticipación. La app lo advierte.
- **Views en el backtest:** aplicar las views actuales al pasado es look-ahead. Por eso se reporta también la estrategia de equilibrio sin views.
- **Mercados con horarios distintos:** con datos diarios, los cierres no sincrónicos (por ejemplo BMV frente a NYSE) pueden subestimar las correlaciones.
- **Feriados:** el precio se mantiene constante como máximo 5 días hábiles.

## Referencias

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios.* Goldman Sachs.
- Idzorek, T. (2005). *A Step-by-Step Guide to the Black-Litterman Model.*
- Ledoit, O. & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix.* Journal of Portfolio Management.
- Walters, J. (2014). *The Black-Litterman Model in Detail.*
- J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document.*

---

> Herramienta analítica con fines educativos y de investigación. No constituye asesoría de inversión.
