# Acoustic Diagnostics Project

Проект по акустической диагностике промышленного оборудования на основе звука.  
Основная задача: по аудио определить, является ли работа машины нормальной или аномальной, и при необходимости сохранить/использовать отдельные модели для разных экземпляров оборудования.

## Кейс

Кейс соответствует задаче предиктивного обслуживания:

- вход: аудиоокно или `.wav` файл;
- выход: оценка аномальности, порог и итоговый вердикт;
- тип задачи: в первую очередь `anomaly detection` на нормальных данных.

Используемый датасет и ориентир по постановке:

- MIMII Dataset
- MIMII baseline

## Технологический стек

- **Backend:** Python, Flask
- **DL/ML:** PyTorch
- **Audio processing:** librosa, soundfile
- **Метрики:** scikit-learn
- **Визуализация:** matplotlib, seaborn
- **Frontend:** HTML/CSS/vanilla JS
- **Локальный запуск:** Windows batch scripts

## Что реализовано

### 1. Веб-интерфейс

В `templates/index.html` реализованы:

- запуск обучения;
- выбор `machine_type`;
- выбор `machine_id` (`id_00`, `id_02`, `id_04`, `id_06`);
- выбор режима:
  - `new`
  - `finetune`
- настройка параметров обучения прямо из UI:
  - `epochs`
  - `batch_size`
  - `learning_rate`
  - `patience`
  - `anomaly_quantile`
  - `threshold_method` (`kde_fpr` / `mad` / `quantile`)
  - `threshold_target_fpr`
  - `threshold_mad_k`
- проверка пользовательских `.wav`;
- просмотр метрик, графиков и логов;
- выбор конкретной сохранённой модели.

### 2. Подготовка данных

В `src/dataset.py`:

- загрузка `.wav`;
- нарезка на окна;
- вычисление mel-спектрограмм;
- нормализация mel-спектрограмм в диапазон `[0, 1]`;
- предварительный расчёт признаков для ускорения train loop;
- честный split **по файлам**, а не по окнам, чтобы убрать leakage;
- фильтрация по `machine_id`.

### 3. Архитектура модели

Текущая архитектура в `src/autoencoder.py`:

- **depthwise-separable autoencoder**
- блоки `depthwise conv + pointwise conv`
- без `MLP` / `Linear` bottleneck слоёв
- компактная сверточная модель
- `Sigmoid()` на выходе
- обучение по `L1Loss`
- scheduler `ReduceLROnPlateau`

Преимущество текущей версии:

- меньше параметров;
- ниже риск переобучения на монотонных аудиоданных;
- лучше подходит под спектрограммы, чем тяжёлый fully-connected bottleneck.

### 4. Логика обнаружения аномалий

Подход:

1. обучаем автоэнкодер на нормальных примерах;
2. на инференсе считаем reconstruction error;
3. по ошибке и выбранному методу калибровки получаем threshold:
   - `kde_fpr` (default): KDE + целевой FPR на normal-val
   - `mad`: median + k * MAD
   - `quantile`: классический q-квантиль
4. если ошибка выше порога, считаем сигнал аномальным.

### 5. Метрики и анализ

В проекте используются:

- **AUC-ROC**
- **pAUC**
- `accuracy`
- `F1`

Также добавлен полезный диагностический лог:

- `normal_error_mean`
- `abnormal_error_mean`
- `error_delta = abnormal - normal`

Это помогает понять, реально ли модель разделяет норму и аномалию.

### 6. Per-ID обучение

Один из ключевых выводов проекта:

- `id_00`, `id_02`, `id_04`, `id_06` — это не “каждый отдельный звук”, а **разные экземпляры одной и той же машины**;
- для MIMII лучше обучать **отдельную модель на каждый `machine_id`**, а не одну общую на все ID.

Примеры:

- `fan/id_00` → отдельная модель
- `fan/id_02` → отдельная модель
- `fan/id_04` → отдельная модель
- `fan/id_06` → отдельная модель

### 7. История и версии моделей

Реализовано:

- реестр моделей: `models/model_registry.json`
- история чекпоинтов: `models/history/`
- хранение метаданных:
  - `machine_type`
  - `machine_id`
  - `snr`
  - `mode`
  - `auc_roc`
  - `pauc`
  - `threshold`
  - `epochs_trained`

## Структура проекта

```text
app.py
src/
  autoencoder.py
  dataset.py
  evaluate.py
  classifier.py
  ocsvm.py
  config.py
templates/
  index.html
models/
  history/
  model_registry.json
results/
README.md
CASE_2_Acoustic_Diagnostics.md
```

## Как запускать

### Обычный запуск

```bash
pip install -r requirements.txt
python app.py
```

или через:

- `start.bat`
- `start_public.bat`

После запуска:

- открыть [http://127.0.0.1:228](http://127.0.0.1:228)

### Dev-режим

Для локальной разработки добавлены:

- `dev_start.bat`
- `dev_stop.bat`
- `dev_restart.bat`

`dev_start.bat` включает авто-релоад сервера при изменении кода.

## Рекомендуемые настройки

Стартовые настройки для `fan`:

- `machine_id`: один конкретный (`id_00` / `id_02` / `id_04` / `id_06`)
- `epochs = 60`
- `batch_size = 128`
- `learning_rate = 0.0003`
- `patience = 10–15`
- `anomaly_quantile = 0.995`
- `threshold_method = kde_fpr`
- `threshold_target_fpr = 0.05`
- `threshold_mad_k = 3.0`

Если цель — качество, лучше запускать:

- `new`
- на одном `machine_id`
- без смешивания всех ID в одну модель

## Что ещё есть в проекте

- One-Class SVM baseline: `src/ocsvm.py`
- Logistic Regression baseline: `src/ocsvm.py`
- latent classifier: `src/classifier.py`

Это вспомогательные модули, основной сценарий проекта сейчас построен вокруг автоэнкодера.

## Источники и паттерны

Использованные ориентиры:

- MIMII Dataset: https://arxiv.org/abs/1909.09347
- MIMII baseline: https://github.com/MIMII-hitachi/mimii_baseline
- DCASE workshop paper: https://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Purohit_21.pdf
- PyTorch tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

Архитектурные идеи, на которые опирались:

- reconstruction-error anomaly detection;
- file-level split без leakage;
- per-domain/per-machine-id обучение;
- depthwise separable conv pattern;
- scheduler-based стабилизация обучения.

## Краткий вывод

Проект представляет собой рабочий прототип акустической диагностики:

- есть обучение;
- есть инференс;
- есть история моделей;
- есть per-ID модели;
- есть настройка train-конфига через UI;
- есть визуализация и базовые метрики качества.

Основной practical insight проекта:

> для MIMII качество очень зависит не только от архитектуры, но и от раздельного обучения по `machine_id` и честного split по файлам.
