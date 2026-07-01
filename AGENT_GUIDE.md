# Гайд для агента (кратко)

**Репо:** https://github.com/LilVanya1/ript-mimii  
**Доступ:** collaborator уже выдан — клонируй своим GitHub-аккаунтом.

---

## 1. Старт

```bash
git clone https://github.com/LilVanya1/ript-mimii.git
cd ript-mimii
pip install -r requirements.txt
```

Открыть папку в **Cursor** → залогиниться в GitHub (свой аккаунт).

---

## 2. Что в репо / чего нет

| Есть в git | Нет в git (только у владельца локально) |
|------------|----------------------------------------|
| весь код `src/`, `app.py`, UI | `data/` — датасет MIMII |
| `models/` — веса `.pt`, registry | логи `*.log` |

Модели уже в репо. Датасет для обучения **не клонируется** — либо владелец шарит `data/` отдельно, либо качается через `src/download.py` / UI.

---

## 3. Ключевые файлы (менять тут)

- `src/autoencoder.py` — модель, train loop, threshold
- `src/dataset.py` — данные, split, cache
- `src/config.py` — дефолты (epochs, lr, batch…)
- `app.py` — Flask API: `/api/train`, `/api/tune`, `/api/autopilot`
- `templates/index.html` — GUI

---

## 4. Правила для агента

1. **Работать только внутри репо** — не лезть за пределы проекта.
2. **Обучение строго per `machine_id`** — не смешивать `id_00`, `id_02` и т.д.
3. **Качество > скорости**, цель AutoPilot: **AUC ≥ 0.89**.
4. HPO — **локальный Optuna**, без внешних LLM.
5. Не коммить: `data/`, `.env`, логи, секреты (ngrok token).
6. **Модели коммитить можно** — владелец хочет `models/` в git.
7. Перед пушем: `python -m compileall src app.py` (smoke).

---

## 5. Запуск

```bash
python app.py          # localhost:228
python app.py --public # + ngrok (нужен NGROK_AUTHTOKEN в env)
```

Или `start.bat` / `dev_start.bat` на Windows.

GUI: http://localhost:228 — кнопки **Обучить**, **AutoTune**, **AutoPilot**.

---

## 6. Git workflow

```bash
git pull                    # перед работой
# ... правки ...
git add .
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "fix: ..."
git push origin main
```

Владелец у себя: `git pull`.

---

## 7. API (если без GUI)

- `POST /api/train` — обучение
- `POST /api/tune` — Optuna
- `POST /api/autopilot` — tune → train → eval, стоп при target AUC

Параметры: `machine_type`, `machine_id`, `snr_db`.

---

## 8. Контекст задачи

- Датасет: **MIMII** (fan/pump/slider/valve)
- Задача: **anomaly detection** (one-class, pseudo-anomalies)
- Метрика: **ROC-AUC** на test split
- Известная проблема: AUC был ~0.5, шли фиксы архитектуры — смотри последние коммиты и `CASE_2_Acoustic_Diagnostics.md`

---

Если нужен датасет для train — попроси владельца скинуть папку `data/` (Syncthing/архив), в git её нет.

---

## 9. Удалённый deploy на машину владельца

После `git push` можно подтянуть код и перезапустить сервер владельца:

```bash
curl -X POST "https://PUBLIC_URL/api/deploy" \
  -H "Content-Type: application/json" \
  -H "X-Deploy-Token: SECRET_FROM_OWNER" \
  -d "{\"pull\": true, \"restart\": true}"
```

- `pull: true` — `git pull --ff-only` на ПК владельца
- `restart: true` — перезапуск `app.py` через ~2 сек
- Если идёт train/tune — вернёт `409 Busy`

**Токен** `DEPLOY_TOKEN` выдаёт владелец лично (в `.env`, не в git).

Владелец должен:
1. Скопировать `.env.example` → `.env`
2. Задать `DEPLOY_TOKEN` (длинная случайная строка)
3. Запустить `python app.py --public` (ngrok URL для агента)
4. Передать другу: **PUBLIC_URL** + **DEPLOY_TOKEN**

Типичный workflow агента:
1. Правки → `git push`
2. `POST /api/deploy` с токеном
3. Подождать ~5 сек → проверить `/api/state`