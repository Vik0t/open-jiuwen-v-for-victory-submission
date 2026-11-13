# openJiuwen contest agent

Агент решает задачи при общем бюджете; модельный путь использует нативные Runner и ReActAgent openJiuwen. Задачи и ответы передаются через каталог задач.

## Сборка

```bash
docker build -t jiuwen-contest .
```

Dockerfile использует зафиксированный базовый образ и lock-файлы с проверкой хешей. Для сборки нужны доступ к источникам зависимостей либо подготовленный локальный cache.

## Проверка без модели

Используйте рабочую копию задач: агент может изменять их исходники и записывать ответы.

```bash
export CONTEST_TASKS_DIR=/absolute/path/to/writable-tasks
docker run --rm --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --cpus=4 --memory=16g --pids-limit=128 \
  --tmpfs /tmp:rw,size=2g,mode=1777 \
  -v "$CONTEST_TASKS_DIR:/tasks:rw" -e TASKS=/tasks -e BUDGET=0 \
  jiuwen-contest --no-llm
```

## Запуск с моделью

Создайте env-файл вне репозитория с `LLM_BASE_URL`, `LLM_API_KEY`, `BUDGET` и при необходимости `LLM_MODEL`. Бюджет задаётся средой запуска; по умолчанию он равен 0. Модель по умолчанию — `qwen3.8-27b-noreason`. Для модельного пути необходим доступ к разрешённому LLM proxy.

```bash
export CONTEST_TASKS_DIR=/absolute/path/to/writable-tasks
export CONTEST_ENV_FILE=/absolute/path/to/private.env
docker run --rm --read-only --cap-drop ALL \
  --security-opt no-new-privileges --cpus=4 --memory=16g --pids-limit=128 \
  --tmpfs /tmp:rw,size=2g,mode=1777 --env-file "$CONTEST_ENV_FILE" \
  -v "$CONTEST_TASKS_DIR:/tasks:rw" -e TASKS=/tasks jiuwen-contest
```

Entrypoint — `python /app/run_agent.py`. Вместо `TASKS` можно передать `--tasks-dir /tasks`. Ответ сохраняется в `answer.txt` соответствующей задачи, общий отчёт — в `agent-report.json` в корне каталога задач. Стоимость запросов учитывается как выходные токены плюс одна десятая входных. Сведения о выпуске — в [RELEASE.md](RELEASE.md).
