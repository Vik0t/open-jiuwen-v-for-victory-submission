# openJiuwen contest agent

Агент решает задачи нескольких типов при общем бюджете. Для модельного пути используются нативные `Runner` и `ReActAgent` openJiuwen; вход и ответы передаются через каталог задач. Исходники и зависимости для сборки находятся в этом репозитории.

## Сборка

```bash
docker build -t jiuwen-contest .
```

Dockerfile фиксирует базовый образ и устанавливает зависимости из lock-файлов с проверкой хешей. Для сборки нужны доступ к registry и пакетным источникам либо подготовленный локальный cache.

## Проверка без модели

Используйте рабочую копию задач: агент может изменять исходники задач и записывает ответы.

```bash
export CONTEST_TASKS_DIR=/absolute/path/to/writable-tasks
docker run --rm --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --cpus=4 --memory=16g --pids-limit=128 \
  --tmpfs /tmp:rw,size=2g,mode=1777 \
  -v "$CONTEST_TASKS_DIR:/tasks:rw" -e TASKS=/tasks -e BUDGET=0 \
  jiuwen-contest --no-llm
```

Этот режим запускает локальные обработчики без обращений к модели. `BUDGET=0` — бюджет модельного пути, а не оценка качества решений.

## Запуск с моделью

Создайте env-файл **вне репозитория** с `LLM_BASE_URL`, `LLM_API_KEY`, `BUDGET` и при необходимости `LLM_MODEL`. Фактический бюджет задаёт среда запуска; он не зашит в образ. По умолчанию `LLM_MODEL=qwen3.8-27b-noreason`; при отсутствии `BUDGET` используется 0. Модельному пути требуется доступ к разрешённому LLM proxy.

```bash
export CONTEST_TASKS_DIR=/absolute/path/to/writable-tasks
export CONTEST_ENV_FILE=/absolute/path/to/private.env
docker run --rm --read-only --cap-drop ALL \
  --security-opt no-new-privileges --cpus=4 --memory=16g --pids-limit=128 \
  --tmpfs /tmp:rw,size=2g,mode=1777 --env-file "$CONTEST_ENV_FILE" \
  -v "$CONTEST_TASKS_DIR:/tasks:rw" -e TASKS=/tasks \
  jiuwen-contest
```

Docker entrypoint — `python /app/run_agent.py`. Вместо переменной `TASKS` можно передать `--tasks-dir /tasks` после имени образа. Ответ записывается в `answer.txt` рядом с соответствующим `task.json`; общий `agent-report.json` сохраняется в корне каталога задач. Стоимость модельных запросов учитывается как выходные токены плюс одна десятая входных.

Сведения о подготовленном базовом архиве — в [RELEASE.md](RELEASE.md).
