# OCR Classifier — AGENTS.md

## Workflow Rules (соблюдают ВСЕ агенты)

1. **Перед началом задачи:** использовать MCP Serena для навигации по кодовой базе
   и понимания точек изменений.
2. **Перед написанием кода:** вызвать Context7 для актуальной документации
   используемых пакетов (`gosseract`, `net/http`, `image`).
3. **Перед коммитом:** убедиться, что проходят все проверки (см. раздел Checks).
4. **Нельзя** изменять сигнатуру `/classify` без явного требования в задаче.
5. **Нельзя** рефакторить код без наличия тестов на изменяемый участок.

## Checks (обязательные перед сдачей)

```bash
go build -o ocr-classifier ./cmd/server   # должен завершиться без ошибок
go vet ./...                              # нет предупреждений
go test -race -cover ./...               # все тесты зелёные, coverage не падает
golangci-lint run                        # нет новых замечаний
```

## Coding Agent
**Роль:** реализация новой функциональности и улучшение качества кода

**Правила:**
- Code style: `gofmt` + `golangci-lint` (конфиг `.golangci.yml` в корне)
- Обработка ошибок: обёртка через `fmt.Errorf("...: %w", err)`, без `panic`
- Все публичные функции должны иметь godoc-комментарий
- Не добавлять зависимости без обоснования в PR-описании

## Test Agent
**Роль:** поддержка и расширение тестового покрытия

**Правила:**
- Стиль тестов: table-driven (`[]struct{ name, input, want }`)
- Датасеты: `test/dataset/{lang}/{case}.jpg`, описание в `test/dataset/README.md`
- Регрессия: точность классификации не должна падать ниже X% (определить при первом baseline)
- Команда с покрытием: `go test -race -coverprofile=coverage.out ./...`

## Documentation Agent
**Роль:** актуализация технической документации

**Правила:**
- Техническая документация - в папке `/docs`, пользовательская - в README.md
- Первичный язык документации: русский
- Диаграммы: формат Mermaid
- При изменении API обязательно обновить README.md

## Refactoring Agent
**Роль:** улучшение структуры и производительности

**Правила:**
- Рефакторинг только при наличии тестов на изменяемый код
- Целевые метрики: p95 latency `/classify` < 5000ms, memory < 100MB per request
- Никаких breaking changes в публичном API без явного требования

## Commands

### Build
```bash
go build -o ocr-classifier ./cmd/server
```

### Lint
```bash
golangci-lint run
```

### Test
```bash
go test -race -cover ./...
```

### Run
```bash
OCR_PORT=8080 ./ocr-classifier
```

### Health check
```bash
curl http://localhost:8080/health
```

### Classify (usage example)
```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @test/dataset/eng/lightbulb-scheme.jpg \
  http://localhost:8080/classify
```