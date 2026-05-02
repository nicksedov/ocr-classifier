# OCR Classifier

HTTP-сервис для определения наличия текста на изображениях с использованием Tesseract OCR. Возвращает детальную информацию о найденном тексте, включая показатели уверенности (confidence), координаты текстовых блоков и количество найденных токенов. Ответ содержит вердикт, является ли данное изображение документом (true/false), который выносится на основе правил.

## Требования

- Go 1.21+
- Tesseract OCR (должен быть установлен в системе)

### Установка Tesseract

**Windows:**
```bash
# Установить через Chocolatey
choco install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Сборка

```bash
go build -o ocr-classifier ./cmd/server
```

## Запуск

```bash
# По умолчанию сервер запускается на порту 8080
./ocr-classifier

# Для изменения порта используйте переменную окружения PORT
PORT=3000 ./ocr-classifier
```

## API

Все эндпоинты находятся под корневым путём `/ocr-classifier/api`.

### Health Check

Проверка работоспособности сервиса.

```
GET /ocr-classifier/api/health
```

**Ответ:**
```json
{"status": "ok"}
```

### Classify (v1)

Классификация изображения на наличие текста. Поддерживаются форматы `image/jpeg` и `image/png`.

```
POST /ocr-classifier/api/v1/classify
Content-Type: image/jpeg
Body: <бинарные данные изображения>
```

**Query параметры:**

- `lang` — языки для Tesseract OCR (например, `eng`, `rus`, `eng+rus`). По умолчанию: `eng+rus`
- `level` — уровень детализации структуры текста (PageIteratorLevel): `RIL_BLOCK`, `RIL_PARA`, `RIL_TEXTLINE`, `RIL_WORD`, `RIL_SYMBOL`. По умолчанию: `RIL_WORD`
- `confidence_threshold` — минимальный порог уверенности (0-1). По умолчанию: 0.66
- `min_token_count` — минимальное количество токенов. По умолчанию: 20

**Успешный ответ (200):**
```json
{
  "mean_confidence": 0.85,
  "weighted_confidence": 0.88,
  "token_count": 12,
  "boxes": [
    {
      "x": 10,
      "y": 20,
      "width": 100,
      "height": 50,
      "word": "Example",
      "confidence": 0.95
    }
  ],
  "angle": 0,
  "scale_factor": 1.0,
  "is_text_document": false,
  "bounding_box_width": 800,
  "bounding_box_height": 600
}
```

**Ошибки (4xx/5xx):**
```json
{"error": "сообщение об ошибке"}
```

- `405` - неверный HTTP метод (только POST)
- `400` - неверный Content-Type, пустое изображение или ошибка чтения данных
- `500` - ошибка обработки изображения или Tesseract OCR

## Тестирование с помощью curl

### Health Check

```bash
curl http://localhost:8080/ocr-classifier/api/health
```

### Классификация изображения

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @path/to/image.jpg \
  http://localhost:8080/ocr-classifier/api/v1/classify
```

**Пример с параметрами:**

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @test/dataset/eng/lightbulb-scheme.jpg \
  "http://localhost:8080/ocr-classifier/api/v1/classify?lang=eng&level=RIL_WORD&confidence_threshold=0.7&min_token_count=10"
```

**Пример c изображением из датасета:**

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @test/dataset/eng/lightbulb-scheme.jpg \
  http://localhost:8080/ocr-classifier/api/v1/classify
```
