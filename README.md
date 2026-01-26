# OCR Classifier

HTTP-сервис для определения наличия текста на изображениях с использованием Tesseract OCR. Возвращает показатель уверенности (confidence) от 0 до 1, указывающий на вероятность присутствия текста на изображении.

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

### Health Check

Проверка работоспособности сервиса.

```
GET /health
```

**Ответ:**
```json
{"status": "ok"}
```

### Classify

Классификация изображения на наличие текста.

```
POST /classify
Content-Type: image/jpeg
Body: <бинарные данные JPEG изображения>
```

**Успешный ответ (200):**
```json
{"confidence": 0.85}
```

**Ошибки:**
- `405` - неверный HTTP метод
- `400` - неверный Content-Type или пустое изображение
- `500` - ошибка обработки изображения

## Тестирование с помощью curl

### Health Check

```bash
curl http://localhost:8080/health
```

### Классификация изображения

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @path/to/image.jpg \
  http://localhost:8080/classify
```

**Пример c изображением из датасета:**

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary @test/dataset/text-deutch.jpg \
  http://localhost:8080/classify
```
