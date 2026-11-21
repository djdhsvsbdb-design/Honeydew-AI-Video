# Honeydew AI Video

Отдельный экспериментальный веб-проект. Не является частью приложения Honeydew.

## Текущий этап

Стартовый интерфейс музыкального эскиза с процедурным звуковым генератором, canvas-визуализацией и записью результата. Подключена обученная NoteLM-модель (`models/honeydew_note_lm.f32`). Браузер загружает 1 038 848 байт весов и использует LSTM для предсказания MIDI-высот. Если модель не загрузится, включается процедурный резервный генератор.

## Запуск

Откройте `index.html` в современном браузере или включите GitHub Pages в настройках репозитория (ветка `main`, папка `/ (root)`).

## Экспорт

Экспорт использует MediaRecorder и запись canvas + audio stream. В зависимости от браузера будет создан MP4 или WebM: браузеры не гарантируют поддержку MP4-кодека для MediaRecorder.

## План

1. Проверить работу NoteLM в браузерах Android и desktop.
2. Расширить музыкальную генерацию: ритм, длительности, аккорды, бас и структура.
3. Улучшить визуальные темы и управление композицией.
4. Добавить надёжный MP4-рендеринг с совместимым кодеком.


## NoteLM model

The browser runtime in `model.js` reads `models/honeydew_note_lm.f32` as 259,712 little-endian float32 values. The model is the same pitch-only network used by the separate Honeydew Rust project: Embedding(128, 96) → LSTM(96, 192) → Linear(192, 128). It predicts MIDI pitches only; timing, note lengths, harmony, bass, drums, instruments and audio synthesis are not predicted by this checkpoint. If the model cannot be fetched or validated, the app falls back to its procedural sketch generator.
