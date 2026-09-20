import time
import json
import re
import os
from datetime import datetime
from DrissionPage import ChromiumPage, ChromiumOptions

from predictor import (
    EnsemblePredictor,
    PredictorStats,
    read_all_entries,
    CHECKPOINT_FILE,
)

# ================== НАСТРОЙКИ ==================
URL = "https://bulldrop.mobi/ru/games/wheel"
JSON_FILE = "wheel_history.jsonl"
LOG_FILE = "wheel_history.log"
PRED_FILE = "predictions.jsonl"

BROWSER_PATH = os.environ.get("BROWSER_PATH", "/usr/bin/chromium")

JS_GET_CLASSES = '''
    var items = document.querySelectorAll('div[class*="ui-wheel-color-box--color-"]');
    var result = [];
    for (var i = 0; i < items.length; i++) {
        result.push(items[i].className);
    }
    return result;
'''

# ================== ФУНКЦИИ ==================
def get_color_name(class_attr):
    if not class_attr:
        return None
    match = re.search(r'--color-([a-z_]+?)_\d', class_attr)
    if not match:
        return "unknown"
    raw_color = match.group(1).lower()
    if raw_color == "red":
        return "red"
    elif raw_color == "green":
        return "green"
    elif raw_color == "blue":
        return "blue"
    elif raw_color == "lucky_glow":
        return "50x (rainbow)"
    else:
        return f"50x ({raw_color})"

def log_action(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")
    except Exception:
        pass

def log_line(message=""):
    """Только в консоль и лог, без префикса времени — для разделителей."""
    print(message)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass

def get_last_id(json_file):
    if not os.path.exists(json_file):
        return 0
    last_id = 0
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rid = rec.get("id", 0)
                    if isinstance(rid, int) and rid > last_id:
                        last_id = rid
                except Exception:
                    continue
    except Exception:
        pass
    return last_id

# ================== ОСНОВНАЯ ФУНКЦИЯ ==================
def run():
    log_action("=" * 60)
    log_action("Запуск парсера колеса Bulldrop + усиленный ИИ")
    log_action("=" * 60)

    start_id = get_last_id(JSON_FILE)
    log_action(f"Продолжаем с id = {start_id + 1}" if start_id else "Начинаем с id = 1")

    # ---- Загрузка чекпоинта ----
    predictor = EnsemblePredictor.load(CHECKPOINT_FILE)
    stats = PredictorStats(file=PRED_FILE)

    log_action(
        f"[ИИ] Чекпоинт загружен: last_id={predictor.last_processed_id}, "
        f"история={len(predictor.history)}"
    )

    all_entries = read_all_entries(JSON_FILE)
    new_entries = [e for e in all_entries if e[0] > predictor.last_processed_id]

    if new_entries:
        log_action(f"[ИИ] Дообучение на {len(new_entries)} новых записях")
        predictor.learn_new_entries(new_entries)
        predictor.save(CHECKPOINT_FILE)

    log_action("[ИИ] Текущая точность моделей:")
    for name, s in predictor.model_stats.items():
        if s.total > 0:
            acc = s.hits / s.total * 100
            log_action(f"    {name}: {s.hits}/{s.total} ({acc:.1f}%)")

    log_action(
        f"[ИИ] Общий счёт: {stats.score} | "
        f"угадано {stats.correct}/{stats.total} | "
        f"точность {stats.accuracy():.1f}%"
    )
    log_action("-" * 60)

    # ---- Браузер ----
    co = ChromiumOptions()
    co.set_browser_path(BROWSER_PATH)
    co.set_load_mode('eager')
    co.set_timeouts(page_load=30)

    log_action("Запускаем Edge...")
    try:
        page = ChromiumPage(co)
    except Exception as e:
        log_action(f"ОШИБКА запуска браузера: {e}")
        input("Нажми Enter для выхода...")
        return

    log_action(f"Открываем {URL}...")
    try:
        page.get(URL, timeout=30, retry=1)
    except Exception as e:
        log_action(f"ОШИБКА загрузки: {e}")
        input("Нажми Enter для выхода...")
        return

    log_action("Ждём 5 секунд...")
    time.sleep(5)

    try:
        classes = page.run_js(JS_GET_CLASSES)
    except Exception as e:
        log_action(f"ОШИБКА JS: {e}")
        input("Нажми Enter для выхода...")
        return

    if not classes:
        log_action("ОШИБКА: элементы не найдены.")
        input("Нажми Enter для выхода...")
        return

    log_action(f"Найдено элементов истории: {len(classes)}")

    last_snapshot = list(classes)
    total_recorded = start_id
    session_recorded = 0
    not_found_count = 0
    last_status_time = time.time()

    pending_prediction = None
    pending_source = None
    pending_conf = 0.0
    pending_individual = {}

    def make_prediction():
        current_ts = datetime.now().isoformat()
        pred_color, source, conf = predictor.predict(current_ts=current_ts)
        individual = predictor.get_individual_predictions()
        return pred_color, source, conf, individual

    def print_prediction_block(pred_color, source, conf):
        log_line("")
        log_line("╔" + "═" * 58 + "╗")
        log_line(f"║  ПРЕДСКАЗАНИЕ НА СЛЕДУЮЩИЙ РАУНД: {pred_color.upper():<24}║")
        log_line(f"║  Уверенность: {conf:<43.3f}║")
        log_line("╠" + "═" * 58 + "╣")
        log_line(f"║  Голоса моделей:                                    ║")
        for vote in source.split(","):
            vote = vote.strip()
            if vote:
                log_line(f"║    {vote:<52}║")
        log_line("╚" + "═" * 58 + "╝")
        log_line("")

    # ---- Первое предсказание ----
    if predictor.history:
        p, s, c, ind = make_prediction()
        if p:
            pending_prediction = p
            pending_source = s
            pending_conf = c
            pending_individual = ind
            print_prediction_block(p, s, c)
        else:
            log_action("[ИИ] Пока недостаточно данных для предсказания")
    else:
        log_action("[ИИ] Нет истории — предсказания начнутся после накопления данных")

    log_action("=" * 60)
    log_action("Скрипт + ИИ запущены. Ctrl+C для остановки.")
    log_action("=" * 60)

    while True:
        try:
            classes = page.run_js(JS_GET_CLASSES)
            if not classes:
                not_found_count += 1
                if not_found_count % 30 == 0:
                    log_action(f"[ПАРСЕР] Элементы не найдены ({not_found_count} раз)")
                time.sleep(1)
                continue

            not_found_count = 0

            if classes != last_snapshot:
                # Определяем новые элементы
                try:
                    old_first = last_snapshot[0]
                    idx_in_new = classes.index(old_first)
                    new_count = idx_in_new
                except (ValueError, IndexError):
                    new_count = 1
                if new_count < 1:
                    new_count = 1

                new_records = []
                for i in range(new_count - 1, -1, -1):
                    color_i = get_color_name(classes[i])
                    if color_i:
                        total_recorded += 1
                        session_recorded += 1
                        ts = datetime.now().isoformat()
                        record = {
                            "id": total_recorded,
                            "timestamp": ts,
                            "color": color_i,
                        }
                        with open(JSON_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        new_records.append((total_recorded, color_i, ts))

                # ---- Обработка каждого нового раунда ----
                for rec_id, color_i, ts in new_records:
                    log_line("")
                    log_line("━" * 60)
                    log_line(f"  РАУНД ЗАВЕРШЁН")
                    log_line(f"  Выпал цвет:  {color_i.upper()}")
                    log_line(f"  ЗАПИСАНО #{rec_id} в {JSON_FILE}")
                    log_line("━" * 60)

                    # 1) Оценка предыдущего предсказания
                    if pending_prediction is not None:
                        correct = stats.record(
                            predicted=pending_prediction,
                            actual=color_i,
                            source=pending_source or "?",
                            confidence=pending_conf,
                        )
                        mark = "✅ УГАДАЛ" if correct else "❌ ПРОМАХ"
                        log_line(f"  ПРЕДЫДУЩЕЕ ПРЕДСКАЗАНИЕ: {pending_prediction.upper()} → {mark}")
                        log_line(f"  Счёт ИИ: {stats.score:+d} | "
                                 f"Угадано: {stats.correct}/{stats.total} "
                                 f"({stats.accuracy():.1f}%)")
                        predictor.update_stats(pending_individual, color_i)
                        pending_prediction = None
                        pending_individual = {}
                    else:
                        log_line("  (предыдущего предсказания не было)")

                    # 2) Дообучение на новом цвете
                    predictor.learn_new_entries([(rec_id, color_i, ts)])
                    predictor.save(CHECKPOINT_FILE)

                    # 3) Предсказание на СЛЕДУЮЩИЙ раунд
                    p, s, c, ind = make_prediction()
                    if p:
                        pending_prediction = p
                        pending_source = s
                        pending_conf = c
                        pending_individual = ind
                        print_prediction_block(p, s, c)
                    else:
                        pending_prediction = None
                        log_line("  [ИИ] Недостаточно данных для предсказания")
                        log_line("")

                last_snapshot = list(classes)

            # Статус раз в минуту
            now = time.time()
            if now - last_status_time >= 60:
                current_top = get_color_name(classes[0]) if classes else "?"
                log_action(
                    f"[СТАТУС] Сессия: {session_recorded} | Всего: {total_recorded} | "
                    f"Верх: {current_top.upper()} | "
                    f"[ИИ] счёт={stats.score} точность={stats.accuracy():.1f}%"
                )
                last_status_time = now

            time.sleep(1)

        except KeyboardInterrupt:
            log_action("Остановка по Ctrl+C")
            break
        except Exception as e:
            log_action(f"ОШИБКА: {e}")
            time.sleep(5)

    # Финальное сохранение
    try:
        predictor.save(CHECKPOINT_FILE)
        log_action(f"[ИИ] Чекпоинт сохранён в {CHECKPOINT_FILE}")
    except Exception as e:
        log_action(f"[ИИ] Ошибка сохранения: {e}")

    log_action("=" * 60)
    log_action(f"За сессию: {session_recorded} | Всего: {total_recorded}")
    log_action(f"[ИИ] Счёт: {stats.score} | Точность: {stats.accuracy():.1f}%")
    log_action("[ИИ] Точность по моделям:")
    for name, s in predictor.model_stats.items():
        if s.total > 0:
            acc = s.hits / s.total * 100
            log_action(f"    {name}: {s.hits}/{s.total} ({acc:.1f}%)")
    log_action("=" * 60)


if __name__ == "__main__":
    run()