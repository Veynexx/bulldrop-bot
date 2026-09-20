import json
import os
import math
from collections import defaultdict, Counter, deque
from datetime import datetime


CHECKPOINT_FILE = "predictor_state.json"


# =====================================================================
# ЯДРО: МАРКОВ С BACKOFF
# =====================================================================

class MarkovBackoff:
    """
    Цепь Маркова с правильным backoff (Katz).
    Пробует длинные паттерны, плавно откатывается к коротким.
    """
    def __init__(self, max_order=6):
        self.max_order = max_order
        self.tables = [defaultdict(Counter) for _ in range(max_order + 1)]
        # tables[k] — переходы порядка k

    def learn(self, history):
        self.tables = [defaultdict(Counter) for _ in range(self.max_order + 1)]
        for k in range(1, self.max_order + 1):
            for i in range(k, len(history)):
                prev = tuple(history[i - k:i])
                self.tables[k][prev][history[i]] += 1

    def learn_one(self, history, new_color):
        """history — уже без new_color."""
        for k in range(1, self.max_order + 1):
            if len(history) >= k:
                prev = tuple(history[-k:])
                self.tables[k][prev][new_color] += 1

    def predict(self, history):
        """
        Katz-backoff: начинаем с max_order, если pattern редко встречается —
        откатываемся. Confidence зависит от того, на каком порядке сработало.
        """
        if len(history) < 1:
            return None, 0.0

        for k in range(self.max_order, 0, -1):
            if len(history) < k:
                continue
            prev = tuple(history[-k:])
            counter = self.tables[k].get(prev)
            if not counter:
                continue
            total = sum(counter.values())
            # Требуем минимум наблюдений для доверия
            min_required = max(2, k)  # для длинных паттернов нужен больший порог
            if total < min_required:
                continue
            color, count = counter.most_common(1)[0]
            p = count / total
            # Бонус за длину паттерна
            length_bonus = 1.0 + (k - 1) * 0.1
            conf = min(0.9, p * length_bonus)
            return color, conf
        return None, 0.0

    def to_dict(self):
        return {
            "max_order": self.max_order,
            "tables": [
                {"|".join(k): dict(v) for k, v in table.items()}
                for table in self.tables
            ],
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(max_order=data.get("max_order", 6))
        obj.tables = []
        for table_data in data.get("tables", []):
            table = defaultdict(Counter)
            for k_str, counts in table_data.items():
                key = tuple(k_str.split("|")) if k_str else ()
                table[key] = Counter(counts)
            obj.tables.append(table)
        # Дополняем, если структура не совпала
        while len(obj.tables) < obj.max_order + 1:
            obj.tables.append(defaultdict(Counter))
        return obj


# =====================================================================
# КЛАССИЧЕСКИЕ МОДЕЛИ
# =====================================================================

class FrequencyPredictor:
    def __init__(self):
        self.counter = Counter()
        self.total = 0

    def learn(self, history):
        self.counter = Counter(history)
        self.total = len(history)

    def learn_one(self, color, weight=1.0):
        self.counter[color] += weight
        self.total += weight

    def predict(self, history):
        if not self.counter:
            return None, 0.0
        total = sum(self.counter.values())
        color, count = self.counter.most_common(1)[0]
        return color, (count / total) * 0.3

    def to_dict(self):
        return {"counter": dict(self.counter), "total": self.total}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.counter = Counter(data["counter"])
        obj.total = data.get("total", sum(obj.counter.values()))
        return obj


class AntiStreakPredictor:
    def __init__(self, streak_len=3):
        self.streak_len = streak_len

    def learn(self, history): pass
    def learn_one(self, *a, **k): pass

    def predict(self, history):
        if len(history) < self.streak_len:
            return None, 0.0
        last = history[-self.streak_len:]
        if len(set(last)) == 1:
            counter = Counter(history)
            for color, _ in counter.most_common():
                if color != last[0]:
                    return color, 0.6
        return None, 0.0

    def to_dict(self): return {"streak_len": self.streak_len}
    @classmethod
    def from_dict(cls, d): return cls(streak_len=d.get("streak_len", 3))


class GapPredictor:
    def __init__(self):
        self.gaps = defaultdict(list)
        self.last_seen = {}
        self.total_seen = 0

    def learn(self, history):
        self.gaps = defaultdict(list)
        self.last_seen = {}
        for i, c in enumerate(history):
            if c in self.last_seen:
                self.gaps[c].append(i - self.last_seen[c])
            self.last_seen[c] = i
        self.total_seen = len(history)

    def learn_one(self, color, index):
        if color in self.last_seen:
            self.gaps[color].append(index - self.last_seen[color])
        self.last_seen[color] = index
        self.total_seen = index + 1

    def predict(self, history):
        if not self.gaps:
            return None, 0.0
        n = len(history)
        best_color, best_ratio = None, 1.0
        for color, gl in self.gaps.items():
            if not gl:
                continue
            avg = sum(gl) / len(gl)
            if avg <= 0:
                continue
            ratio = (n - self.last_seen.get(color, 0)) / avg
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color
        if best_color:
            conf = min(0.8, 0.3 + (best_ratio - 1.0) * 0.4)
            return best_color, conf
        return None, 0.0

    def to_dict(self):
        return {
            "gaps": {k: list(v) for k, v in self.gaps.items()},
            "last_seen": dict(self.last_seen),
            "total_seen": self.total_seen,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.gaps = defaultdict(list, {k: list(v) for k, v in d.get("gaps", {}).items()})
        obj.last_seen = dict(d.get("last_seen", {}))
        obj.total_seen = d.get("total_seen", 0)
        return obj


class PeriodicityPredictor:
    def __init__(self, min_p=3, max_p=40, window=80):
        self.min_p = min_p
        self.max_p = max_p
        self.window = window
        self.best_period = None
        self.best_score = 0.0

    def learn(self, history): pass
    def learn_one(self, *a, **k): pass

    def predict(self, history):
        n = len(history)
        if n < self.max_p * 2:
            return None, 0.0

        window = min(self.window, n // 2)
        best_p, best_score = None, 0.0
        for p in range(self.min_p, self.max_p + 1):
            matches, total = 0, 0
            for i in range(n - window, n):
                if i - p >= 0:
                    total += 1
                    if history[i] == history[i - p]:
                        matches += 1
            if total < 15:
                continue
            score = matches / total
            if score > best_score:
                best_score = score
                best_p = p

        if best_p and best_score > 0.55:
            self.best_period = best_p
            self.best_score = best_score
            idx = n - best_p
            if 0 <= idx < n:
                return history[idx], min(0.75, best_score)
        self.best_period = None
        return None, 0.0

    def to_dict(self):
        return {"min_p": self.min_p, "max_p": self.max_p, "window": self.window}

    @classmethod
    def from_dict(cls, d):
        return cls(min_p=d.get("min_p", 3), max_p=d.get("max_p", 40), window=d.get("window", 80))


class TimePredictor:
    def __init__(self):
        self.hour_counter = defaultdict(Counter)

    def learn(self, history, timestamps=None):
        if not timestamps:
            return
        self.hour_counter = defaultdict(Counter)
        for c, ts in zip(history, timestamps):
            try:
                h = datetime.fromisoformat(ts).hour
                self.hour_counter[h][c] += 1
            except Exception:
                continue

    def learn_one(self, color, ts=None):
        if not ts:
            return
        try:
            h = datetime.fromisoformat(ts).hour
            self.hour_counter[h][color] += 1
        except Exception:
            pass

    def predict(self, history, current_ts=None):
        if not current_ts:
            return None, 0.0
        try:
            h = datetime.fromisoformat(current_ts).hour
        except Exception:
            return None, 0.0
        counter = self.hour_counter.get(h)
        if not counter:
            return None, 0.0
        total = sum(counter.values())
        if total < 15:
            return None, 0.0
        color, count = counter.most_common(1)[0]
        return color, min(0.7, (count / total) * min(1.0, total / 60.0))

    def to_dict(self):
        return {"hour_counter": {str(h): dict(c) for h, c in self.hour_counter.items()}}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for h_str, c in d.get("hour_counter", {}).items():
            obj.hour_counter[int(h_str)] = Counter(c)
        return obj


class BayesianConditionalPredictor:
    """Байесовская модель с prior, bigram и trigram."""
    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.total = 0
        self.colors = set()

    def learn(self, history):
        self.unigram = Counter(history)
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.total = len(history)
        self.colors = set(history)
        for i, c in enumerate(history):
            if i >= 1:
                self.bigram[history[i - 1]][c] += 1
            if i >= 2:
                self.trigram[(history[i - 2], history[i - 1])][c] += 1

    def learn_one(self, prev2, prev1, color):
        self.unigram[color] += 1
        if prev1 is not None:
            self.bigram[prev1][color] += 1
        if prev2 is not None and prev1 is not None:
            self.trigram[(prev2, prev1)][color] += 1
        self.total += 1
        self.colors.add(color)

    def predict(self, history):
        if not self.colors or len(history) < 2:
            return None, 0.0

        scores = {}
        for c in self.colors:
            scores[c] = (self.unigram[c] + 1) / (self.total + len(self.colors))

        prev1 = history[-1]
        b_total = sum(self.bigram[prev1].values()) + len(self.colors)
        for c in self.colors:
            p = (self.bigram[prev1].get(c, 0) + 1) / b_total
            scores[c] *= (p ** 0.5)

        if len(history) >= 2:
            prev2 = history[-2]
            tri = self.trigram.get((prev2, prev1))
            if tri:
                t_total = sum(tri.values()) + len(self.colors)
                for c in self.colors:
                    p = (tri.get(c, 0) + 1) / t_total
                    scores[c] *= (p ** 0.7)

        if not scores:
            return None, 0.0
        best = max(scores, key=scores.get)
        total_s = sum(scores.values())
        if total_s <= 0:
            return None, 0.0
        return best, min(0.85, scores[best] / total_s * 2.5)

    def to_dict(self):
        return {
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {f"{a}|{b}": dict(v) for (a, b), v in self.trigram.items()},
            "total": self.total,
            "colors": list(self.colors),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.unigram = Counter(d.get("unigram", {}))
        obj.bigram = defaultdict(Counter)
        for k, v in d.get("bigram", {}).items():
            obj.bigram[k] = Counter(v)
        obj.trigram = defaultdict(Counter)
        for k, v in d.get("trigram", {}).items():
            a, b = k.split("|")
            obj.trigram[(a, b)] = Counter(v)
        obj.total = d.get("total", 0)
        obj.colors = set(d.get("colors", []))
        return obj


# =====================================================================
# НОВЫЕ МОЩНЫЕ МОДЕЛИ
# =====================================================================

class PairStickinessPredictor:
    """
    Матрица «притяжения» пар цветов.
    Если после A часто идёт B — stickiness[A][B] высокое.
    """
    def __init__(self):
        self.matrix = defaultdict(Counter)  # prev -> Counter(next)
        self.totals = Counter()             # prev -> total

    def learn(self, history):
        self.matrix = defaultdict(Counter)
        self.totals = Counter()
        for i in range(len(history) - 1):
            self.matrix[history[i]][history[i + 1]] += 1
            self.totals[history[i]] += 1

    def learn_one(self, prev, next_color):
        if prev is not None:
            self.matrix[prev][next_color] += 1
            self.totals[prev] += 1

    def predict(self, history):
        if not history:
            return None, 0.0
        last = history[-1]
        counter = self.matrix.get(last)
        if not counter or self.totals[last] < 5:
            return None, 0.0
        total = self.totals[last]
        color, count = counter.most_common(1)[0]
        p = count / total
        # Чем более «липкая» пара, тем увереннее
        conf = min(0.8, p * min(1.0, total / 30.0))
        return color, conf

    def to_dict(self):
        return {
            "matrix": {k: dict(v) for k, v in self.matrix.items()},
            "totals": dict(self.totals),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.matrix = defaultdict(Counter)
        for k, v in d.get("matrix", {}).items():
            obj.matrix[k] = Counter(v)
        obj.totals = Counter(d.get("totals", {}))
        return obj


class RegimeDetector:
    """
    Определяет текущий режим колеса:
    - streak: серия одинаковых
    - balanced: разнообразие цветов
    - biased: явное доминирование одного цвета
    """
    def __init__(self, window=20):
        self.window = window

    def learn(self, history): pass
    def learn_one(self, *a, **k): pass

    def detect(self, history):
        if len(history) < self.window:
            return "unknown"
        recent = history[-self.window:]
        counter = Counter(recent)
        # Streak — если последние 3+ одинаковые
        if len(history) >= 3 and len(set(history[-3:])) == 1:
            return "streak"
        # Biased — если один цвет занимает >55%
        top_count = counter.most_common(1)[0][1]
        if top_count / self.window > 0.55:
            return "biased"
        # Balanced — все цвета примерно поровну
        if len(counter) >= 3 and top_count / self.window < 0.45:
            return "balanced"
        return "mixed"

    def to_dict(self): return {"window": self.window}
    @classmethod
    def from_dict(cls, d): return cls(window=d.get("window", 20))


class AntiPatternDetector:
    """
    Ловит ситуации, где все модели регулярно ошибаются.
    Запоминает context (последние 3 цвета) -> {predicted -> Counter(actual)}
    Если в этом контексте была предсказана X раз, а выпало Y раз —
    модель рекомендует ставить на Y, а не на X.
    """
    def __init__(self):
        self.context_errors = defaultdict(lambda: defaultdict(Counter))
        # context -> predicted -> Counter(actual)

    def learn(self, context, predicted, actual):
        if context is None or predicted is None:
            return
        self.context_errors[context][predicted][actual] += 1

    def predict(self, history):
        if len(history) < 3:
            return None, 0.0
        context = tuple(history[-3:])
        errs = self.context_errors.get(context)
        if not errs:
            return None, 0.0
        # Ищем предсказание, где модель стабильно ошибается
        best_anti_color, best_score = None, 0.0
        for pred_color, counter in errs.items():
            total = sum(counter.values())
            if total < 3:
                continue
            # Что РЕАЛЬНО выпадало, когда предсказывали pred_color?
            actual, count = counter.most_common(1)[0]
            if actual != pred_color and count / total > 0.5:
                score = count / total
                if score > best_score:
                    best_score = score
                    best_anti_color = actual
        if best_anti_color:
            return best_anti_color, min(0.7, best_score)
        return None, 0.0

    def to_dict(self):
        return {
            "context_errors": {
                "|".join(k): {p: dict(c) for p, c in v.items()}
                for k, v in self.context_errors.items()
            }
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.context_errors = defaultdict(lambda: defaultdict(Counter))
        for k_str, preds in d.get("context_errors", {}).items():
            context = tuple(k_str.split("|"))
            for p, c in preds.items():
                obj.context_errors[context][p] = Counter(c)
        return obj


# =====================================================================
# СТАТИСТИКА С RECENCY WEIGHT
# =====================================================================

class ModelStats:
    """
    Per-model статистика с recency weighting.
    Старые предсказания забываются — свежие весят больше.
    """
    def __init__(self, decay=0.995):
        self.decay = decay
        self.total = 0.0            # weighted
        self.hits = 0.0             # weighted
        self.confusion = defaultdict(Counter)  # для lift
        self.pred_counts = Counter()           # простые счётчики
        # Храним последние 30 предсказаний для оценки свежей точности
        self.recent = deque(maxlen=30)  # (predicted, actual)

    def update(self, predicted, actual):
        if predicted is None:
            return
        self.total = self.total * self.decay + 1.0
        self.pred_counts[predicted] += 1
        correct = (predicted == actual)
        if correct:
            self.hits = self.hits * self.decay + 1.0
        else:
            self.hits = self.hits * self.decay
        self.confusion[predicted][actual] += 1
        self.recent.append((predicted, actual))

    def accuracy_weighted(self):
        if self.total < 1:
            return 0.0
        return self.hits / self.total

    def accuracy_recent(self):
        if not self.recent:
            return 0.0
        hits = sum(1 for p, a in self.recent if p == a)
        return hits / len(self.recent)

    def per_color_lift(self, base_rates, min_count=5):
        result = {}
        for pred_color, counter in self.confusion.items():
            total = sum(counter.values())
            if total < min_count:
                continue
            p_correct = counter.get(pred_color, 0) / total
            base = base_rates.get(pred_color, 0.0)
            if base > 0:
                result[pred_color] = p_correct / base
        return result

    def to_dict(self):
        return {
            "decay": self.decay,
            "total": self.total,
            "hits": self.hits,
            "confusion": {p: dict(c) for p, c in self.confusion.items()},
            "pred_counts": dict(self.pred_counts),
            "recent": list(self.recent),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(decay=d.get("decay", 0.995))
        obj.total = d.get("total", 0.0)
        obj.hits = d.get("hits", 0.0)
        obj.confusion = defaultdict(Counter)
        for p, c in d.get("confusion", {}).items():
            obj.confusion[p] = Counter(c)
        obj.pred_counts = Counter(d.get("pred_counts", {}))
        obj.recent = deque(d.get("recent", []), maxlen=30)
        return obj


# =====================================================================
# ГЛАВНЫЙ АНСАМБЛЬ С МЕТА-ОБУЧЕНИЕМ
# =====================================================================

class EnsemblePredictor:
    def __init__(self):
        self.models = {
            "markov_backoff": MarkovBackoff(max_order=6),
            "frequency": FrequencyPredictor(),
            "anti_streak": AntiStreakPredictor(streak_len=3),
            "gap": GapPredictor(),
            "periodicity": PeriodicityPredictor(),
            "time": TimePredictor(),
            "bayesian": BayesianConditionalPredictor(),
            "pair_stickiness": PairStickinessPredictor(),
            "anti_pattern": AntiPatternDetector(),
        }

        self.regime = RegimeDetector(window=20)
        self.history = []
        self.timestamps = []
        self.last_processed_id = 0
        self.model_stats = {name: ModelStats() for name in self.models}

        # Мета-модель: какой режим -> какие модели работают хорошо
        self.regime_weights = defaultdict(lambda: defaultdict(float))
        # regime -> {model_name: weight}

        # Recent ensemble predictions
        self.recent_predictions = deque(maxlen=7)

        # Счётчик для exploration
        self.predictions_since_last_exploration = 0

    # ---------------------------------------------------------------
    # Обучение
    # ---------------------------------------------------------------
    def learn_full(self, history, timestamps=None, last_id=0):
        self.history = list(history)
        self.timestamps = list(timestamps) if timestamps else []
        for model in self.models.values():
            if isinstance(model, TimePredictor):
                model.learn(self.history, self.timestamps)
            else:
                try:
                    model.learn(self.history)
                except TypeError:
                    model.learn(self.history)
        self.last_processed_id = last_id

    def learn_new_entries(self, new_entries):
        for entry in new_entries:
            entry_id, color, ts = entry
            idx = len(self.history)

            for name, model in self.models.items():
                try:
                    if isinstance(model, MarkovBackoff):
                        model.learn_one(self.history, color)
                    elif isinstance(model, FrequencyPredictor):
                        model.learn_one(color)
                    elif isinstance(model, GapPredictor):
                        model.learn_one(color, idx)
                    elif isinstance(model, TimePredictor):
                        model.learn_one(color, ts)
                    elif isinstance(model, BayesianConditionalPredictor):
                        p1 = self.history[-1] if len(self.history) >= 1 else None
                        p2 = self.history[-2] if len(self.history) >= 2 else None
                        model.learn_one(p2, p1, color)
                    elif isinstance(model, PairStickinessPredictor):
                        p = self.history[-1] if self.history else None
                        model.learn_one(p, color)
                    # AntiPatternDetector учится отдельно через update_stats
                except Exception:
                    pass

            self.history.append(color)
            self.timestamps.append(ts)
            self.last_processed_id = entry_id

    # ---------------------------------------------------------------
    # Обновление мета-знаний
    # ---------------------------------------------------------------
    def update_stats(self, individual_predictions, actual):
        # Обновляем каждую модель
        for name, predicted in individual_predictions.items():
            if name in self.model_stats:
                self.model_stats[name].update(predicted, actual)

        # Anti-pattern: запоминаем ошибки в контексте
        if len(self.history) >= 3:
            context = tuple(self.history[-3:])
            for name, predicted in individual_predictions.items():
                if predicted is not None:
                    self.models["anti_pattern"].learn(context, predicted, actual)

        # Мета-обучение: обновляем веса моделей по режиму
        current_regime = self.regime.detect(self.history)
        for name, predicted in individual_predictions.items():
            if predicted is None:
                continue
            # Плавно обновляем вес: если угадал — увеличиваем, ошибся — уменьшаем
            old = self.regime_weights[current_regime][name]
            if old == 0:
                old = 0.5
            if predicted == actual:
                new = old * 0.95 + 0.05 * 1.0  # тянем к 1.0
            else:
                new = old * 0.95 + 0.05 * 0.0  # тянем к 0.0
            self.regime_weights[current_regime][name] = new

    # ---------------------------------------------------------------
    # Основное предсказание
    # ---------------------------------------------------------------
    def _base_rates(self):
        if not self.history:
            return {}
        counter = Counter(self.history)
        total = len(self.history)
        return {c: cnt / total for c, cnt in counter.items()}

    def _model_weight(self, name, color, base_rates):
        """
        Комбинированный вес:
        - 50% от per-color lift (weighted)
        - 30% от recent accuracy
        - 20% от regime weight
        """
        s = self.model_stats.get(name)
        if s is None:
            return 0.3

        # 1) Lift
        lifts = s.per_color_lift(base_rates)
        color_lift = lifts.get(color, 1.0)
        # Сглаживаем lift к 1.0 — если данных мало
        lift_component = 0.5 + 0.5 * min(2.0, max(0.0, color_lift - 0.5))

        # 2) Recent accuracy
        recent_acc = s.accuracy_recent()
        recent_component = 0.3 + 0.7 * recent_acc

        # 3) Regime weight
        current_regime = self.regime.detect(self.history)
        regime_w = self.regime_weights[current_regime].get(name, 0.5)
        regime_component = 0.3 + 0.7 * regime_w

        # Взвешенная комбинация
        weight = (
            0.5 * lift_component +
            0.3 * recent_component +
            0.2 * regime_component
        )

        # Штраф за «привычку» ставить один цвет
        total_p = sum(s.pred_counts.values())
        if total_p > 20:
            freq = s.pred_counts[color] / total_p
            if freq > 0.6:
                weight *= max(0.3, 1.0 - (freq - 0.6) * 1.5)

        return max(0.05, min(2.0, weight))

    def predict(self, current_ts=None):
        base_rates = self._base_rates()
        votes = defaultdict(float)
        sources = defaultdict(list)

        for name, model in self.models.items():
            try:
                if isinstance(model, TimePredictor):
                    color, conf = model.predict(self.history, current_ts)
                else:
                    color, conf = model.predict(self.history)
            except Exception:
                color, conf = None, 0.0

            if not color:
                continue

            weight = self._model_weight(name, color, base_rates)

            # Rarity boost для редких цветов
            base = base_rates.get(color, 0.0)
            rarity = 1.0
            if base < 0.15:
                rarity = 1.4
            elif base < 0.25:
                rarity = 1.2

            votes[color] += conf * weight * rarity
            sources[color].append(f"{name}({conf:.2f}×{weight:.2f})")

        if not votes:
            return None, None, 0.0

        # Anti-habit: если 5+ раз подряд ставили одно и то же — разнообразим
        if (len(self.recent_predictions) >= 5 and
                len(set(self.recent_predictions)) == 1):
            blocked = self.recent_predictions[0]
            for c in votes:
                if c != blocked:
                    votes[c] *= 1.6

        # Exploration: раз в 15 предсказаний даём шанс «непопулярному» цвету
        self.predictions_since_last_exploration += 1
        if self.predictions_since_last_exploration >= 15 and len(votes) > 1:
            # Берём второй по популярности цвет
            sorted_colors = sorted(votes, key=votes.get, reverse=True)
            if len(sorted_colors) > 1:
                second = sorted_colors[1]
                votes[second] *= 1.4
                self.predictions_since_last_exploration = 0

        best = max(votes, key=votes.get)
        self.recent_predictions.append(best)
        return best, ", ".join(sources[best]), votes[best]

    def get_individual_predictions(self):
        result = {}
        for name, model in self.models.items():
            try:
                if isinstance(model, TimePredictor):
                    ts = self.timestamps[-1] if self.timestamps else None
                    color, conf = model.predict(self.history, ts)
                else:
                    color, conf = model.predict(self.history)
                result[name] = color
            except Exception:
                result[name] = None
        return result

    # ---------------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------------
    def save(self, path=CHECKPOINT_FILE):
        data = {
            "history": self.history,
            "timestamps": self.timestamps,
            "last_processed_id": self.last_processed_id,
            "model_stats": {n: s.to_dict() for n, s in self.model_stats.items()},
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "recent_predictions": list(self.recent_predictions),
            "regime_weights": {
                r: dict(w) for r, w in self.regime_weights.items()
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path=CHECKPOINT_FILE):
        obj = cls()
        if not os.path.exists(path):
            return obj
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj.history = data.get("history", [])
            obj.timestamps = data.get("timestamps", [])
            obj.last_processed_id = data.get("last_processed_id", 0)
            obj.recent_predictions = deque(data.get("recent_predictions", []), maxlen=7)

            # Regime weights
            rw = data.get("regime_weights", {})
            for regime, weights in rw.items():
                for name, w in weights.items():
                    obj.regime_weights[regime][name] = w

            # Model stats
            stats_data = data.get("model_stats", {})
            for name in obj.model_stats:
                if name in stats_data:
                    obj.model_stats[name] = ModelStats.from_dict(stats_data[name])

            # Модели
            models_data = data.get("models", {})
            for name, model in list(obj.models.items()):
                if name in models_data:
                    try:
                        obj.models[name] = type(model).from_dict(models_data[name])
                    except Exception as e:
                        print(f"[ИИ] Не загрузил модель {name}: {e}")
            return obj
        except Exception as e:
            print(f"[ИИ] Не загрузил checkpoint: {e}")
            return obj


# =====================================================================
# СТАТИСТИКА ПРЕДСКАЗАНИЙ
# =====================================================================

class PredictorStats:
    def __init__(self, file="predictions.jsonl"):
        self.file = file
        self.score = 0
        self.total = 0
        self.correct = 0
        self.wrong = 0
        self.load_existing()

    def load_existing(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.total += 1
                        if rec.get("correct"):
                            self.correct += 1
                            self.score += 1
                        else:
                            self.wrong += 1
                            self.score -= 1
                    except Exception:
                        continue
        except Exception:
            pass

    def record(self, predicted, actual, source, confidence):
        correct = (predicted == actual)
        if correct:
            self.score += 1
            self.correct += 1
        else:
            self.score -= 1
            self.wrong += 1
        self.total += 1
        rec = {
            "predicted": predicted,
            "actual": actual,
            "correct": correct,
            "source": source,
            "confidence": round(confidence, 3),
            "score": self.score,
        }
        try:
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return correct

    def accuracy(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total * 100


def read_all_entries(json_file):
    entries = []
    if not os.path.exists(json_file):
        return entries
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rid = rec.get("id")
                color = rec.get("color")
                ts = rec.get("timestamp")
                if rid is not None and color:
                    entries.append((rid, color, ts))
            except Exception:
                continue
    entries.sort(key=lambda x: x[0])
    return entries