import numpy as np
import pandas as pd
from scipy.linalg import inv, eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from datetime import datetime, timedelta
prior_kwargs = {}


def minnesota_prior(Y, p, lambda1=0.1, lambda2=0.5, lambda3=1.0, lambda4=0.01, exog=None):
    """Миннесотский априор с поддержкой экзогенных переменных."""
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # + экзогенные переменные

    prior_mean = np.zeros(n * k)
    prior_var = np.zeros((n * k, n * k))

    for i in range(n):
        # Априор для лагов эндогенных переменных
        prior_mean[i * k + i * p + 1] = 1
        for j in range(n):
            for l in range(1, p + 1):
                idx = i * k + j * p + l
                if i == j:
                    prior_var[idx, idx] = (lambda1 / l) ** 2
                else:
                    std_j = np.std(Y[:, j])
                    if std_j == 0:
                        std_j = 1e-6  # или другое небольшое значение
                    prior_var[idx, idx] = (lambda1 * lambda2 / (l * std_j)) ** 2
        prior_var[i * k, i * k] = (lambda1 * lambda3) ** 2

        # Априор для экзогенных переменных (слабая информация)
        if m > 0:
            for j in range(m):
                idx = i * k + n * p + 1 + j
                prior_var[idx, idx] = lambda4 ** 2  # Малая дисперсия

    return prior_mean, prior_var


def normal_flat_prior(Y, p, exog=None):
    """Normal-Flat априор с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (матрица T x n)
    - p: Порядок лага
    - exog: Экзогенные переменные (матрица T x m), опционально

    Возвращает:
    - prior_mean: Вектор средних (нули)
    - prior_var: Диагональная матрица ковариации (большие дисперсии)
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Константа + лаги эндогенных + экзогенные

    prior_mean = np.zeros(n * k)
    prior_var = np.eye(n * k) * 1e6  # Большая дисперсия ~ "плоский" априор

    return prior_mean, prior_var


def normal_wishart_prior(Y, p, v_0=None, S_0=None, M_0=None, Omega_0=None, exog=None):
    """
    Normal-Wishart априор для BVAR с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (матрица T x n)
    - p: Порядок лага
    - v_0: Степени свободы для Wishart (v_0 > n-1)
    - S_0: Масштабная матрица для Wishart (n x n)
    - M_0: Матрица средних для Normal (k x n)
    - Omega_0: Ковариационная матрица для Normal (k x k)
    - exog: Экзогенные переменные (матрица T x m), опционально

    Возвращает:
    - Словарь с параметрами априора {'v_0', 'S_0', 'M_0', 'Omega_0'}
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Константа + лаги эндогенных + экзогенные

    # Значения по умолчанию
    if v_0 is None:
        v_0 = n + 2
    if S_0 is None:
        S_0 = np.eye(n) * 0.1
    if M_0 is None:
        M_0 = np.zeros((k, n))
        for i in range(n):
            M_0[i * p + 1, i] = 1.0  # Собственные лаги ~ N(1, 1)
    if Omega_0 is None:
        Omega_0 = np.eye(k)
        # Уменьшаем априорную дисперсию для экзогенных переменных
        if m > 0:
            for i in range(n):
                for j in range(m):
                    idx = n * p + 1 + j
                    Omega_0[idx, idx] = 0.1  # Меньшая дисперсия для экзогенных

    return {'v_0': v_0, 'S_0': S_0, 'M_0': M_0, 'Omega_0': Omega_0}


def sims_zha_normal_flat_prior(Y, p, lambda1=0.1, lambda2=0.5, lambda3=1.0, mu5=0.1, mu6=1.0, exog=None, lambda4=0.01):
    """
    Априор Sims-Zha для структурной BVAR с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - lambda1: Собственные лаги
    - lambda2: Перекрестные лаги
    - lambda3: Константы
    - mu5: Коинтеграция
    - mu6: Временная устойчивость
    - exog: Экзогенные переменные (T x m), опционально
    - lambda4: Дисперсия для экзогенных переменных (по умолчанию 0.01)

    Возвращает:
    - prior_mean: Вектор априорных средних
    - prior_var: Диагональная матрица априорных дисперсий
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Константа + лаги + экзогенные

    # 1. Априорные средние
    prior_mean = np.zeros(n * k)
    for i in range(n):
        prior_mean[i * k + i * p + 1] = 1  # Собственные лаги ~ N(1, sigma2)

    # 2. Ковариационная структура Sims-Zha
    prior_var = np.eye(n * k)
    sigma = np.std(Y, axis=0)

    # Для эндогенных переменных
    for i in range(n):
        for j in range(n):
            for l in range(1, p + 1):
                idx = i * k + j * p + l
                if i == j:
                    prior_var[idx, idx] = (lambda1 / (l ** mu5 * sigma[i])) ** 2
                else:
                    prior_var[idx, idx] = (lambda1 * lambda2 / (l ** mu5 * sigma[j])) ** 2

        # Для константы
        prior_var[i * k, i * k] = (lambda1 * lambda3) ** 2

        # Для экзогенных переменных (если есть)
        if m > 0:
            for j in range(m):
                idx = i * k + n * p + 1 + j
                prior_var[idx, idx] = lambda4 ** 2  # Фиксированная малая дисперсия

    # 3. Компонент временной устойчивости (только для эндогенных)
    for i in range(n):
        for l in range(1, p + 1):
            idx = i * k + i * p + l
            prior_var[idx, idx] /= mu6 ** (2 * (l - 1))

    return prior_mean, prior_var


def sims_zha_normal_wishart_prior(Y, p, lambda1=0.1, lambda2=0.5, lambda3=1.0, mu5=0.1, mu6=1.0,
                                  v_0=None, S_0=None, exog=None, lambda4=0.01):
    """
    Sims-Zha Normal-Wishart prior для структурной BVAR с экзогенными переменными.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - lambda1, lambda2, lambda3: Параметры регуляризации
    - mu5: Параметр коинтеграции
    - mu6: Параметр временной устойчивости
    - v_0: Степени свободы для Wishart
    - S_0: Масштабная матрица для Wishart
    - exog: Экзогенные переменные (T x m)
    - lambda4: Дисперсия для экзогенных переменных

    Возвращает:
    - Словарь с параметрами априора
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Учитываем экзогенные переменные

    # 1. Априорные средние
    prior_mean = np.zeros(n * k)
    for i in range(n):
        prior_mean[i * k + i * p + 1] = 1  # Собственные лаги ~ N(1, sigma2)

    # 2. Ковариационная структура
    sigma = np.std(Y, axis=0)
    prior_var = np.eye(n * k)

    # Для эндогенных переменных
    for i in range(n):
        for j in range(n):
            for l in range(1, p + 1):
                idx = i * k + j * p + l
                if i == j:
                    prior_var[idx, idx] = (lambda1 / (l ** mu5 * sigma[i])) ** 2
                else:
                    prior_var[idx, idx] = (lambda1 * lambda2 / (l ** mu5 * sigma[j])) ** 2

        # Для константы
        prior_var[i * k, i * k] = (lambda1 * lambda3) ** 2

        # Для экзогенных переменных
        if m > 0:
            for j in range(m):
                idx = i * k + n * p + 1 + j
                prior_var[idx, idx] = lambda4 ** 2  # Фиксированная малая дисперсия

    # 3. Компонент временной устойчивости (только для эндогенных)
    for i in range(n):
        for l in range(1, p + 1):
            idx = i * k + i * p + l
            prior_var[idx, idx] /= mu6 ** (2 * (l - 1))

    # 4. Настройки Wishart
    if v_0 is None:
        v_0 = n + 2
    if S_0 is None:
        S_0 = np.eye(n) * 0.1

    return {
        'prior_mean': prior_mean,
        'prior_var': prior_var,
        'v_0': v_0,
        'S_0': S_0
    }


def bvar_estimate(Y, p, prior_type='minnesota', exog=None, **prior_kwargs):
    """Оценка BVAR с выбором априора и поддержкой экзогенных переменных."""
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Учитываем экзогенные переменные

    # Построение матрицы регрессоров
    X = np.ones((T - p, k))
    for t in range(p, T):
        X[t - p, 1:n*p+1] = Y[t - p:t, :].flatten()  # Лаги эндогенных
        if m > 0:
            X[t - p, n*p+1:] = exog[t, :]  # Текущие экзогенные переменные
    Y = Y[p:]

    if prior_type == 'minnesota':
        prior_mean, prior_var = minnesota_prior(Y, p, exog=exog, **prior_kwargs)
        X_full = np.zeros((n * (T - p), n * k))
        for i in range(n):
            X_full[i * (T - p):(i + 1) * (T - p), i * k:(i + 1) * k] = X
        Y_full = Y.flatten()

        posterior_var = inv(inv(prior_var) + X_full.T @ X_full)
        posterior_mean = posterior_var @ (inv(prior_var) @ prior_mean + X_full.T @ Y_full)

    elif prior_type == 'normal_flat':
        prior_mean, prior_var = normal_flat_prior(Y, p, exog=exog)
        X_full = np.zeros((n * (T - p), n * k))
        for i in range(n):
            X_full[i * (T - p):(i + 1) * (T - p), i * k:(i + 1) * k] = X
        Y_full = Y.flatten()

        posterior_var = inv(inv(prior_var) + X_full.T @ X_full)
        posterior_mean = posterior_var @ (inv(prior_var) @ prior_mean + X_full.T @ Y_full)

    elif prior_type == 'normal_wishart':
        prior = normal_wishart_prior(Y, p, exog=exog, **prior_kwargs)
        v_0, S_0, M_0, Omega_0 = prior['v_0'], prior['S_0'], prior['M_0'], prior['Omega_0']

        # Апостериорные параметры для Normal-Wishart
        v_T = v_0 + T - p
        Omega_T = inv(inv(Omega_0) + X.T @ X)
        M_T = Omega_T @ (inv(Omega_0) @ M_0 + X.T @ Y)
        S_T = S_0 + Y.T @ Y + M_0.T @ inv(Omega_0) @ M_0 - M_T.T @ inv(Omega_T) @ M_T

        posterior_mean = M_T.flatten()
        posterior_var = np.kron(S_T, Omega_T) / (v_T - n - 1)

    elif prior_type == 'sims_zha_normal_flat':
        prior_mean, prior_var = sims_zha_normal_flat_prior(Y, p, exog=exog, **prior_kwargs)

        # Структурная ковариационная матрица шока
        structural_shock = np.linalg.cholesky(np.cov(Y.T))
        scaling_matrix = np.kron(np.eye(k), structural_shock)
        prior_var = scaling_matrix @ prior_var @ scaling_matrix.T

        X_full = np.zeros((n * (T - p), n * k))
        for i in range(n):
            X_full[i * (T - p):(i + 1) * (T - p), i * k:(i + 1) * k] = X
        Y_full = Y.flatten()

        posterior_var = inv(inv(prior_var) + X_full.T @ X_full)
        posterior_mean = posterior_var @ (inv(prior_var) @ prior_mean + X_full.T @ Y_full)

    elif prior_type == 'sims_zha_normal_wishart':
        prior = sims_zha_normal_wishart_prior(Y, p, exog=exog, **prior_kwargs)
        prior_mean, prior_var = prior['prior_mean'], prior['prior_var']
        v_0, S_0 = prior['v_0'], prior['S_0']

        # Структурная ковариационная матрица
        structural_shock = np.linalg.cholesky(np.cov(Y.T))
        scaling_matrix = np.kron(structural_shock, np.eye(k))
        prior_var = scaling_matrix @ prior_var @ scaling_matrix.T

        # Подготовка данных
        X_full = np.zeros((n * (T - p), n * k))
        for i in range(n):
            X_full[i * (T - p):(i + 1) * (T - p), i * k:(i + 1) * k] = X
        Y_full = Y.flatten()

        # Апостериорные параметры
        v_T = v_0 + T - p
        Omega_T = inv(inv(prior_var) + X_full.T @ X_full)
        M_T = Omega_T @ (inv(prior_var) @ prior_mean + X_full.T @ Y_full)

        # Расчет S_T
        residuals = Y - X @ M_T.reshape(k, n)
        S_T = S_0 + residuals.T @ residuals

        posterior_mean = M_T.flatten()
        posterior_var = Omega_T

    else:
        raise ValueError(
            "Неизвестный тип априора. Доступно: 'minnesota', 'normal_flat', 'normal_wishart', 'sims_zha_normal_flat', 'sims_zha_normal_wishart'")

    return posterior_mean, posterior_var


def forecast_bvar(Y, post_mean, post_var, p, steps=10, n_samples=1000, exog_future=None, scaler_Y=None):
    """
    Генерация прогнозов с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - post_mean: Апостериорные средние коэффициентов
    - post_var: Апостериорная ковариационная матрица
    - p: Порядок лага
    - steps: Длина прогноза
    - n_samples: Количество сэмплов
    - exog_future: Будущие значения экзогенных переменных (steps x m)
    - scaler_Y: Объект StandardScaler для обратного преобразования (опционально)

    Возвращает:
    - forecasts: Медианные прогнозы (steps x n)
    - conf_intervals: Доверительные интервалы
    """
    T, n = Y.shape
    m = exog_future.shape[1] if exog_future is not None else 0
    k = n * p + 1 + m  # Учитываем экзогенные переменные

    samples = np.zeros((n_samples, steps, n))

    for i in range(n_samples):
        # Проверка и исправление post_var
        post_var = (post_var + post_var.T) / 2  # Делаем симметричной
        eigvals, eigvecs = eigh(post_var)
        eigvals[eigvals < 1e-8] = 1e-8  # Убираем отрицательные значения
        post_var = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Пробуем сэмплировать, если не получается — добавляем регуляризацию
        try:
            coeffs = np.random.multivariate_normal(post_mean, post_var)
        except np.linalg.LinAlgError:
            # В случае ошибки — добавить малую регуляризацию
            epsilon = 1e-6
            post_var += np.eye(post_var.shape[0]) * epsilon
            coeffs = np.random.multivariate_normal(post_mean, post_var)

        coeffs = coeffs.reshape(n, -1)
        last_obs = Y[-p:].flatten()

        for step in range(steps):
            X = np.concatenate([[1], last_obs])
            if m > 0:
                X = np.concatenate([X, exog_future[step, :]])
            pred = coeffs @ X
            samples[i, step] = pred
            last_obs = np.concatenate([pred, last_obs])[:-n]

    forecasts = np.median(samples, axis=0)
    conf_intervals = {
        'samples': samples,
        '5%': np.percentile(samples, 5, axis=0),
        '95%': np.percentile(samples, 95, axis=0)
    }

    return forecasts, conf_intervals


def plot_results(Y, forecasts, conf_intervals, p, endog_vars=None, exog_vars=None,
                exog_data=None, scaler_Y=None, scaler_exog=None, dates=None,
                Y_min=None, exog_min=None):
    """Визуализация результатов с разделением на эндогенные и экзогенные переменные."""
    if scaler_Y is None:
        raise ValueError("Не передан scaler_Y для обратного преобразования данных")

    # Обратное преобразование данных
    Y_original = inverse_transform_data(Y, scaler_Y, Y_min)
    forecasts_original = inverse_transform_data(forecasts, scaler_Y, Y_min)

    # Обратное преобразование доверительных интервалов для samples (n_samples, steps, n_endog)
    # Для этого надо "развернуть" массив и обратно "свернуть"
    n_samples, steps, n_endog = conf_intervals['samples'].shape
    samples_reshaped = conf_intervals['samples'].reshape(-1, n_endog)
    samples_original = inverse_transform_data(samples_reshaped, scaler_Y, Y_min)
    samples_original = samples_original.reshape(n_samples, steps, n_endog)

    # Названия переменных
    endog_names = {
        0: 'Uninvested Funds',
        1: 'Netto Funds',
        2: 'Reinvested Funds',
        3: 'Total Clients'
    }
    exog_names = {
        0: 'Invested Funds',
        1: 'Planned Rate'
    }

    # Подготовка дат
    if dates is not None:
        # Оставляем только те даты, которые соответствуют Y (эндогенным переменным)
        if isinstance(dates[0], pd.Period):
            dates_main = [d.to_timestamp() for d in dates]
        else:
            dates_main = list(dates)
        # Используем только последние len(Y) дат для исторических наблюдений
        dates_y = dates_main[-len(Y_original):]
        # Даты для прогноза - продолжаем после последней даты
        last_date = dates_y[-1]
        forecast_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(forecasts.shape[0])]
    else:
        dates_y = list(range(len(Y_original)))
        forecast_dates = list(range(len(Y_original), len(Y_original) + forecasts.shape[0]))

    # Количество графиков
    n_endog = Y.shape[1]
    n_exog = exog_data.shape[1] if exog_data is not None else 0

    if endog_vars is None:
        endog_vars = list(range(n_endog))
    if exog_vars is None and n_exog > 0:
        exog_vars = list(range(n_exog))

    n_plots = len(endog_vars) + (len(exog_vars) if exog_vars else 0)
    if n_plots == 0:
        print("Нет переменных для визуализации")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Стиль и палитра
    colors = ['#FF6B6B', '#FF8E8E', '#FFB5B5', '#FFD8D8']
    alphas = [0.15, 0.5]
    percentiles = [5, 10]
    plt.style.use('ggplot')

    current_plot = 0

    # --- Графики эндогенных переменных ---
    for var in endog_vars:
        ax = axes[current_plot]
        # Исторические данные
        ax.plot(dates_y, Y_original[:, var], 'b-', lw=2, label='Фактические данные')
        # Прогноз
        ax.plot(forecast_dates, forecasts_original[:, var], 'r--', lw=2, label='Прогноз (медиана)')
        # Доверительные интервалы
        for j in range(len(percentiles)):
            lower = np.percentile(samples_original[:, :, var], 2 * percentiles[j], axis=0)
            upper = np.percentile(samples_original[:, :, var], 100 - 2 * percentiles[j], axis=0)
            ax.fill_between(forecast_dates, lower, upper,
                            color=colors[j], alpha=alphas[j],
                            label=f'{100 - 2 * percentiles[j]}% интервал' if j == 0 else None)
        # Вертикальная линия
        if len(dates_y) > 0:
            ax.axvline(x=dates_y[-1], color='gray', linestyle='--', label='Начало прогноза')
        ax.set_title(f"{endog_names.get(var, f'Переменная {var + 1}')}")
        ax.set_xlabel('Дата')
        ax.grid(True)
        ax.legend(loc='upper left')

        # Форматирование дат
        if dates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates_y) // 12)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        current_plot += 1

    # --- Графики экзогенных переменных ---
    if exog_vars and exog_data is not None:
        # Обратное преобразование экзогенных данных
        exog_data_original = inverse_transform_data(exog_data, scaler_exog, exog_min)
        # Только последние len(exog_data_original) дат
        dates_exog = dates_main[-len(exog_data_original):]
        for var in exog_vars:
            ax = axes[current_plot]
            ax.plot(dates_exog, exog_data_original[:, var], 'g-', lw=2, label='Экзогенные данные')
            ax.set_title(f"{exog_names.get(var, f'Экзогенная {var + 1}')}")
            ax.set_xlabel('Дата')
            ax.grid(True)
            ax.legend(loc='upper left')

            if dates is not None:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates_exog) // 12)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            current_plot += 1

    plt.tight_layout()
    plt.show()


def optimize_minnesota_hyperparameters(Y, p, steps=5, n_samples=1000, exog=None):
    """
    Оптимизация гиперпараметров Миннесотского априора с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные
    - p: Порядок лага
    - steps: Число шагов прогноза для валидации
    - n_samples: Количество сэмплов
    - exog: Экзогенные переменные

    Возвращает:
    - Оптимальные значения [lambda1, lambda2, lambda3]
    """

    def objective(params):
        lambda1, lambda2, lambda3 = params
        split_idx = int(0.8 * len(Y))
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        exog_train, exog_val = None, None

        if exog is not None:
            exog_train, exog_val = exog[:split_idx], exog[split_idx:]

        post_mean, post_var = bvar_estimate(
            Y_train, p, 'minnesota',
            lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
            exog=exog_train
        )

        step_len = min(steps, len(Y_val))
        exog_future = exog_val[:step_len] if exog is not None and exog.shape[1] > 0 else None

        forecasts, _ = forecast_bvar(
            Y_train, post_mean, post_var, p,
            steps=step_len, n_samples=n_samples,
            exog_future=exog_future
        )

        return np.mean((forecasts - Y_val[:step_len]) ** 2)

    initial_params = [0.2, 0.5, 1.0]
    bounds = [(0.01, 1.0), (0.01, 1.0), (0.1, 10.0)]

    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
    return result.x


def optimize_normal_wishart_hyperparameters(Y, p, steps=5, n_samples=500, exog=None):
    """
    Оптимизация гиперпараметров Normal-Wishart априора с поддержкой экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - steps: Число шагов прогноза для валидации
    - n_samples: Количество сэмплов
    - exog: Экзогенные переменные (T x m), опционально

    Возвращает:
    - Оптимальные значения (v_0, s_0)
    """
    T, n = Y.shape

    def objective(params):
        v_0, s_0 = params
        split_idx = int(0.8 * T)
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        exog_train, exog_val = None, None

        if exog is not None:
            exog_train, exog_val = exog[:split_idx], exog[split_idx:]

        post_mean, post_var = bvar_estimate(
            Y_train, p, 'normal_wishart',
            v_0=int(np.round(v_0)),
            S_0=np.eye(n) * s_0,
            exog=exog_train
        )

        step_len = min(steps, len(Y_val))
        exog_future = exog_val[:step_len] if exog is not None and exog.shape[1] > 0 else None

        forecasts, _ = forecast_bvar(
            Y_train, post_mean, post_var, p,
            steps=step_len, n_samples=n_samples,
            exog_future=exog_future
        )

        return np.mean((forecasts - Y_val[:step_len]) ** 2)

    bounds = [(n, n + 10), (0.01, 10.0)]  # v_0, масштаб S_0
    init = [n + 2, 0.1]
    result = minimize(objective, init, bounds=bounds, method='L-BFGS-B')
    v_0_opt, s_0_opt = result.x
    return int(np.round(v_0_opt)), s_0_opt


def optimize_sims_zha_normal_flat_hyperparameters(Y, p, steps=5, n_samples=1000, exog=None):
    """
    Оптимизация гиперпараметров Sims-Zha Normal-Flat априора с экзогенными переменными.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - steps: Число шагов прогноза для валидации
    - n_samples: Количество сэмплов
    - exog: Экзогенные переменные (T x m), опционально

    Возвращает:
    - Оптимальные значения [lambda1, lambda2, lambda3, mu5, mu6]
    """

    def objective(params):
        lambda1, lambda2, lambda3, mu5, mu6 = params
        split_idx = int(0.8 * len(Y))
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        exog_train, exog_val = None, None

        if exog is not None:
            exog_train, exog_val = exog[:split_idx], exog[split_idx:]

        post_mean, post_var = bvar_estimate(
            Y_train, p, 'sims_zha_normal_flat',
            lambda1=lambda1, lambda2=lambda2,
            lambda3=lambda3, mu5=mu5, mu6=mu6,
            exog=exog_train
        )

        step_len = min(steps, len(Y_val))
        exog_future = exog_val[:step_len] if exog is not None and exog.shape[1] > 0 else None

        forecasts, _ = forecast_bvar(
            Y_train, post_mean, post_var, p,
            steps=step_len, n_samples=n_samples,
            exog_future=exog_future
        )

        return np.mean((forecasts - Y_val[:step_len]) ** 2)

    # Начальные значения и границы
    initial_params = [0.1, 0.5, 1.0, 0.1, 1.0]
    bounds = [(0.01, 1.0), (0.01, 1.0), (0.1, 10.0), (0.01, 0.5), (0.5, 5.0)]

    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
    return result.x


def optimize_sims_zha_normal_wishart_hyperparameters(Y, p, steps=5, n_samples=1000, exog=None):
    """
    Оптимизация гиперпараметров Sims-Zha Normal-Wishart с экзогенными переменными.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - steps: Горизонт прогнозирования для валидации
    - n_samples: Количество сэмплов
    - exog: Экзогенные переменные (T x m), опционально

    Возвращает:
    - Оптимальные значения [lambda1, lambda2, lambda3, mu5, mu6]
    """

    def objective(params):
        lambda1, lambda2, lambda3, mu5, mu6 = params
        split_idx = int(0.8 * len(Y))
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        exog_train, exog_val = None, None

        if exog is not None:
            exog_train, exog_val = exog[:split_idx], exog[split_idx:]

        post_mean, post_var = bvar_estimate(
            Y_train, p, 'sims_zha_normal_wishart',
            lambda1=lambda1, lambda2=lambda2,
            lambda3=lambda3, mu5=mu5, mu6=mu6,
            exog=exog_train
        )

        step_len = min(steps, len(Y_val))
        exog_future = exog_val[:step_len] if exog is not None and exog.shape[1] > 0 else None

        forecasts, _ = forecast_bvar(
            Y_train, post_mean, post_var, p,
            steps=step_len, n_samples=n_samples,
            exog_future=exog_future
        )

        return np.mean((forecasts - Y_val[:step_len]) ** 2)

    # Начальные значения и границы параметров
    initial_params = [0.1, 0.5, 1.0, 0.1, 1.0]
    bounds = [
        (0.01, 1.0),  # lambda1
        (0.01, 1.0),  # lambda2
        (0.1, 10.0),  # lambda3
        (0.01, 0.5),  # mu5
        (0.5, 5.0)  # mu6
    ]

    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
    return result.x


def calculate_bic_aic(Y, p, prior_type, prior_kwargs, criterion='bic', exog=None):
    """
    Расчёт BIC/AIC для BVAR модели с учётом экзогенных переменных.

    Параметры:
    - Y: Эндогенные переменные (T x n)
    - p: Порядок лага
    - prior_type: Тип априорного распределения
    - prior_kwargs: Параметры априора
    - criterion: 'bic' или 'aic'
    - exog: Экзогенные переменные (T x m), опционально

    Возвращает:
    - Значение информационного критерия
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Учитываем экзогенные переменные

    # Получаем оценки модели
    post_mean, post_var = bvar_estimate(Y, p, prior_type, exog=exog, **prior_kwargs)

    # Строим матрицу регрессоров с учётом экзогенных переменных
    X = np.ones((T - p, k))
    for t in range(p, T):
        X[t - p, 1:n * p + 1] = Y[t - p:t, :].flatten()
        if m > 0:
            X[t - p, n * p + 1:] = exog[t, :]

    # Вычисляем остатки
    residuals = Y[p:] - X @ post_mean.reshape(n, k).T
    sigma_hat = np.cov(residuals.T)

    # Логарифмическое правдоподобие
    ll = -0.5 * (T - p) * n * np.log(2 * np.pi) - 0.5 * (T - p) * np.log(np.linalg.det(sigma_hat)) \
         - 0.5 * (T - p) * n

    # Количество параметров (учитываем экзогенные переменные)
    num_params = n * k

    # Штрафной член
    if criterion == 'bic':
        penalty = num_params * np.log(T - p)
    else:  # 'aic'
        penalty = 2 * num_params

    score = -2 * ll + penalty

    return score

# Предобработка данных
def preprocess_data(Y, exog=None):
    # Сохраняем минимальные значения для обратного преобразования
    Y_min = np.min(Y, axis=0) if np.any(Y < 0) else 0

    # Сдвигаем в положительную область
    if np.any(Y < 0):
        Y = Y - Y_min + 1e-6

    # Логарифмирование
    Y_log = np.log1p(Y)

    # Стандартизация
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_log)

    # Аналогично для экзогенных переменных
    if exog is not None:
        exog_min = np.min(exog, axis=0) if np.any(exog < 0) else 0
        if np.any(exog < 0):
            exog = exog - exog_min + 1e-6
        exog_log = np.log1p(exog)
        scaler_exog = StandardScaler()
        exog_scaled = scaler_exog.fit_transform(exog_log)
    else:
        exog_scaled = None
        scaler_exog = None
        exog_min = None

    return Y_scaled, exog_scaled, scaler_Y, scaler_exog, Y_min, exog_min


def inverse_transform_data(Y_scaled, scaler, Y_min):
    """Полное обратное преобразование данных (стандартизация + логарифмирование + сдвиг)"""
    if Y_scaled is None or scaler is None:
        return Y_scaled

    # Обратная стандартизация
    Y_log = scaler.inverse_transform(Y_scaled)

    # Обратное логарифмирование
    Y = np.expm1(Y_log)

    # Обратный сдвиг
    if Y_min is not None:
        Y = Y + Y_min - 1e-6

    return Y


# ------------------------------------------------------------------
# Конструктор регрессоров
# ------------------------------------------------------------------
def build_X(Y, p, exog=None):
    """
    Построение матрицы регрессоров для BVAR с учётом экзогенных переменных.

    Параметры:
    - Y: Матрица эндогенных переменных (T x n)
    - p: Порядок лага
    - exog: Матрица экзогенных переменных (T x m), опционально

    Возвращает:
    - Матрицу регрессоров X размерности (T-p) x (n*p + 1 + m)
    """
    T, n = Y.shape
    m = exog.shape[1] if exog is not None else 0
    k = n * p + 1 + m  # Константа + лаги эндогенных + экзогенные

    X = np.ones((T - p, k))

    for t in range(p, T):
        # Лаги эндогенных переменных
        X[t - p, 1:n * p + 1] = Y[t - p:t, :].flatten()

        # Текущие значения экзогенных переменных
        if m > 0:
            X[t - p, n * p + 1:] = exog[t, :]

    return X


# Функция определения стандартных гиперпараметров априорных распределений
def default_kwargs(prior_name, n):
    if prior_name == 'minnesota':
        return {'lambda1': 0.2, 'lambda2': 0.5, 'lambda3': 1.0}
    if prior_name == 'normal_wishart':
        return {'v_0': n + 2, 'S_0': np.eye(n) * 0.1}
    if prior_name == 'sims_zha_normal_flat':
        return {'lambda1': 0.1, 'lambda2': 0.5, 'lambda3': 1.0, 'mu5': 0.1, 'mu6': 1.0}
    if prior_name == 'sims_zha_normal_wishart':
        return {'lambda1': 0.1, 'lambda2': 0.5, 'lambda3': 1.0,
                'mu5': 0.1, 'mu6': 1.0, 'v_0': n + 2, 'S_0': np.eye(n) * 0.1}
    return {}   # normal_flat


# Расчёт максимально доступной длины лага модели
def compute_max_p(T, n, m=0, max_ratio=0.25):
    max_p = 1
    for p in range(1, T):
        k = n * p + 1 + m
        total_params = n * k
        if total_params >= max_ratio * (T - p):
            break
        max_p = p
    return max_p


def load_data_from_excel(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Преобразование и очистка данных
        for col in ['Uninvestedfunds', 'NettoFunds', 'Reinvestedfunds', 'Totalclients', 'Investedfunds', 'Plannedrate']:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(' ', '', regex=False)
                .str.replace('\u202f', '', regex=False)
                .replace('', np.nan)
                .astype(float)
            )

        # Выделение данных
        dates = pd.to_datetime(df['Month'], format='%d.%m.%Y').dt.to_period('M')
        mask_endog = ~df[['Uninvestedfunds', 'NettoFunds', 'Reinvestedfunds', 'Totalclients']].isnull().any(axis=1)
        Y = df.loc[mask_endog, ['Uninvestedfunds', 'NettoFunds', 'Reinvestedfunds', 'Totalclients']].values

        mask_exog = ~df[['Investedfunds', 'Plannedrate']].isnull().any(axis=1)
        exog = df.loc[mask_exog, ['Investedfunds', 'Plannedrate']].values

        # Нормализация данных с сохранением минимальных значений
        Y, exog, scaler_Y, scaler_exog, Y_min, exog_min = preprocess_data(Y, exog)

        print(f"Y.shape={Y.shape}, exog.shape={exog.shape}")

        return Y, exog, scaler_Y, scaler_exog, dates, Y_min, exog_min

    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        return None, None, None, None, None, None, None


def user_interface(dates=None):
    """Консольный интерфейс для работы с реальными данными из Excel."""
    # Инициализация переменных
    prior_type = None
    prior_kwargs = {}
    forecasts = None
    conf = None
    prior_choice = None
    p = 1  # Значение по умолчанию для порядка лага модели
    steps = 0  # Длина прогноза будет определена автоматически

    # Автоматическая загрузка данных из Excel
    file_path = "C:/Users/lihop/Личные данные/Работа/Робофинанс/Темы исследований/2025/P2P-платформа/Прогноз RBC на май/Портфель.xlsx"
    sheet_name = "Eviews"

    print("\nЗагрузка данных из Excel...")
    Y, exog, scaler_Y, scaler_exog, dates, Y_min, exog_min = load_data_from_excel(file_path, sheet_name)

    if Y is None:
        raise ValueError("Не удалось загрузить данные из Excel. Проверьте файл и структуру данных.")

    print("\nДанные успешно загружены и нормализованы:")
    print(f"Эндогенные переменные: Uninvestedfunds, NettoFunds, Reinvestedfunds, Totalclients")
    print(f"Количество наблюдений: {len(Y)}")

    use_exog = exog is not None
    if use_exog:
        print(f"\nЭкзогенные переменные: Investedfunds, Plannedrate")
        print(f"Количество наблюдений экзогенных переменных: {len(exog)}")

        # Определение доступного диапазона будущих значений
        steps = len(exog) - len(Y)
        if steps <= 0:
            print("\nВнимание: Длина экзогенных переменных больше или равна эндогенным.")
            print("Необходимо задать будущие значения экзогенных переменных.")

            # Запрос способа задания будущих значений
            print("\nВыберите способ задания будущих значений экзогенных переменных:")
            print("1. Использовать последние доступные значения (продолжить текущий тренд)")
            print("2. Ввести вручную будущие значения")
            print("3. Прогнозировать с помощью ARIMA (автоматически для всех переменных)")
            print("4. Оставить существующие значения и перейти далее")
            future_choice = input("Введите номер (1-4): ").strip()

            if future_choice == '1':
                steps = int(input("\nВведите количество будущих периодов для прогноза: ") or "12")
                exog_future = np.tile(exog[-1, :], (steps, 1))
                print(f"\nИспользуются последние доступные значения для {steps} будущих периодов.")
            elif future_choice == '2':
                steps = int(input("\nВведите количество будущих периодов для прогноза: ") or "12")
                exog_future = np.zeros((steps, exog.shape[1]))
                print(f"\nВведите {steps} будущих значений для каждой из {exog.shape[1]} переменных:")
                for j in range(exog.shape[1]):
                    print(f"\nПеременная {j + 1}:")
                    for i in range(steps):
                        val = float(input(f"Период {i + 1}: "))
                        exog_future[i, j] = val
            elif future_choice == '3':
                steps = int(input("\nВведите количество будущих периодов для прогноза: ") or "12")
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    print("\nПрогнозирование будущих значений экзогенных переменных с помощью ARIMA...")
                    exog_future = np.zeros((steps, exog.shape[1]))
                    for j in range(exog.shape[1]):
                        best_order = (1, 1, 1)
                        model = ARIMA(exog[:, j], order=best_order)
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=steps)
                        exog_future[:, j] = forecast
                        print(f"Переменная {j + 1}: ARIMA{best_order} прогноз на {steps} периодов")
                except ImportError:
                    print("Ошибка: Для использования ARIMA необходимо установить statsmodels.")
                    exog_future = np.tile(exog[-1, :], (steps, 1))
            elif future_choice == '4':
                steps = 0
                exog_future = None
                print(
                    "\nБудут использоваться только существующие значения экзогенных переменных. "
                    "Прогноз не выполняется, анализ только на исторических данных.")
            else:
                steps = int(input("\nВведите количество будущих периодов для прогноза: ") or "12")
                print("Неверный выбор. Используются последние доступные значения.")
                exog_future = np.tile(exog[-1, :], (steps, 1))
        else:
            print(f"\nДоступно {steps} будущих значений эндогенных переменных для прогноза.")
            exog_future = exog[-steps:] if steps > 0 else None
    else:
        exog_future = None
        print("\nЭкзогенные переменные не обнаружены. Будут использованы только эндогенные переменные.")
        steps = 12  # Значение по умолчанию, если нет экзогенных переменных

    # Выбор порядка лага
    print("\nВыберите порядок лага p для BVAR:")
    print("1. Ввести вручную")
    print("2. Оптимизировать по BIC")
    print("3. Оптимизировать по AIC")
    lag_choice = input("Введите номер (1-3): ").strip()

    selected_prior_for_lag = None
    auto_apply_prior = False

    if lag_choice == '1':
        p = int(input("Введите число лагов (например, 1-5): ").strip())
    else:
        print("\nВыберите априор для оценки при оптимизации лагов:")
        print("1. Litterman-Minnesota prior")
        print("2. Normal-Flat prior")
        print("3. Normal-Wishart prior")
        print("4. Sims-Zha Normal-Flat prior")
        print("5. Sims-Zha Normal-Wishart prior")
        print("6. Optimal prior (рекомендуется)")
        prior_for_lag = input("Введите номер (1-6): ").strip()

        prior_type_map = {
            '1': 'minnesota',
            '2': 'normal_flat',
            '3': 'normal_wishart',
            '4': 'sims_zha_normal_flat',
            '5': 'sims_zha_normal_wishart',
            '6': 'modal'
        }

        selected_prior_for_lag = prior_type_map.get(prior_for_lag)

        if selected_prior_for_lag != 'modal':
            apply = input(f"\nИспользовать '{selected_prior_for_lag}' как основной априор? (y/n): ").strip().lower()
            auto_apply_prior = (apply == 'y')

        m = exog.shape[1] if use_exog and exog is not None else 0
        max_p = compute_max_p(len(Y), Y.shape[1], m)
        min_p = 1

        if selected_prior_for_lag == 'modal':
            priors_for_modal = ['minnesota', 'normal_flat', 'normal_wishart',
                                'sims_zha_normal_flat', 'sims_zha_normal_wishart']
            p_results = []
            for pt in priors_for_modal:
                kwargs = default_kwargs(pt, Y.shape[1])
                best_p = None
                best_score = float('inf')
                for candidate_p in range(min_p, max_p + 1):
                    score = calculate_bic_aic(Y, candidate_p, pt, kwargs,
                                              exog=exog,
                                              criterion='bic' if lag_choice == '2' else 'aic')
                    if score < best_score:
                        best_score, best_p = score, candidate_p
                p_results.append(best_p)
                print(f"{pt}: лучший p = {best_p}")

            counts = Counter(p_results)
            modal_p = max(counts, key=counts.get)
            ties = [k for k, v in counts.items() if v == counts[modal_p]]
            modal_p = min(ties)
            print(f"\nМодальное число лагов среди всех априоров: p = {modal_p}")
            p = modal_p
        else:
            kwargs = default_kwargs(selected_prior_for_lag, Y.shape[1])
            best_p = min_p
            best_score = float('inf')
            for candidate_p in range(min_p, max_p + 1):
                score = calculate_bic_aic(Y, candidate_p, selected_prior_for_lag, kwargs,
                                          exog=exog,
                                          criterion='bic' if lag_choice == '2' else 'aic')
                print(f"p={candidate_p}, score={score:.3f}")
                if score < best_score:
                    best_score = score
                    best_p = candidate_p
            print(f"Оптимальное число лагов: {best_p}")
            p = best_p

    print(f"\nВыбран порядок лагов p = {p}")

    # Выбор априора
    if not auto_apply_prior:
        print("\nВыберите тип априорного распределения для дальнейшей оценки и прогноза:")
        print("1. Litterman-Minnesota prior (стандартная рекурсивная BVAR)")
        print("2. Normal-Flat prior (слабая информация)")
        print("3. Normal-Wishart prior (усиление ковариации)")
        print("4. Sims-Zha Normal-Flat prior (структурная BVAR со слабой информацией)")
        print("5. Sims-Zha Normal-Wishart prior (структурная BVAR с усилением ковариации)")
        print("6. Автоматический подбор распределения с оптимизацией гиперпараметров")
        print("7. Медианная оценка по всем доступным распределениям")
        prior_choice = input("Введите номер (1-7): ").strip()

    if prior_choice == '7':
        # Медианная оценка по всем априорам
        print("\nРасчёт медианного прогноза по всем априорам...")
        priors_for_median = [
            'minnesota',
            'normal_flat',
            'normal_wishart',
            'sims_zha_normal_flat',
            'sims_zha_normal_wishart'
        ]

        all_forecasts = []
        all_conf_samples = []

        for prior_name in priors_for_median:
            print(f"\nОбработка априора: {prior_name}")
            try:
                if prior_name == 'minnesota':
                    lambda1, lambda2, lambda3 = optimize_minnesota_hyperparameters(Y, p, exog=exog)
                    kwargs = {'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3}
                elif prior_name == 'normal_wishart':
                    v_0, s_0 = optimize_normal_wishart_hyperparameters(Y, p, exog=exog)
                    kwargs = {'v_0': v_0, 'S_0': np.eye(Y.shape[1]) * s_0}
                elif prior_name == 'sims_zha_normal_flat':
                    lambda1, lambda2, lambda3, mu5, mu6 = optimize_sims_zha_normal_flat_hyperparameters(Y, p, exog=exog)
                    kwargs = {'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'mu5': mu5, 'mu6': mu6}
                elif prior_name == 'sims_zha_normal_wishart':
                    lambda1, lambda2, lambda3, mu5, mu6 = optimize_sims_zha_normal_wishart_hyperparameters(Y, p,
                                                                                                           exog=exog)
                    kwargs = {
                        'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3,
                        'mu5': mu5, 'mu6': mu6, 'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1
                    }
                else:
                    kwargs = {}

                post_mean, post_var = bvar_estimate(Y, p, prior_name, exog=exog, **kwargs)
                fc, ci = forecast_bvar(Y, post_mean, post_var, p, steps=steps,
                                       exog_future=exog_future if use_exog else None)
                all_forecasts.append(fc)
                all_conf_samples.append(ci['samples'])
                print(f"Успешно: {prior_name}")
            except Exception as e:
                print(f"Ошибка в {prior_name}: {str(e)}")
                continue

        if not all_forecasts:
            raise ValueError("Не удалось построить ни одного прогноза.")

        forecasts = np.median(np.stack(all_forecasts), axis=0)
        combined_samples = np.concatenate(all_conf_samples, axis=0)
        conf = {
            'samples': combined_samples,
            '5%': np.percentile(combined_samples, 5, axis=0),
            '95%': np.percentile(combined_samples, 95, axis=0)
        }
        prior_type = 'median_combined'

    elif prior_choice == '6':
        # Автоматический подбор лучшего априора
        print("\nАвтоматический подбор априора в процессе...")
        priors_to_test = [
            ('minnesota', optimize_minnesota_hyperparameters),
            ('normal_flat', None),
            ('normal_wishart', optimize_normal_wishart_hyperparameters),
            ('sims_zha_normal_flat', optimize_sims_zha_normal_flat_hyperparameters),
            ('sims_zha_normal_wishart', optimize_sims_zha_normal_wishart_hyperparameters)
        ]

        best_score = float('inf')
        best_prior = None
        best_params = None

        for prior_name, optimizer in priors_to_test:
            print(f"\nТестирование априора: {prior_name}")
            try:
                if optimizer:
                    if prior_name == 'minnesota':
                        params = optimizer(Y, p, exog=exog)
                        kwargs = {'lambda1': params[0], 'lambda2': params[1], 'lambda3': params[2]}
                    elif prior_name == 'normal_wishart':
                        v_0, s_0 = optimizer(Y, p, exog=exog)
                        kwargs = {'v_0': v_0, 'S_0': np.eye(Y.shape[1]) * s_0}
                    elif prior_name.startswith('sims_zha'):
                        params = optimizer(Y, p, exog=exog)
                        kwargs = {
                            'lambda1': params[0], 'lambda2': params[1], 'lambda3': params[2],
                            'mu5': params[3], 'mu6': params[4]
                        }
                        if prior_name == 'sims_zha_normal_wishart':
                            kwargs.update({'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1})
                else:
                    kwargs = {}

                split_idx = int(0.8 * len(Y))
                Y_train, Y_val = Y[:split_idx], Y[split_idx:]
                exog_train = exog[:split_idx] if use_exog and exog is not None else None
                exog_val = exog[split_idx:] if use_exog and exog is not None else None

                post_mean, post_var = bvar_estimate(Y_train, p, prior_name, exog=exog_train, **kwargs)

                step_len = min(steps, len(Y_val))
                exog_future_val = exog_val[:step_len] if use_exog and exog is not None else None

                fc, _ = forecast_bvar(Y_train, post_mean, post_var, p, steps=step_len, exog_future=exog_future_val)
                score = np.mean((fc - Y_val[:step_len]) ** 2)
                print(f"Ошибка прогноза: {score:.4f}")

                if score < best_score:
                    best_score = score
                    best_prior = prior_name
                    best_params = kwargs.copy()
            except Exception as e:
                print(f"Ошибка при тестировании {prior_name}: {str(e)}")
                continue

        if best_prior is None:
            print("\nНе удалось подобрать априор. Используется Minnesota prior по умолчанию.")
            best_prior = 'minnesota'
            best_params = {'lambda1': 0.2, 'lambda2': 0.5, 'lambda3': 1.0}

        print(f"\nВыбран лучший априор: {best_prior} с ошибкой {best_score:.4f}")
        print("Оптимальные параметры:", best_params)

        prior_type = best_prior
        prior_kwargs = best_params
        post_mean, post_var = bvar_estimate(Y, p, prior_type, exog=exog, **prior_kwargs)
        forecasts, conf = forecast_bvar(Y, post_mean, post_var, p, steps=steps,
                                        exog_future=exog_future if use_exog else None, scaler_Y=scaler_Y)

    else:
        # Обработка стандартных априоров (1-5)
        prior_type_map = {
            '1': 'minnesota',
            '2': 'normal_flat',
            '3': 'normal_wishart',
            '4': 'sims_zha_normal_flat',
            '5': 'sims_zha_normal_wishart'
        }
        prior_type = prior_type_map.get(prior_choice)

        if prior_type is None:
            print("Неверный выбор априора. Используется Minnesota prior по умолчанию.")
            prior_type = 'minnesota'
            prior_kwargs = {'lambda1': 0.2, 'lambda2': 0.5, 'lambda3': 1.0}

        if prior_type == 'minnesota':
            print("\nМиннесотский априор:")
            print("1. Значения по умолчанию")
            print("2. Ввести λ-параметры вручную")
            print("3. Оптимизировать λ-параметры")
            mn_choice = input("Выберите вариант (1-3): ").strip()

            if mn_choice == '2':
                prior_kwargs = {
                    'lambda1': float(input("lambda1 (0.01-1.0): ") or 0.2),
                    'lambda2': float(input("lambda2 (0.01-1.0): ") or 0.5),
                    'lambda3': float(input("lambda3 (0.1-10.0): ") or 1.0)
                }
            elif mn_choice == '3':
                print("Оптимизация гиперпараметров...")
                λ1, λ2, λ3 = optimize_minnesota_hyperparameters(Y, p, exog=exog)
                print(f"Оптимальные: λ1={λ1:.3f}, λ2={λ2:.3f}, λ3={λ3:.3f}")
                prior_kwargs = {'lambda1': λ1, 'lambda2': λ2, 'lambda3': λ3}
            else:
                prior_kwargs = {'lambda1': 0.2, 'lambda2': 0.5, 'lambda3': 1.0}

        elif prior_type == 'normal_flat':
            print("\nИспользуется Normal-Flat априор (параметров нет).")
            prior_kwargs = {}

        elif prior_type == 'normal_wishart':
            print("\nNormal-Wishart априор:")
            print("1. Значения по умолчанию")
            print("2. Ввести параметры вручную")
            print("3. Оптимизировать параметры")
            nw_choice = input("Выберите вариант (1-3): ").strip()

            if nw_choice == '2':
                prior_kwargs = {
                    'v_0': int(input(f"v₀ (> {Y.shape[1] - 1}, по умолчанию {Y.shape[1] + 2}): ") or Y.shape[1] + 2),
                    'S_0': np.eye(Y.shape[1]) * float(input("Масштаб S₀ (по умолчанию 0.1): ") or 0.1)
                }
            elif nw_choice == '3':
                print("Оптимизация гиперпараметров Normal-Wishart...")
                v_0_opt, s_0_opt = optimize_normal_wishart_hyperparameters(Y, p, exog=exog)
                print(f"Оптимальные параметры: v₀={v_0_opt}, масштаб S₀={s_0_opt:.3f}")
                prior_kwargs = {'v_0': v_0_opt, 'S_0': np.eye(Y.shape[1]) * s_0_opt}
            else:
                prior_kwargs = {'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1}

        elif prior_type.startswith('sims_zha'):
            print(f"\n{prior_type} априор:")
            print("1. Значения по умолчанию")
            print("2. Ввести параметры вручную")
            print("3. Оптимизировать параметры")
            sz_choice = input("Выберите вариант (1-3): ").strip()

            if sz_choice == '2':
                prior_kwargs = {
                    'lambda1': float(input("λ1 (0.01-1.0): ") or 0.1),
                    'lambda2': float(input("λ2 (0.01-1.0): ") or 0.5),
                    'lambda3': float(input("λ3 (0.1-10.0): ") or 1.0),
                    'mu5': float(input("μ5 (0.01-0.5): ") or 0.1),
                    'mu6': float(input("μ6 (0.5-5.0): ") or 1.0)
                }
                if prior_type == 'sims_zha_normal_wishart':
                    prior_kwargs.update({'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1})
            elif sz_choice == '3':
                print(f"Оптимизация параметров {prior_type}...")
                if prior_type == 'sims_zha_normal_flat':
                    params = optimize_sims_zha_normal_flat_hyperparameters(Y, p, exog=exog)
                else:
                    params = optimize_sims_zha_normal_wishart_hyperparameters(Y, p, exog=exog)
                print(
                    f"Оптимальные: λ1={params[0]:.3f}, λ2={params[1]:.3f}, λ3={params[2]:.3f}, μ5={params[3]:.3f}, μ6={params[4]:.3f}")
                prior_kwargs = {
                    'lambda1': params[0], 'lambda2': params[1], 'lambda3': params[2],
                    'mu5': params[3], 'mu6': params[4]
                }
                if prior_type == 'sims_zha_normal_wishart':
                    prior_kwargs.update({'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1})
            else:
                prior_kwargs = {
                    'lambda1': 0.1, 'lambda2': 0.5, 'lambda3': 1.0,
                    'mu5': 0.1, 'mu6': 1.0
                }
                if prior_type == 'sims_zha_normal_wishart':
                    prior_kwargs.update({'v_0': Y.shape[1] + 2, 'S_0': np.eye(Y.shape[1]) * 0.1})

        post_mean, post_var = bvar_estimate(Y, p, prior_type, exog=exog, **prior_kwargs)
        forecasts, conf = forecast_bvar(Y, post_mean, post_var, p, steps=steps,
                                        exog_future=exog_future if use_exog else None, scaler_Y=scaler_Y)

    # Выбор переменных для визуализации
    print("\nВыберите переменные для визуализации:")
    print(f"\nЭндогенные переменные (доступно: 1-{Y.shape[1]}):")
    endog_choice = input("Введите номера через пробел (например '1 2'), или оставьте пустым для всех: ").strip()
    try:
        endog_vars = [int(x) - 1 for x in endog_choice.split() if x.isdigit() and 1 <= int(x) <= Y.shape[1]]
        if not endog_vars and endog_choice != '':
            print("Неверный ввод. Будут показаны все эндогенные переменные.")
            endog_vars = None
    except:
        print("Неверный ввод. Будут показаны все эндогенные переменные.")
        endog_vars = None

    exog_vars = None
    if use_exog and exog is not None:
        print(f"\nЭкзогенные переменные (доступно: 1-{exog.shape[1]}):")
        exog_choice = input("Введите номера через пробел (например '1 2'), или оставьте пустым для всех: ").strip()
        try:
            exog_vars = [int(x) - 1 for x in exog_choice.split() if x.isdigit() and 1 <= int(x) <= exog.shape[1]]
            if not exog_vars and exog_choice != '':
                print("Неверный ввод. Будут показаны все экзогенные переменные.")
                exog_vars = None
        except:
            print("Неверный ввод. Будут показаны все экзогенные переменные.")
            exog_vars = None

    # Вывод результатов
    print("\nРезультаты:")
    if prior_choice == '7':
        print("Тип априора: Медианный прогноз по всем априорам")
    else:
        print(f"Тип априора: {prior_type}")
        if prior_kwargs:
            print("Параметры априора:",
                  {k: (v if not isinstance(v, np.ndarray) else v[0, 0]) for k, v in prior_kwargs.items()})

    print(f"\nДлина прогноза: {steps} периодов")
    print(f"Y shape: {Y.shape}, forecasts shape: {forecasts.shape}")

    # Визуализация
    plot_results(Y, forecasts, conf, p,
                 endog_vars=endog_vars,
                 exog_vars=exog_vars,
                 exog_data=exog,
                 scaler_Y=scaler_Y,
                 scaler_exog=scaler_exog,
                 dates=dates,
                 Y_min=Y_min,
                 exog_min=exog_min)


if __name__ == "__main__":
    user_interface()
