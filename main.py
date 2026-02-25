# -*- coding: utf-8 -*-
"""
Delta Trader - Backtest + Optimización + Live (Batch Testing 1 semana)
Dual Timeframe, RSI/ADX/ATR, SQ3 Lite persistente, plug & play para Railway/Delta Space
"""

import time
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np

from sq3lite_db import init_database, save_optimization_result
from fetch_data import fetch_klines
from indicators import compute_rsi, compute_adx, compute_atr

# ==================== CONFIGURACIÓN ====================
SYMBOL = 'ETHUSDT'
HOURS = 168  # 1 semana
BASE_CAPITAL = 1000
MAX_LEVERAGE = 20
MIN_WIN_RATE_FOR_LEVERAGE = 0.4
SLIPPAGE = 0.001
COMMISSION = 0.001

# Rango de optimización: RSI/ADX de 2 en 2
RSI_PERIODS = list(range(2, 30, 2))  # 2,4,6,...28
ADX_PERIODS = list(range(2, 30, 2))
ATR_PERIODS = list(range(2, 30, 2))
MULT_STOP_RANGE = [1.0, 1.5, 2.0, 2.5]
MULT_TP_RANGE = [1.0, 1.5, 2.0, 2.5]

# Timeframes
TIMEFRAMES = {
    '1m': '1min',
    '3m': '3min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h'
}

# Combinaciones de TF: entrada, tendencia
TF_COMBINATIONS = [
    ('1m', '5m'), ('1m', '15m'), ('1m', '1h'),
    ('3m', '15m'), ('3m', '1h'), ('5m', '1h'),
    ('5m', '4h'), ('15m', '1h'), ('15m', '4h'), ('30m', '4h')
]

# ==================== UTILS ====================
def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

# ==================== BACKTEST ====================
def backtest_dual(df_entrada, df_tendencia, rsi_period, adx_period, atr_period, mult_stop, mult_tp, use_slope=True):
    """Backtest dual timeframe con parámetros dinámicos"""
    df_entrada = df_entrada.copy()
    df_tendencia = df_tendencia.copy()

    # Indicadores
    df_entrada['RSI'] = compute_rsi(df_entrada['close'], rsi_period)
    df_entrada['ATR'] = compute_atr(df_entrada, atr_period)
    adx_df = compute_adx(df_tendencia, adx_period)
    df_tendencia = df_tendencia.reindex(df_entrada.index, method='ffill')
    df = df_entrada.join(adx_df)

    position = None
    entry_price = 0.0
    entry_atr = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    take_profit = 0.0
    equity_curve = [0.0]
    trades = []

    for idx, row in df.iterrows():
        # Tendencia
        long_trend = row['ADX'] > adx_period and row['DI_plus'] > row['DI_minus']
        short_trend = row['ADX'] > adx_period and row['DI_minus'] > row['DI_plus']
        if use_slope:
            long_trend = long_trend and (row['ADX_slope'] > 0)
            short_trend = short_trend and (row['ADX_slope'] < 0)

        # Entrada
        long_signal = row['RSI'] < 30
        short_signal = row['RSI'] > 70

        # Gestión de posición
        exit_reason = None
        exit_price = None

        if position is None:
            if long_trend and long_signal:
                position = 'long'
                entry_price = row['close'] * (1 + SLIPPAGE)
                entry_atr = row['ATR']
                extreme_price = row['high']
                stop_price = extreme_price - mult_stop * entry_atr
                take_profit = entry_price + mult_tp * entry_atr
            elif short_trend and short_signal:
                position = 'short'
                entry_price = row['close'] * (1 - SLIPPAGE)
                entry_atr = row['ATR']
                extreme_price = row['low']
                stop_price = extreme_price + mult_stop * entry_atr
                take_profit = entry_price - mult_tp * entry_atr
        else:
            # Actualizar stops
            if position == 'long':
                extreme_price = max(extreme_price, row['high'])
                stop_price = extreme_price - mult_stop * entry_atr
                if row['low'] <= stop_price:
                    exit_reason = 'trailing_stop'
                    exit_price = stop_price
                elif row['high'] >= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif short_trend:
                    exit_reason = 'tendencia_opuesta'
                    exit_price = row['close']
            else:
                extreme_price = min(extreme_price, row['low'])
                stop_price = extreme_price + mult_stop * entry_atr
                if row['high'] >= stop_price:
                    exit_reason = 'trailing_stop'
                    exit_price = stop_price
                elif row['low'] <= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif long_trend:
                    exit_reason = 'tendencia_opuesta'
                    exit_price = row['close']

            # Registrar trade
            if exit_reason:
                if position == 'long':
                    exit_price_adj = exit_price * (1 - SLIPPAGE)
                    ret = (exit_price_adj - entry_price) / entry_price - COMMISSION
                else:
                    exit_price_adj = exit_price * (1 + SLIPPAGE)
                    ret = (entry_price - exit_price_adj) / entry_price - COMMISSION
                trades.append({'tipo': position, 'entrada': entry_price, 'salida': exit_price_adj, 'retorno': ret})
                equity_curve.append(equity_curve[-1] + ret)
                position = None

    # Métricas
    if trades:
        profits = [t['retorno'] for t in trades]
        total_profit = sum(profits)
        num_trades = len(trades)
        win_rate = sum(1 for p in profits if p > 0) / num_trades
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        max_dd = ((running_max - equity) / (1 + running_max)).max()
    else:
        total_profit = 0
        num_trades = 0
        win_rate = 0
        max_dd = 0

    return {'profit': total_profit, 'trades': num_trades, 'win_rate': win_rate, 'max_dd': max_dd}

# ==================== OPTIMIZACIÓN ====================
def optimizar_combinacion(dfs, tf_entrada, tf_tendencia):
    df_entrada = dfs[tf_entrada]
    df_tendencia = dfs[tf_tendencia]

    mejor_profit = -np.inf
    mejor_params = None
    mejores_metricas = None

    total = len(RSI_PERIODS) * len(ADX_PERIODS) * len(ATR_PERIODS) * len(MULT_STOP_RANGE) * len(MULT_TP_RANGE) * 2
    count = 0

    for rsi_p, adx_p, atr_p, mult_stop, mult_tp, use_slope in product(
        RSI_PERIODS, ADX_PERIODS, ATR_PERIODS, MULT_STOP_RANGE, MULT_TP_RANGE, [True, False]
    ):
        count += 1
        if count % 100 == 0:
            print(f"Progreso {count}/{total}")
        metrics = backtest_dual(df_entrada, df_tendencia, rsi_p, adx_p, atr_p, mult_stop, mult_tp, use_slope)
        if metrics['profit'] > mejor_profit:
            mejor_profit = metrics['profit']
            mejor_params = {
                'rsi_period': rsi_p, 'adx_period': adx_p, 'atr_period': atr_p,
                'mult_stop': mult_stop, 'mult_tp': mult_tp, 'use_slope': use_slope
            }
            mejores_metricas = metrics

    # Guardar mejor resultado
    if mejor_params:
        save_optimization_result(tf_entrada, tf_tendencia, mejor_params, mejores_metricas)
        print(f"Mejor {tf_entrada}/{tf_tendencia}: Profit={mejores_metricas['profit']:.4f}, WR={mejores_metricas['win_rate']*100:.1f}%")
    return mejor_params, mejores_metricas

# ==================== MAIN ====================
def main():
    init_database()
    print("Descargando velas 1m (1 semana)...")
    df_1m = fetch_klines(SYMBOL, '1m', HOURS)

    dfs = {nombre: resample_ohlc(df_1m, rule) for nombre, rule in TIMEFRAMES.items()}

    # Optimizar cada combinación
    for tf_entrada, tf_tendencia in TF_COMBINATIONS:
        print(f"\n--- Combinación: {tf_entrada}/{tf_tendencia} ---")
        optimizar_combinacion(dfs, tf_entrada, tf_tendencia)

    print("\nOptimización completada. BD lista para live trading.")

if __name__ == "__main__":
    main()
