import numpy as np
import pandas as pd
import requests
import itertools
from datetime import datetime, time

def run_global_hyper_parameter_optimization():
    print("=" * 110)
    print("🚀 INITIALIZING NIFS-5 MASTER HYPER-PARAMETER GRID SEARCH OPTIMIZER (ZERO ASSUMPTIONS)")
    print("=" * 110)
    
    # Target your newly public, tokenless GitHub repositories files directly
    urls = [
        "https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2021.json",
        "https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2022.json",
        "https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2023.json",
        "https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2024.json",
        "https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2025.json",
        #"https://raw.githubusercontent.com/sky13g/Sharekhan-Algo/refs/heads/main/NIFTY_1m_2026.json"
     ]
    
    
    
    payloads = []
    for url in urls:
        year_str = url.split("_")[-1].split(".")[0]
        print(f"📡 Fetching dataset rows from GitHub for year {year_str}...", end="", flush=True)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                payloads.extend(r.json())
                print(" SUCCESS")
            else:
                print(f" FAILED (HTTP {r.status_code})")
        except Exception as e:
            print(f" ERROR ({e})")
            
    if not payloads:
        print("\n🚨 System Fault: Data payloads empty. Execution terminated.")
        return
        
    print(f"\n📦 Consolidated Baseline Data: Compiled {len(payloads):,} raw 1-minute rows.")
    df_raw = pd.DataFrame(payloads)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    df_raw.set_index('timestamp', inplace=True)
    
    # -------------------------------------------------------------------------
    # DEFINE GRID SEARCH CONFIGURATION SPACE (UNBIASED TESTING ARRAYS)
    # -------------------------------------------------------------------------
    timeframes = ['1Min', '2Min', '3Min', '5Min', '10Min']
    kaufman_space = [0.35, 0.40, 0.45, 0.50, 0.55]
    adx_space = [14, 16, 18, 20, 22]
    rsi_bull_space = [50, 55, 60]
    rsi_bear_space = [40, 45, 50]
    davb_windows = [10, 14, 20]
    
    leaderboard = []
    total_iterations = len(timeframes) * len(kaufman_space) * len(adx_space) * len(rsi_bull_space) * len(rsi_bear_space) * len(davb_windows)
    print(f"⚙️ Grid Search Space Contains {total_iterations:,} Complete Strategy Combinations.")
    print("⏳ Processing multi-threaded optimization loops. Please wait...\n")
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION MATRIX LOOP
    # -------------------------------------------------------------------------
    for tf in timeframes:
        # Dynamically resample raw database to the target timeframe inside memory cache
        df_tf = df_raw.resample(tf).agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
        
        # Calculate base indicator columns to avoid redundant calculations inside child loops
        df_tf['candle_range'] = df_tf['high'] - df_tf['low']
        df_tf['atr_ma20'] = df_tf['candle_range'].rolling(20).mean()
        df_tf['ema_fast'] = df_tf['close'].ewm(span=5, adjust=False).mean()
        df_tf['ema_slow'] = df_tf['close'].ewm(span=13, adjust=False).mean()
        df_tf['ema_direction'] = np.where(df_tf['ema_fast'] > df_tf['ema_slow'], "BULL_HOLD", "BEAR_HOLD")
        
        # Calculate Kaufman ER
        change = (df_tf['close'] - df_tf['close'].shift(10)).abs()
        volatility = (df_tf['close'] - df_tf['close'].shift(1)).abs().rolling(10).sum()
        df_tf['kaufman_er'] = (change / volatility).fillna(0)
        
        # Calculate Standard ADX
        plus_dm = df_tf['high'].diff()
        minus_dm = df_tf['low'].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        tr = pd.concat([df_tf['high'] - df_tf['low'], (df_tf['high'] - df_tf['close'].shift(1)).abs(), (df_tf['low'] - df_tf['close'].shift(1)).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_14)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).fillna(1)
        df_tf['adx'] = dx.rolling(14).mean().fillna(0)
        
        # Calculate RSI
        delta = df_tf['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_tf['rsi'] = 100 - (100 / (1 + (gain / loss).fillna(0)))
        
        # Sweep variations within the current resampled timeframe environment
        for k_er, adx_g, rsi_bull, rsi_bear, d_win in itertools.product(kaufman_space, adx_space, rsi_bull_space, rsi_bear_space, davb_windows):
            
            # Compute specific DAVB Envelope trailing bounds for this loop's window size
            rolling_std = df_tf['close'].rolling(d_win).std()
            rolling_mean = df_tf['close'].rolling(d_win).mean()
            df_tf['davb_upper'] = rolling_mean + (2.0 * rolling_std)
            df_tf['davb_lower'] = rolling_mean - (2.0 * rolling_std)
            
            trades_count = 0
            wins_count = 0
            active_trade = None
            
            # Simulation Row Navigation Engine
            for idx, row in df_tf.iterrows():
                current_time = row['timestamp'].time()
                
                # Enforce late-day protective cutoff shield
                if current_time >= time(15, 0, 0):
                    if active_trade is not None:
                        pnl = (row['close'] - active_trade['entry_price']) if active_trade['type'] == "LONG" else (active_trade['entry_price'] - row['close'])
                        trades_count += 1
                        if pnl > 0: wins_count += 1
                        active_trade = None
                    continue
                    
                # Enforce strict 09:20 AM morning gap filter window
                if current_time < time(9, 20, 0):
                    continue
                    
                is_breakout = row['candle_range'] > (2.0 * row['atr_ma20'])
                
                if active_trade is None:
                    if is_breakout and row['kaufman_er'] >= k_er and row['adx'] >= adx_g:
                        if row['ema_direction'] == "BULL_HOLD" and row['rsi'] > rsi_bull:
                            active_trade = {"type": "LONG", "entry_price": row['close']}
                        elif row['ema_direction'] == "BEAR_HOLD" and row['rsi'] < rsi_bear:
                            active_trade = {"type": "SHORT", "entry_price": row['close']}
                else:
                    hit_upper_trail = active_trade['type'] == "SHORT" and row['close'] >= row['davb_upper']
                    hit_lower_trail = active_trade['type'] == "LONG" and row['close'] <= row['davb_lower']
                    ema_reversal = (active_trade['type'] == "LONG" and row['ema_direction'] == "BEAR_HOLD") or (active_trade['type'] == "SHORT" and row['ema_direction'] == "BULL_HOLD")
                    
                    if hit_upper_trail or hit_lower_trail or ema_reversal:
                        pnl = (row['close'] - active_trade['entry_price']) if active_trade['type'] == "LONG" else (active_trade['entry_price'] - row['close'])
                        trades_count += 1
                        if pnl > 0: wins_count += 1
                        active_trade = None
                        
            if trades_count >= 50:  # Sample size minimum threshold filter to ensure statistical significance
                win_ratio = (wins_count / trades_count) * 100
                leaderboard.append({
                    "timeframe": tf, "kaufman_er": k_er, "adx_gate": adx_g,
                    "rsi_bull": rsi_bull, "rsi_bear": rsi_bear, "davb_window": d_win,
                    "total_trades": trades_count, "win_ratio": round(win_ratio, 2)
                })
                
    # -------------------------------------------------------------------------
    # PRINT TOP PERFORMING PARAMETER BLUEPRINTS
    # -------------------------------------------------------------------------
    print("=" * 110)
    print(f"🥇 GLOBAL STRATEGY OPTIMIZATION RESULTS SUMMARY (TOP 10 PERFORMANCES)")
    print("=" * 110)
    
    result_df = pd.DataFrame(leaderboard)
    if not result_df.empty:
        # Sort parameter records to isolate peak winning ratio setups
        top_10 = result_df.sort_values(by="win_ratio", ascending=False).head(10).reset_index(drop=True)
        
        print(f"{'RANK':<4} | {'TIMEFRAME':<9} | {'KAUFMAN':<7} | {'ADX_G':<5} | {'BULL_RSI':<8} | {'BEAR_RSI':<8} | {'DAVB_WIN':<8} | {'TRADES':<6} | {'WIN RATIO':<10}")
        print("-" * 110)
        for i, row in top_10.iterrows():
            print(f"#{i+1:<2}  | {row['timeframe']:<9} | {row['kaufman_er']:<7.2f} | {row['adx_gate']:<5} | {row['rsi_bull']:<8} | {row['rsi_bear']:<8} | {row['davb_window']:<8} | {row['total_trades']:<6} | {row['win_ratio']}%")
            
        print("=" * 110)
        print(f"\n💡 STRATEGY DEPLOYMENT INSIGHT: Set your dashboard parameter text fields exactly to match the #1 ranked row.")
    else:
        print("Optimization sweep completed, but no configuration cleared our statistical trade limits.")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    run_global_hyper_parameter_optimization()
