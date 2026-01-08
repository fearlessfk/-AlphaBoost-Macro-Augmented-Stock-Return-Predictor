# %%
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import math
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 数据加载
df = pd.read_csv('my_portfolio_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
df.drop('Unnamed: 0',axis=1,inplace=True)

# %%
# %%
# RSI计算函数（不变）
def calculate_rsi(prices, window=14):
    prices = prices.shift(1)  # 避免未来数据泄露
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 核心训练函数（调整均值模型为涨跌幅均值预测）
def xgb_train(df,ticker):
    # 1. 前置校验
    df = df.copy()
    if 'Close' not in df.columns or len(df) < 100:
        print(f'❌ 【{ticker}】数据不足/缺失Close列，跳过')
        return None
    
    # 2. 特征工程
    df['ma5'] = df['Close'].shift(1).rolling(5, min_periods=5).mean()
    df['pct_change'] = df['Close'].pct_change(1)
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)
    df['lag_1_pct'] = df['lag_1'].pct_change(1)
    df['daily_pct'] = df['Close'].pct_change(1)  # 单日涨跌幅
    df['lag_3_pct'] = df['daily_pct'].shift(1).rolling(3, min_periods=1).mean()
    df['lag_1_ratio'] = df['Close'] / df['lag_1']
    df['lag_2_ratio'] = df['lag_1'] / df['lag_2']
    df['ma10'] = df['Close'].shift(1).rolling(10).mean()
    df['ma20'] = df['Close'].shift(1).rolling(20).mean()
    df['volatility_7d'] = df['Close'].shift(1).rolling(7).std()
    df['rsi_14d'] = calculate_rsi(df['Close'], 14)
    df['price_ma5_ratio'] = df['Close'] / df['ma5']
    
    # 目标变量：未来5日涨跌幅均值（XGBoost预测目标）
    df['future_1d_pct'] = df['Close'].shift(-1).pct_change(1)  # 次日涨跌幅（真实值）
    df['target_pct'] = df['future_1d_pct'].rolling(5, min_periods=1).mean()  # 5日均值（XGBoost训练目标）
    
    # ========== 核心调整：均值模型改为过去5天涨跌幅均值 ==========
    # 计算过去5天涨跌幅均值（用于预测次日涨跌幅）
    df['past_5d_pct_mean'] = df['daily_pct'].shift(1).rolling(5, min_periods=1).mean()
    df = df.dropna()
    
    if len(df) < 80:
        print(f'❌ 【{ticker}】特征生成后数据不足，跳过')
        return None
    
    # 3. 滚动回测参数
    FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_1_pct', 'lag_3_pct', 'lag_1_ratio', 'lag_2_ratio', 
                'ma5', 'ma10', 'ma20', 'pct_change', 'volatility_7d', 'rsi_14d', 'price_ma5_ratio']
    train_window = 90
    test_window = 10   
    step = 10          
    
    # 4. 初始化存储变量（新增涨跌幅均值相关）
    all_y_test = []          # 真实涨跌幅（target_pct）
    all_y_pred_original = [] # XGBoost原始预测涨跌幅
    all_y_pred_calibrated = []# XGBoost校准后预测涨跌幅
    all_test_close = []      # 测试集收盘价
    all_test_index = []      # 测试集日期
    all_pct_mse_original = []# 原始MSE
    all_test_past_5d_pct_mean = []  # 测试集过去5天涨跌幅均值
    all_test_future_1d_pct = []     # 测试集真实次日涨跌幅
    
    # 5. 滚动回测主循环
    max_start = len(df) - train_window - test_window
    if max_start <= 0:
        print(f'❌ 【{ticker}】数据量不足以支撑滚动回测，跳过')
        return None
    
    for start in range(0, max_start, step):
        train_end = start + train_window
        test_end = train_end + test_window
        
        train = df.iloc[start:train_end]
        test = df.iloc[train_end:test_end]
        
        if len(test) < test_window:
            break
        
        X_train, y_train = train[FEATURES], train['target_pct']
        X_test, y_test = test[FEATURES], test['target_pct']
        
        # 训练模型
        model = xgb.XGBRegressor(
            n_estimators=100,         
            learning_rate=0.05,       
            max_depth=3,              
            subsample=0.9,            
            colsample_bytree=0.9,     
            reg_alpha=0.01,          
            reg_lambda=0.1,           
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        model.fit(X_train, y_train)
        
        # 预测+校准
        y_pred = model.predict(X_test)
        pred_mean = y_pred.mean()
        true_mean = y_test.mean()
        y_pred_calibrated = y_pred - pred_mean + true_mean
        
        # 存储结果（新增涨跌幅均值和真实次日涨跌幅）
        all_pct_mse_original.append(mean_squared_error(y_test, y_pred))
        all_y_test.extend(y_test.values)
        all_y_pred_original.extend(y_pred)
        all_y_pred_calibrated.extend(y_pred_calibrated)
        all_test_close.extend(test['Close'].values)
        all_test_index.extend(test.index)
        all_test_past_5d_pct_mean.extend(test['past_5d_pct_mean'].values)
        all_test_future_1d_pct.extend(test['future_1d_pct'].values)
    
    # 结果合并
    all_y_test = np.array(all_y_test)
    all_y_pred_calibrated = np.array(all_y_pred_calibrated)
    all_test_close = np.array(all_test_close)
    all_test_past_5d_pct_mean = np.array(all_test_past_5d_pct_mean)
    all_test_future_1d_pct = np.array(all_test_future_1d_pct)
    
    if len(all_y_test) == 0:
        print(f'❌ 【{ticker}】滚动回测无有效结果，跳过')
        return None
    
    # 评估涨跌幅预测效果
    pct_mse_original = np.mean(all_pct_mse_original)
    pct_mse_optimized = mean_squared_error(all_y_test, all_y_pred_calibrated)
    pct_r2 = r2_score(all_y_test, all_y_pred_calibrated)
    
    # ========== 基准对比（统一基于涨跌幅预测推导价格） ==========
    # 真实次日价：T+1日真实收盘价
    y_true_price = all_test_close[1:] * (1 + all_test_future_1d_pct[:-1])
    
    # 1. 傻瓜模型：T日收盘价 → 预测T+1日价（无涨跌幅预测，直接用当日价）
    y_naive_price = all_test_close[:-1]
    
    # 2. 均值模型：用过去5天涨跌幅均值 → 预测T+1日涨跌幅 → 推导价格
    # 均值模型预测的涨跌幅：past_5d_pct_mean
    y_mean_pct_pred = all_test_past_5d_pct_mean[:-1]
    # 均值模型预测价格 = T日收盘价 × (1 + 预测涨跌幅)
    y_mean_price = all_test_close[:-1] * (1 + y_mean_pct_pred)
    
    # 3. XGBoost模型：校准后涨跌幅预测 → 推导价格
    y_xgb_pct_pred = all_y_pred_calibrated[:-1]
    y_xgb_price = all_test_close[:-1] * (1 + y_xgb_pct_pred)
    
    # 计算三个模型的RMSE（价格维度）
    RMSE_naive = math.sqrt(mean_squared_error(y_true_price, y_naive_price))
    RMSE_mean = math.sqrt(mean_squared_error(y_true_price, y_mean_price))  # 调整后的均值模型
    RMSE_xgb = math.sqrt(mean_squared_error(y_true_price, y_xgb_price))
    
    # 验证是否打败基准
    is_beat_naive = RMSE_xgb < RMSE_naive
    is_beat_mean = RMSE_xgb < RMSE_mean
    beat_naive_ratio = (RMSE_naive - RMSE_xgb) / RMSE_naive * 100 if RMSE_naive !=0 else 0
    beat_mean_ratio = (RMSE_mean - RMSE_xgb) / RMSE_mean * 100 if RMSE_mean !=0 else 0
    
    # 输出结果
    print(f'\n✅ 【{ticker}】滚动回测结果')
    print(f'   原始MSE: {pct_mse_original:.6f} → 优化后MSE: {pct_mse_optimized:.6f}（降幅 {(pct_mse_original-pct_mse_optimized)/pct_mse_original*100:.1f}%）')
    print(f'   涨跌幅MSE: {pct_mse_optimized:.6f} | R²: {pct_r2:.4f}')
    print(f"\n📌 【{ticker}】基准对比（滚动回测版）")
    print(f"   傻瓜模型RMSE: {RMSE_naive:.2f} USD")
    print(f"   均值模型RMSE: {RMSE_mean:.2f} USD（过去5天涨跌幅均值预测）")  # 标注调整后的逻辑
    print(f"   XGBoost模型RMSE: {RMSE_xgb:.2f} USD")
    
    if is_beat_naive and is_beat_mean:
        print(f"   ✅ 同时打败两个基准！对比傻瓜模型误差降低 {beat_naive_ratio:.1f}%，对比均值模型误差降低 {beat_mean_ratio:.1f}%")
    elif is_beat_naive and not is_beat_mean:
        print(f"   ⚠️ 仅打败傻瓜模型（误差降低 {beat_naive_ratio:.1f}%），未打败均值模型（误差高 {(RMSE_xgb - RMSE_mean)/RMSE_mean*100:.1f}%）")
    elif not is_beat_naive and is_beat_mean:
        print(f"   ⚠️ 仅打败均值模型（误差降低 {beat_mean_ratio:.1f}%），未打败傻瓜模型（误差高 {(RMSE_xgb - RMSE_naive)/RMSE_naive*100:.1f}%）")
    else:
        print(f"   ❌ 未打败任何基准！对比傻瓜模型误差高 {(RMSE_xgb - RMSE_naive)/RMSE_naive*100:.1f}%，对比均值模型误差高 {(RMSE_xgb - RMSE_mean)/RMSE_mean*100:.1f}%")
    
    # 特征重要性
    feature_importance = pd.DataFrame(
        data=model.feature_importances_,
        index=FEATURES,
        columns=['importance']
    ).sort_values(by='importance', ascending=False)
    print(f'\n📈 【{ticker}】特征重要性排名：')
    print(feature_importance)
    
    # 返回结果（包含均值模型预测数据）
    return {
        'y_test_pct': all_y_test,
        'y_pred_pct_calibrated': all_y_pred_calibrated,
        'y_test_price': all_test_close * (1 + all_y_test),
        'y_pred_price': all_test_close * (1 + all_y_pred_calibrated),
        'test_index': all_test_index,
        'test_close': all_test_close,
        'test_past_5d_pct_mean': all_test_past_5d_pct_mean,  # 涨跌幅均值
        'test_future_1d_pct': all_test_future_1d_pct,
        'pct_mse_original': pct_mse_original,
        'pct_mse_optimized': pct_mse_optimized,
        'pct_r2': pct_r2,
        'RMSE_naive': RMSE_naive,
        'RMSE_mean': RMSE_mean,  # 调整后的均值模型RMSE
        'RMSE_xgb': RMSE_xgb,
        'beat_naive_ratio': beat_naive_ratio,
        'beat_mean_ratio': beat_mean_ratio
    }


# %%
# 运行回测
stock_results = {}
for ticker, single_stock_df in df.groupby('Ticker'):
    stock_results[ticker] = xgb_train(single_stock_df, ticker)
#值不是 “每一行单独当值”，而是这个 Ticker 对应的所有行 + 所有列，打包成一个完整的子 DataFrame（不是零散的行，是整表）
# 可视化AAPL效果（包含新增特征后的预测）
if 'AAPL' in stock_results and stock_results['AAPL'] is not None:
    res = stock_results['AAPL']
    plt.figure(figsize=(16, 12))
    
    # 涨跌幅对比
    plt.subplot(2, 1, 1)
    test_index = pd.to_datetime(res['test_index'])
    plt.plot(test_index, res['y_test_pct'], label='Actual Return (5d Avg)', color='blue', linewidth=1.5)
    plt.plot(test_index, res['y_pred_pct_calibrated'], label='XGBoost Predicted Return (New Features)', color='red', linewidth=1.2)
    plt.plot(test_index, res['test_past_5d_pct_mean'], label='Mean Model Predicted Return', color='purple', linestyle='-.', linewidth=1.2)
    plt.title(f'AAPL Return Prediction Comparison (With Rolling/Momentum Features)', fontsize=14)
    plt.ylabel('Return (5d Average)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 股价对比
    plt.subplot(2, 1, 2)
    test_index_arr = pd.to_datetime(res['test_index'])
    plt.plot(test_index_arr, res['y_test_price'], label='Actual Price', color='blue', linewidth=1.5)
    plt.plot(test_index_arr, res['y_pred_price'], label='XGBoost Predicted Price', color='red', linestyle='--', linewidth=1.2)
    if len(res['test_close']) > 1:
        plt.plot(test_index_arr[1:], res['test_close'][:-1], label='Naive Model Price', color='green', linestyle=':', linewidth=1.2)
    if len(res['test_past_5d_pct_mean']) > 1 and len(res['test_close']) > 1:
        mean_pred_price = res['test_close'][:-1] * (1 + res['test_past_5d_pct_mean'][:-1])
        plt.plot(test_index_arr[1:], mean_pred_price, label='Mean Model Price', color='purple', linestyle='-.', linewidth=1.2)
    
    plt.title(f'AAPL Price Prediction Comparison (RMSE: Naive={res["RMSE_naive"]:.2f}, Mean={res["RMSE_mean"]:.2f}, XGBoost={res["RMSE_xgb"]:.2f})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 汇总所有股票结果
print("\n📊 All Stocks MSE Comparison (With New Features):")
for ticker, res in stock_results.items():
    if res is not None:
        print(f'{ticker}: ')
        print(f'  - 涨跌幅：原始MSE {res["pct_mse_original"]:.6f} → 优化后 {res["pct_mse_optimized"]:.6f} (Reduction: {(res["pct_mse_original"]-res["pct_mse_optimized"])/res["pct_mse_original"]*100:.1f}%)')
        print(f'  - R²: {res["pct_r2"]:.4f}')
        print(f'  - RMSE对比：傻瓜模型 {res["RMSE_naive"]:.2f} USD | 均值模型 {res["RMSE_mean"]:.2f} USD | XGBoost {res["RMSE_xgb"]:.2f} USD')
        print(f'  - 打败基准：傻瓜模型{"✅" if res["RMSE_xgb"] < res["RMSE_naive"] else "❌"} | 均值模型{"✅" if res["RMSE_xgb"] < res["RMSE_mean"] else "❌"}')
        print('-'*80)


