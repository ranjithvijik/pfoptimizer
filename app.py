"""
Portfolio Optimizer
pip install streamlit pandas numpy scipy matplotlib yfinance plotly scikit-learn xlsxwriter openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import optimize
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    layout="wide", 
    page_title="Portfolio Optimizer", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric { 
        background-color: #800020; 
        padding: 10px; 
        border-radius: 5px; 
        border: 1px solid #dee2e6; 
    }
    div.block-container { 
        padding-top: 2rem; 
    }
    .stAlert {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 1. Caching & Data Layer
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers, start, end, interval):
    """
    Cached data fetcher to prevent API rate limits.
    Returns raw yfinance DataFrame.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    try:
        download_list = list(set(tickers + ['SPY']))
        data = yf.download(
            download_list, 
            start=start, 
            end=end, 
            interval=interval, 
            group_by='ticker', 
            auto_adjust=True,
            progress=False,
            threads=True
        )
        return data
    except Exception as e:
        return None

# --------------------------
# 2. Enhanced Math & Statistics Library
# --------------------------
TRADING_DAYS = 252

def annualized_mean(rets, freq=TRADING_DAYS): 
    return rets.mean() * freq

def annualized_vol(rets, freq=TRADING_DAYS): 
    return rets.std(ddof=1) * np.sqrt(freq)

def sharpe_ratio(rets, rf=0.0, freq=TRADING_DAYS): 
    vol = annualized_vol(rets, freq)
    # rf is assumed to be annual here
    return (annualized_mean(rets, freq) - rf) / vol if vol > 0 else 0

def downside_deviation(rets, mar=0.0, freq=TRADING_DAYS):
    neg = rets[rets <= mar]
    if len(neg) == 0: 
        return 0.0
    return np.sqrt((neg**2).mean()) * np.sqrt(freq)

def sortino_ratio(rets, rf=0.0, freq=TRADING_DAYS):
    dd = downside_deviation(rets, rf/freq, freq)
    return (annualized_mean(rets, freq) - rf) / dd if dd > 0 else 0

def max_drawdown(rets):
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def calmar_ratio(rets, freq=TRADING_DAYS):
    dd = max_drawdown(rets)
    if dd == 0: 
        return np.nan
    return annualized_mean(rets, freq) / abs(dd)

def omega_ratio(rets, threshold=0.0):
    excess = rets - threshold
    pos = excess[excess > 0].sum()
    neg = -excess[excess < 0].sum()
    return pos / neg if neg != 0 else np.nan

def capture_ratios(bench_rets, asset_rets):
    """Calculate upside and downside capture ratios"""
    if bench_rets.empty or asset_rets.empty:
        return np.nan, np.nan
    
    df = pd.concat([bench_rets, asset_rets], axis=1).dropna()
    if df.empty: 
        return np.nan, np.nan
    
    bm = df.iloc[:, 0]
    at = df.iloc[:, 1]
    
    up = bm > 0
    down = bm < 0
    
    def geom(r): 
        if len(r) == 0: 
            return 0
        return (1+r).prod() - 1
    
    uc = geom(at[up]) / geom(bm[up]) if geom(bm[up]) != 0 else np.nan
    dc = geom(at[down]) / geom(bm[down]) if geom(bm[down]) != 0 else np.nan
    return uc, dc

def var_cvar(rets, alpha=0.05):
    """Calculate Value at Risk and Conditional Value at Risk"""
    if len(rets) == 0: 
        return 0, 0
    var = np.percentile(rets, alpha * 100)
    cvar = rets[rets <= var].mean()
    return var, cvar

def information_ratio(rets, bench_rets, freq=TRADING_DAYS):
    """Calculate Information Ratio"""
    active_rets = rets - bench_rets
    tracking_error = active_rets.std() * np.sqrt(freq)
    return (active_rets.mean() * freq) / tracking_error if tracking_error > 0 else 0

# --- Advanced Estimation Methods ---
def james_stein_shrinkage(returns):
    """James-Stein shrinkage estimator for expected returns"""
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean(axis=0).values
    grand_mean = np.mean(mu)
    S = returns.cov().values
    
    try:
        inv_S = np.linalg.pinv(S)
        lambda_js = min(1.0, (n - 2) / (T * (mu - grand_mean).T @ inv_S @ (mu - grand_mean)))
    except:
        lambda_js = 0.5
    
    return (1 - lambda_js) * mu + lambda_js * grand_mean

def ledoit_wolf_cov(returns):
    """Ledoit-Wolf shrinkage estimator for covariance matrix"""
    return LedoitWolf().fit(returns).covariance_

def black_litterman(market_weights, cov, views_dict, tau=0.05):
    """
    Black-Litterman model for expected returns
    """
    pi = tau * cov @ market_weights  # Equilibrium returns
    
    if not views_dict:
        return pi
    
    # Build P and Q matrices from views
    n = len(market_weights)
    P = np.zeros((len(views_dict), n))
    Q = np.array(list(views_dict.values()))
    
    for i, (ticker, _) in enumerate(views_dict.items()):
        if ticker in market_weights.index:
            P[i, market_weights.index.get_loc(ticker)] = 1
    
    # Black-Litterman formula
    omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
    try:
        tau_cov_inv = np.linalg.inv(tau * cov)
        omega_inv = np.linalg.inv(omega)
        M = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
        mu_bl = M @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
    except:
        mu_bl = pi # Fallback if singular
        
    return mu_bl

# --------------------------
# 3. Enhanced Unified Analyzer Class
# --------------------------
class EnhancedUnifiedAnalyzer:
    def __init__(self):
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.benchmark_returns = pd.Series()
        self.metrics = pd.DataFrame()
        self.snapshot = pd.DataFrame()
        self.rolling = {}
        self.tickers = []
        self.weights = None
        self.backtest_res = None
        self.freq_scaler = 252
        self.data_quality_issues = []
        self.monte_carlo_results = None
        self.regime = None
        self.attribution = None
        
    def fetch_data(self, tickers, start, end, interval='1d'):
        """Fetch and validate market data"""
        self.tickers = tickers
        
        # Adjust scaler based on interval
        if interval == '1wk': 
            self.freq_scaler = 52
        elif interval == '1mo': 
            self.freq_scaler = 12
        else: 
            self.freq_scaler = 252
            
        download_list = list(set(tickers + ['SPY']))
        
        raw_data = get_stock_data(download_list, start, end, interval)
        
        if raw_data is None or raw_data.empty:
            st.error("No data returned. Please check tickers.")
            return False

        # --- Robust Flattening Logic ---
        df_close = pd.DataFrame()
        failed_tickers = []
        successful_tickers = []
        # Handle single ticker vs multi ticker structure
        if len(tickers) == 1:
            ticker = tickers
            # Try finding the ticker in columns if multi-index
            if isinstance(raw_data.columns, pd.MultiIndex):
                try:
                    df_close[ticker] = raw_data[ticker]['Close']
                    successful_tickers.append(ticker)
                except KeyError:
                    # Sometimes single ticker is flat
                    if 'Close' in raw_data.columns:
                        df_close[ticker] = raw_data['Close']
                        successful_tickers.append(ticker)
                    else:
                        failed_tickers.append(f"{ticker} (Close column not found)")
            else:
                if 'Close' in raw_data.columns:
                    df_close[ticker] = raw_data['Close']
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(f"{ticker} (Close column not found)")
        else:
            # Multi-ticker extraction with comprehensive error handling
            for t in tickers:
                try:
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        # MultiIndex case: raw_data[ticker]['Close']
                        if t in raw_data.columns.get_level_values(0):
                            df_close[t] = raw_data[t]['Close']
                            successful_tickers.append(t)
                        else:
                            failed_tickers.append(f"{t} (not in MultiIndex)")
                    elif t in raw_data.columns:
                        # Single level columns: direct access
                        df_close[t] = raw_data[t]
                        successful_tickers.append(t)
                    else:
                        failed_tickers.append(f"{t} (not found in data)")
        
                except KeyError as e:
                    failed_tickers.append(f"{t} (KeyError: {str(e)})")
                except Exception as e:
                    failed_tickers.append(f"{t} ({type(e).__name__}: {str(e)})")

        if failed_tickers and successful_tickers:
            # Mixed success/failure
            st.warning(
                f"⚠️ Partial data extraction:\n"
                f"✅ Successfully extracted: {', '.join(successful_tickers)}\n"
                f"❌ Could not extract: {', '.join(failed_tickers)}"
            )
        elif failed_tickers and not successful_tickers:
            # Complete failure
            st.error(
                f"❌ Could not extract Close prices for any tickers.\n"
                f"Failed: {', '.join(failed_tickers)}\n"
                f"Please verify ticker symbols are correct."
            )
            return False
        
        if df_close.empty:
            st.error("Could not parse 'Close' prices from data.")
            return False

        # Data Quality Checks
        self.data_quality_issues = self._validate_data_quality(df_close)
        
        # Data Cleaning
        df_close = df_close.ffill()
        # Drop columns with > 30% missing
        df_close = df_close.dropna(thresh=int(0.7*len(df_close)), axis=1) 
        df_close = df_close.dropna(axis=0) 
        
        all_returns = df_close.pct_change().dropna()
        
        if all_returns.empty:
            st.error("Not enough data to calculate returns.")
            return False

        # Extract SPY for calculations
        if 'SPY' in all_returns.columns:
            self.benchmark_returns = all_returns['SPY']
            self.has_benchmark = True
        else:
            self.benchmark_returns = pd.Series(dtype=float)
            self.has_benchmark = False
        valid_tickers = [t for t in tickers if t in df_close.columns]
        
        self.prices = df_close[valid_tickers]
        metrics_tickers = valid_tickers.copy()
        if 'SPY' in all_returns.columns:
            metrics_tickers.append('SPY')
        self.returns = all_returns[metrics_tickers]

        # Computations
        self._compute_metrics()
        self._compute_snapshot()
        self._compute_rolling()
        self._detect_market_regime()
        
        return True
    
    def _validate_data_quality(self, prices):
        """Check for data quality issues"""
        issues = []
        
        # Check for missing data
        missing_pct = prices.isnull().sum() / len(prices)
        high_missing = missing_pct[missing_pct > 0.1]
        if not high_missing.empty:
            issues.append(f"High missing data: {high_missing.to_dict()}")
        
        # Check for stale prices (unchanged for 5+ days)
        for col in prices.columns:
            if len(prices[col]) > 5:
                unchanged = (prices[col].diff() == 0).rolling(5).sum() == 5
                if unchanged.any():
                    # Just note the ticker
                    issues.append(f"{col}: Stale prices detected")
        
        return issues
    
    def _compute_metrics(self):
        """Compute comprehensive metrics for each asset"""
        bench_rets = self.returns.get('SPY', self.returns.iloc[:, 0])
        
        met_list = []
        for t in self.returns.columns:
            r = self.returns[t]
            var95, cvar95 = var_cvar(r)
            uc, dc = capture_ratios(bench_rets, r)
            
            met_list.append({
                "Ticker": t,
                "Ann. Return": annualized_mean(r, self.freq_scaler),
                "Volatility": annualized_vol(r, self.freq_scaler),
                "Sharpe": sharpe_ratio(r, 0, self.freq_scaler),
                "Sortino": sortino_ratio(r, 0, self.freq_scaler),
                "Calmar": calmar_ratio(r, self.freq_scaler),
                "Omega": omega_ratio(r),
                "Max DD": max_drawdown(r),
                "Downside Dev": downside_deviation(r, 0, self.freq_scaler),
                "VaR (95%)": var95,
                "CVaR (95%)": cvar95,
                "Upside Cap": uc,
                "Downside Cap": dc,
                "Info Ratio": information_ratio(r, bench_rets, self.freq_scaler) if t != 'SPY' else np.nan
            })
        self.metrics = pd.DataFrame(met_list).set_index("Ticker")
    
    def _compute_snapshot(self):
        """Compute period returns snapshot"""
        snap = []
        for t in self.prices.columns:
            p = self.prices[t]
            last = p.iloc[-1]
            
            def get_ret(days_back):
                if len(p) > days_back: 
                    return p.iloc[-1]/p.iloc[-days_back-1] - 1
                return np.nan
            
            r1m = get_ret(21)
            r3m = get_ret(63)
            r1y = get_ret(252)
            
            # YTD
            curr_year = p.index[-1].year
            ytd_data = p[p.index.year == curr_year]
            ytd = (ytd_data.iloc[-1] / ytd_data.iloc[0] - 1) if not ytd_data.empty else np.nan
            
            snap.append([
                t, last, r1m, r3m, ytd, r1y, 
                annualized_vol(self.returns[t], self.freq_scaler), 
                max_drawdown(self.returns[t])
            ])
            
        self.snapshot = pd.DataFrame(
            snap, 
            columns=['Ticker','Last Price','1M','3M','YTD','1Y','Volatility','MaxDD']
        ).set_index('Ticker')
    
    def _compute_rolling(self, windows=[21, 63, 126]):
        """Compute rolling statistics"""
        self.rolling = {}
        for t in self.returns.columns:
            r = self.returns[t]
            for w in windows:
                if len(r) > w:
                    vol = r.rolling(w).std() * np.sqrt(self.freq_scaler)
                    sharpe = (r.rolling(w).mean() * self.freq_scaler) / vol
                    self.rolling[f"{t}_vol_{w}"] = vol
                    self.rolling[f"{t}_sharpe_{w}"] = sharpe
    
    def _detect_market_regime(self, lookback=60):
        """Detect market regimes using simple rules"""
        if 'SPY' in self.returns.columns:
            spy_returns = self.returns['SPY']
        else:
            spy_returns = self.returns.mean(axis=1)
        
        if len(spy_returns) < lookback:
            self.regime = pd.Series('Unknown', index=spy_returns.index)
            return
        
        # Calculate indicators
        sma_short = spy_returns.rolling(20, min_periods=1).mean()
        sma_long = spy_returns.rolling(60, min_periods=1).mean()
        volatility = spy_returns.rolling(20, min_periods=1).std()
        
        # Classify regime
        regime = pd.Series('Normal', index=spy_returns.index)
        conditions = [
            volatility > volatility.quantile(0.8),
            (sma_short > sma_long) & (volatility <= volatility.quantile(0.8)),
            (sma_short <= sma_long) & (volatility <= volatility.quantile(0.8))
        ]
        choices = ['High Vol', 'Bull', 'Bear']
        regime_array = np.select(conditions, choices, default='Normal')
        self.regime = pd.Series(regime_array, index=spy_returns.index)
    
    def optimize_portfolio(self, method="Sharpe", risk_model="Sample", 
                           ret_model="Mean", bounds=(0,1), rf=0.04, views_dict=None):
        """Optimize portfolio with various methods"""
        
        # Prepare expected returns
        if ret_model == "Mean":
            mu = self.returns.mean().values
        elif ret_model == "James-Stein":
            mu = james_stein_shrinkage(self.returns)
        elif ret_model == "Black-Litterman" and views_dict:
            market_weights = pd.Series(1/len(self.returns.columns), index=self.returns.columns)
            cov = self.returns.cov().values
            mu = black_litterman(market_weights, cov, views_dict)
        else:
            mu = self.returns.mean().values
        
        # Prepare covariance matrix
        if risk_model == "Ledoit-Wolf":
            cov = ledoit_wolf_cov(self.returns)
        else:
            cov = self.returns.cov().values
        
        n = len(mu)
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bnds = tuple([bounds for _ in range(n)])
        
        # Optimization objectives
        if method == "Sharpe":
            def obj(w):
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                p_ret = np.dot(w, mu)
                # Note: mu is daily mean, cov is daily cov. 
                # rf (annual) / scaler = daily rf.
                # Optimization is done on daily terms, which aligns with maximizing Annual Sharpe.
                daily_rf = rf / self.freq_scaler
                return -((p_ret - daily_rf) / p_vol) if p_vol > 0 else 0
                
        elif method == "Min CVaR":
            def obj(w):
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                p_ret = np.dot(w, mu)
                return -p_ret + 2.0 * p_vol
                
        elif method == "Risk Parity":
            def risk_contrib(w):
                port_vol = np.sqrt(w @ cov @ w)
                marginal_contrib = cov @ w
                contrib = w * marginal_contrib / port_vol
                return contrib
            
            def obj(w):
                rc = risk_contrib(w)
                target = np.ones(n) / n
                return np.sum((rc - target)**2)
                
        elif method == "Max Diversification":
            def obj(w):
                weighted_vols = w @ np.sqrt(np.diag(cov))
                port_vol = np.sqrt(w @ cov @ w)
                return -weighted_vols / port_vol if port_vol > 0 else 0
        
        # Run optimization
        x0 = np.array([1/n]*n)
        try:
            res = optimize.minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons)
            optimal_weights = res.x
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            optimal_weights = x0
        
        self.weights = pd.DataFrame({
            "Ticker": self.returns.columns, 
            "Weight": optimal_weights
        })
        self.weights = self.weights[self.weights['Weight'] > 0.001]
        
        # Calculate performance attribution
        self._calculate_attribution()
        
        return self.weights
    
    def _calculate_attribution(self):
        """Calculate performance attribution for the portfolio"""
        if self.weights is None:
            return
        
        weights = self.weights.set_index('Ticker')['Weight']
        
        # Calculate contributions
        contributions = pd.DataFrame()
        for ticker in weights.index:
            if ticker in self.returns.columns:
                contributions[ticker] = self.returns[ticker] * weights[ticker]
        
        # Create attribution summary
        self.attribution = pd.DataFrame({
            'Weight': weights,
            'Return': self.metrics.loc[weights.index, 'Ann. Return'],
            'Risk': self.metrics.loc[weights.index, 'Volatility'],
            'Return Contribution': contributions.sum() * self.freq_scaler,
            'Risk Contribution': weights * self.metrics.loc[weights.index, 'Volatility']
        })
    
    def run_enhanced_backtest(self, window=252, rebalance=21, cost_bps=10):
        """Enhanced backtest with tracking metrics"""
        n = len(self.returns)
        vals = [100.0]
        dates = []
        turnovers = []
        weights_history = []
        curr_w = np.array([1/len(self.tickers)]*len(self.tickers))
        
        for i in range(window, n, rebalance):
            train = self.returns.iloc[i-window:i]
            if train.empty: 
                continue
            
            mu = train.mean().values
            cov = train.cov().values
            
            # Max Sharpe optimization
            def neg_s(w):
                v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                return -(np.dot(w, mu)/v) if v>0 else 0
            
            cons = ({'type':'eq','fun':lambda x: np.sum(x)-1})
            bnds = tuple([(0,1) for _ in range(len(mu))])
            
            try:
                res = optimize.minimize(
                    neg_s, [1/len(mu)]*len(mu), 
                    method='SLSQP', bounds=bnds, constraints=cons
                )
                new_w = res.x
            except:
                new_w = curr_w
            
            new_w = pd.Series(new_w, index=tickers) if not isinstance(new_w, pd.Series) else new_w
            curr_w = pd.Series(curr_w, index=tickers) if not isinstance(curr_w, pd.Series) else curr_w
            new_w = new_w.reindex(tickers, fill_value=0)
            curr_w = curr_w.reindex(tickers, fill_value=0)
            turnover = np.sum(np.abs(new_w.values - curr_w.values))
            cost = turnover * (cost_bps/10000)
            
            # Apply returns
            hold_end = min(i+rebalance, n)
            per_ret = self.returns.iloc[i:hold_end]
            # Assumes rebalance happens at closing prices
            cum_per = (1 + per_ret @ new_w).prod()
            
            # Explicitly cast to float to prevent Series accumulation
            strategy_return = float(cum_per - 1)
            net_return = strategy_return - cost
            vals.append(vals[-1] * (1 + net_return))
            dates.append(self.returns.index[i])
            turnovers.append(turnover)
            weights_history.append(new_w)
            curr_w = new_w
        
        if not dates:
            self.backtest_res = None
            self.backtest_metrics = {}
            return None

        self.backtest_res = pd.DataFrame({"Value": vals[1:]}, index=dates)
        
        # Calculate enhanced metrics
        strategy_returns = self.backtest_res.pct_change().dropna()
        
        if not self.benchmark_returns.empty:
            benchmark_aligned = self.benchmark_returns.reindex(strategy_returns.index).fillna(0)
            
            active_returns = strategy_returns['Value'] - benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            info_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
        else:
            # Use equal-weight portfolio as fallback
            asset_returns = self.returns.iloc[:, :-1]
            equal_weight_returns = asset_returns.mean(axis=1)
            equal_weight_port = (1 + equal_weight_returns).cumprod() * 100
            equal_weight_bench_rets = equal_weight_port.pct_change().dropna()
    
            benchmark_aligned = equal_weight_bench_rets.reindex(strategy_returns.index).fillna(0)
            active_returns = strategy_returns['Value'] - benchmark_aligned
    
            if len(active_returns) > 0 and active_returns.std() > 0:
                tracking_error = active_returns.std() * np.sqrt(252)
                info_ratio = active_returns.mean() / tracking_error
            else:
                tracking_error = 0.0
                info_ratio = 0.0
        
        # FIX: Explicit float conversion for all metrics
        self.backtest_metrics = {
            'total_return': float((vals[-1] / 100 - 1)) if vals else 0.0,
            'sharpe_ratio': float(sharpe_ratio(strategy_returns['Value'])) if len(strategy_returns) > 0 else 0.0,
            'max_drawdown': float(max_drawdown(strategy_returns['Value'])) if len(strategy_returns) > 0 else 0.0,
            'avg_turnover': float(np.mean(turnovers)) if turnovers else 0.0,
            'tracking_error': float(tracking_error),
            'information_ratio': float(info_ratio),
            'weights_history': weights_history
        }
        
        return self.backtest_res
    
    def monte_carlo_simulation(self, n_simulations=1000, n_days=252, distribution='normal'):
        """Run Monte Carlo simulation for portfolio outcomes"""
        if self.weights is None:
            return None
        
        # Map weights to full ticker list (handle missing tickers in optimized weights)
        optimized_weights = self.weights.set_index('Ticker')['Weight']
        full_weights = np.array([optimized_weights.get(t, 0.0) for t in self.returns.columns])
        
        mu = self.returns.mean().values
        cov = self.returns.cov().values
        
        # Annual parameters for display/check
        annual_mu = mu * self.freq_scaler
        annual_cov = cov * self.freq_scaler
        
        # Portfolio parameters (Daily)
        port_mu = full_weights @ mu
        port_vol = np.sqrt(full_weights @ cov @ full_weights.T)
        
        # Run simulations
        simulations = np.zeros((n_simulations, n_days))
        
        for i in range(n_simulations):
            if distribution == 'normal':
                daily_returns = np.random.normal(port_mu, port_vol, n_days)
            elif distribution == 'tstudent':
                from scipy.stats import t
                daily_returns = t.rvs(df=4, loc=port_mu, scale=port_vol, size=n_days)
            elif distribution == 'empirical':
                # Calculate historical portfolio returns using current weights
                historical_port_rets = (self.returns * full_weights).sum(axis=1)
    
                # Sample with replacement from historical returns
                daily_returns = np.random.choice(
                    historical_port_rets.values, 
                    size=n_days, 
                    replace=True
                )
            simulations[i] = (1 + daily_returns).cumprod() * 100
        
        # Calculate statistics
        percentiles = np.percentile(simulations, [5, 25, 50, 75, 95], axis=0)
        
        self.monte_carlo_results = {
            'simulations': simulations,
            'percentiles': percentiles,
            'expected_return': port_mu * self.freq_scaler,
            'expected_volatility': port_vol * np.sqrt(self.freq_scaler),
            'var_95': np.percentile(simulations[:, -1], 5) - 100,
            'cvar_95': simulations[simulations[:, -1] < np.percentile(simulations[:, -1], 5), -1].mean() - 100
        }
        
        return self.monte_carlo_results
    
    def get_excel(self):
        """Generate comprehensive Excel report"""
        output = BytesIO()
        
        # Fallback engine
        try:
            import xlsxwriter
            engine = 'xlsxwriter'
        except ImportError:
            engine = 'openpyxl'

        with pd.ExcelWriter(output, engine=engine) as writer:
            # Basic data
            self.prices.to_excel(writer, sheet_name='Prices')
            self.returns.to_excel(writer, sheet_name='Returns')
            self.snapshot.to_excel(writer, sheet_name='Snapshot')
            self.metrics.to_excel(writer, sheet_name='Metrics')
            self.returns.corr().to_excel(writer, sheet_name='Correlation')
            
            # Rolling statistics
            if self.rolling:
                roll_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in self.rolling.items()]))
                roll_df.to_excel(writer, sheet_name='Rolling Stats')
            
            # Optimization results
            if self.weights is not None:
                self.weights.to_excel(writer, sheet_name='Optimal Weights', index=False)
            
            if self.attribution is not None:
                self.attribution.to_excel(writer, sheet_name='Attribution')
            
            # Backtest results
            if self.backtest_res is not None:
                self.backtest_res.to_excel(writer, sheet_name='Backtest')
                
                # Backtest metrics
                metrics_df = pd.DataFrame([self.backtest_metrics])
                metrics_df.to_excel(writer, sheet_name='Backtest Metrics')
            
            # Data quality report
            if self.data_quality_issues:
                quality_df = pd.DataFrame({'Issues': self.data_quality_issues})
                quality_df.to_excel(writer, sheet_name='Data Quality', index=False)
            
            # Market regime
            if self.regime is not None:
                regime_df = pd.DataFrame({'Date': self.regime.index, 'Regime': self.regime.values})
                regime_df.to_excel(writer, sheet_name='Market Regime', index=False)
                
        return output.getvalue()

# --------------------------
# 4. Streamlit UI
# --------------------------
st.title("Portfolio Optimizer")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Parameters
    with st.expander("📈 Data Parameters", expanded=True):
        tickers_txt = st.text_area(
            "Tickers (comma-separated)", 
            "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
            help="Enter stock tickers separated by commas. SPY is used as benchmark."
        )
        import re
        ticker_pattern = re.compile(r'^[A-Z]{1,5}$')
        raw_tickers = [t.strip().upper() for t in tickers_txt.replace("\n",",").split(",")]
        valid_tickers = [t for t in raw_tickers if ticker_pattern.match(t)]
        if len(valid_tickers) < len(raw_tickers):
            st.warning(f"Invalid tickers removed")
        tickers = valid_tickers
    
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                datetime.now() - timedelta(days=3*365),
                help="Historical data start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                datetime.today(),
                help="Historical data end date"
            )
    
        # Validate date range
        if start_date >= end_date:
            st.error("❌ Start date must be before end date!")
            st.stop()
    
        # Calculate days difference
        days_diff = (end_date - start_date).days
    
        # Warn if data period is too short
        if days_diff < 30:
            st.warning(
                f"⚠️ Only {days_diff} days of data."
                f"Recommend at least 60 days for reliable statistical analysis."
            )
        
        interval = st.selectbox(
            "Data Frequency", 
            ['1d', '1wk', '1mo'],
            help="Daily data recommended for backtesting"
        )
    
    # Optimization Settings
    with st.expander("🎯 Optimization Settings", expanded=True):
        opt_method = st.selectbox(
            "Optimization Method", 
            ["Sharpe", "Min CVaR", "Risk Parity", "Max Diversification"],
            help="Choose portfolio optimization objective"
        )
        
        risk_model = st.selectbox(
            "Risk Model", 
            ["Ledoit-Wolf", "Sample"],
            help="Ledoit-Wolf recommended for small samples"
        )
        
        ret_model = st.selectbox(
            "Return Model", 
            ["James-Stein", "Mean", "Black-Litterman"],
            help="James-Stein reduces estimation error"
        )
        
        allow_short = st.checkbox("Allow Short Selling", False)
        
        rf_rate = st.slider(
            "Risk-Free Rate (%)", 
            0.0, 10.0, 4.0, 0.1,
            help="Annual risk-free rate for Sharpe ratio"
        ) / 100
    
    # Advanced Options
    with st.expander("🔬 Advanced Options"):
        # Black-Litterman Views
        use_views = st.checkbox("Use Black-Litterman Views", False)
        views_dict = {}
        if use_views and ret_model == "Black-Litterman":
            st.markdown("**Market Views** (Expected Annual Returns)")
            for ticker in tickers[:5]:  # Limit to first 5 for UI
                view = st.number_input(
                    f"{ticker} (%)", 
                    -50.0, 100.0, 10.0, 1.0,
                    key=f"view_{ticker}"
                ) / 100
                views_dict[ticker] = view
        
        # Monte Carlo
        run_monte_carlo = st.checkbox("Run Monte Carlo Simulation", True)
        if run_monte_carlo:
            n_simulations = st.slider(
                "Number of Simulations", 
                100, 10000, 2000, 500
            )
            mc_days = st.slider(
                "Simulation Days", 
                30, 500, 252, 10
            )
        
        # Backtest Settings
        run_backtest = st.checkbox("Run Backtest", True)
        if run_backtest:
            backtest_window = st.slider(
                "Lookback Window (days)", 
                60, 500, 252, 10
            )
            rebalance_freq = st.slider(
                "Rebalance Frequency (days)", 
                5, 60, 21, 1
            )
            transaction_cost = st.slider(
                "Transaction Cost (bps)", 
                0, 50, 10, 1
            )
    
    # Run Analysis Button
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", width="stretch")

# --- Main Analysis ---
if run_btn:
    analyzer = EnhancedUnifiedAnalyzer()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch Data
    status_text.text("Fetching market data...")
    progress_bar.progress(10)
    
    success = analyzer.fetch_data(tickers, start_date, end_date, interval)
    
    if success:
        # Data Quality Alert
        if analyzer.data_quality_issues:
            with st.expander("⚠️ Data Quality Issues Detected", expanded=False):
                for issue in analyzer.data_quality_issues:
                    st.warning(issue)
        
        # Run Optimization
        status_text.text("Optimizing portfolio...")
        progress_bar.progress(30)
        
        analyzer.optimize_portfolio(
            method=opt_method,
            risk_model=risk_model,
            ret_model=ret_model,
            bounds=(-1,1) if allow_short else (0,1),
            rf=rf_rate,
            views_dict=views_dict if use_views else None
        )
        
        # Run Backtest
        if run_backtest and interval == '1d':
            status_text.text("Running backtest...")
            progress_bar.progress(50)
            analyzer.run_enhanced_backtest(
                window=backtest_window,
                rebalance=rebalance_freq,
                cost_bps=transaction_cost
            )
        
        # Run Monte Carlo
        if run_monte_carlo and analyzer.weights is not None:
            status_text.text("Running Monte Carlo simulation...")
            progress_bar.progress(70)
            analyzer.monte_carlo_simulation(n_simulations, mc_days)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # --- Create Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Analysis", 
            "⚖️ Portfolio Optimization", 
            "🕰️ Backtest Results",
            "🎲 Monte Carlo",
            "📊 Advanced Analytics"
        ])
        
        # TAB 1: Market Analysis
        with tab1:
            st.header("Market Analysis Dashboard")
            
            # Performance Snapshot
            st.subheader("📊 Performance Snapshot")
            
            # Format snapshot table
            snapshot_styled = analyzer.snapshot.style.format({
                'Last Price': '${:.2f}',
                '1M': '{:.2%}', 
                '3M': '{:.2%}', 
                'YTD': '{:.2%}', 
                '1Y': '{:.2%}', 
                'Volatility': '{:.2%}', 
                'MaxDD': '{:.2%}'
            }).background_gradient(subset=['1M', '3M', 'YTD', '1Y'], cmap='RdYlGn', vmin=-0.2, vmax=0.2)
            
            st.dataframe(snapshot_styled, width="stretch")
            
            # Price Performance Chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Normalized Price Performance")
                norm_prices = (analyzer.prices / analyzer.prices.iloc[0]) * 100
                fig_norm = px.line(
                    norm_prices, 
                    title="Asset Performance (Rebased to 100)",
                    labels={'value': 'Normalized Price', 'index': 'Date'}
                )
                fig_norm.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_norm, width="stretch")
            
            with col2:
                st.subheader("🎯 Risk-Return Profile")
                met = analyzer.metrics.reset_index()
                fig_scatter = px.scatter(
                    met, 
                    x="Volatility", 
                    y="Ann. Return",
                    text="Ticker",
                    color="Sharpe",
                    size=[20]*len(met),
                    color_continuous_scale='RdYlGn',
                    title="Risk-Return Scatter"
                )
                fig_scatter.update_traces(textposition='top center')
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, width="stretch")
            
            # Correlation Matrix and Metrics
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = analyzer.returns.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Asset Correlations"
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, width="stretch")
            
            with col4:
                st.subheader("📊 Market Regime")
                if analyzer.regime is not None:
                    regime_counts = analyzer.regime.value_counts()
                    fig_regime = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title=f"Market Regime Distribution",
                        color_discrete_map={
                            'Bull': '#2ca02c',
                            'Bear': '#d62728',
                            'High Vol': '#ff7f0e',
                            'Normal': '#1f77b4'
                        }
                    )
                    fig_regime.update_layout(height=400)
                    st.plotly_chart(fig_regime, width="stretch")
            
            # Detailed Metrics Table
            st.subheader("📋 Detailed Performance Metrics")
            metrics_styled = analyzer.metrics.style.format({
                'Ann. Return': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe': '{:.2f}',
                'Sortino': '{:.2f}',
                'Calmar': '{:.2f}',
                'Omega': '{:.2f}',
                'Max DD': '{:.2%}',
                'Downside Dev': '{:.2%}',
                'VaR (95%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}',
                'Upside Cap': '{:.2f}',
                'Downside Cap': '{:.2f}',
                'Info Ratio': '{:.2f}'
            }).background_gradient(subset=['Sharpe', 'Sortino'], cmap='RdYlGn')
            
            st.dataframe(metrics_styled, width="stretch")
            
            # Rolling Statistics
            if analyzer.rolling:
                st.subheader("📈 Rolling Statistics")
                
                # Select metric to display
                col5, col6 = st.columns(2)
                with col5:
                    roll_metric = st.selectbox(
                        "Select Metric",
                        ["Volatility", "Sharpe Ratio"]
                    )
                with col6:
                    roll_window = st.selectbox(
                        "Window Size",
                        [21, 63, 126],
                        format_func=lambda x: f"{x} days"
                    )
                
                # Create rolling chart
                if roll_metric == "Volatility":
                    roll_cols = [c for c in analyzer.rolling.keys() if f"vol_{roll_window}" in c]
                else:
                    roll_cols = [c for c in analyzer.rolling.keys() if f"sharpe_{roll_window}" in c]
                
                if roll_cols:
                    roll_df = pd.DataFrame({
                        k.split('_')[0]: analyzer.rolling[k] 
                        for k in roll_cols
                    }, index=analyzer.returns.index)
                    
                    fig_roll = px.line(
                        roll_df,
                        title=f"Rolling {roll_metric} ({roll_window}-day window)"
                    )
                    fig_roll.update_layout(height=400)
                    st.plotly_chart(fig_roll, width="stretch")
        
        # TAB 2: Portfolio Optimization
        with tab2:
            st.header("Portfolio Optimization Results")
            
            if analyzer.weights is not None:
                # Portfolio composition
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📊 Optimal Weights")
                    weights_display = analyzer.weights.copy()
                    weights_display['Weight'] = weights_display['Weight'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(weights_display, width="stretch")
                    
                    # Portfolio metrics
                    if analyzer.attribution is not None:
                        port_return = (analyzer.attribution['Return'] * analyzer.attribution['Weight']).sum()
                        port_risk = np.sqrt(
                            analyzer.weights['Weight'].values @ 
                            analyzer.returns[analyzer.weights['Ticker']].cov().values @ 
                            analyzer.weights['Weight'].values
                        ) * np.sqrt(analyzer.freq_scaler)
                        port_sharpe = (port_return - rf_rate) / port_risk if port_risk > 0 else 0
                        
                        st.metric("Expected Return", f"{port_return:.2%}")
                        st.metric("Portfolio Risk", f"{port_risk:.2%}")
                        st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
                
                with col2:
                    st.subheader("🥧 Portfolio Allocation")
                    fig_pie = px.pie(
                        analyzer.weights,
                        values='Weight',
                        names='Ticker',
                        title=f"Optimal Portfolio ({opt_method})",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, width="stretch")
                
                # Performance Attribution
                if analyzer.attribution is not None:
                    st.subheader("📊 Performance Attribution")
                    
                    # Create attribution chart
                    fig_attr = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Return Contribution", "Risk Contribution")
                    )
                    
                    # Return contribution
                    fig_attr.add_trace(
                        go.Bar(
                            x=analyzer.attribution.index,
                            y=analyzer.attribution['Return Contribution'],
                            name='Return',
                            marker_color='green'
                        ),
                        row=1, col=1
                    )
                    
                    # Risk contribution
                    fig_attr.add_trace(
                        go.Bar(
                            x=analyzer.attribution.index,
                            y=analyzer.attribution['Risk Contribution'],
                            name='Risk',
                            marker_color='red'
                        ),
                        row=1, col=2
                    )
                    
                    fig_attr.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_attr, width="stretch")
                    
                    # Attribution table
                    st.subheader("📋 Detailed Attribution")
                    attr_styled = analyzer.attribution.style.format({
                        'Weight': '{:.2%}',
                        'Return': '{:.2%}',
                        'Risk': '{:.2%}',
                        'Return Contribution': '{:.2%}',
                        'Risk Contribution': '{:.2%}'
                    })
                    st.dataframe(attr_styled, width="stretch")
        
        # TAB 3: Backtest Results
        with tab3:
            st.header("Backtest Analysis")
            
            if analyzer.backtest_res is not None:
                # Backtest metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Return",
                        f"{float(analyzer.backtest_metrics.get('total_return', 0)):.2%}"
                    )
                with col2:
                    st.metric(
                        "Sharpe Ratio",
                        f"{float(analyzer.backtest_metrics.get('sharpe_ratio', 0)):.2f}"
                    )
                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"{float(analyzer.backtest_metrics.get('max_drawdown', 0)):.2%}"
                    )
                with col4:
                    st.metric(
                        "Avg Turnover",
                        f"{float(analyzer.backtest_metrics.get('avg_turnover', 0)):.2%}"
                    )
                
                # Equity curve
                st.subheader("📈 Strategy Performance")
                
                fig_backtest = go.Figure()
                
                # Add strategy line
                fig_backtest.add_trace(go.Scatter(
                    x=analyzer.backtest_res.index,
                    y=analyzer.backtest_res['Value'],
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ))
                
                # Add benchmark if available
                if 'SPY' in analyzer.prices.columns:
                    spy_prices = analyzer.prices['SPY'].loc[analyzer.backtest_res.index[0]:]
                    spy_norm = (spy_prices / spy_prices.iloc[0]) * 100
                    spy_aligned = spy_norm.loc[analyzer.backtest_res.index]
                    
                    fig_backtest.add_trace(go.Scatter(
                        x=spy_aligned.index,
                        y=spy_aligned.values,
                        name='SPY Benchmark',
                        line=dict(color='gray', width=1, dash='dot')
                    ))
                
                fig_backtest.update_layout(
                    title="Backtest Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_backtest, width="stretch")
                
                # Drawdown chart
                st.subheader("📉 Drawdown Analysis")
                
                returns = analyzer.backtest_res.pct_change().dropna()
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns - running_max) / running_max
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown['Value'] * 100,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red', width=1)
                ))
                
                fig_dd.update_layout(
                    title="Portfolio Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=300
                )
                
                st.plotly_chart(fig_dd, width="stretch")
                
                # Additional metrics
                if analyzer.backtest_metrics.get('tracking_error') is not None:
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric(
                            "Tracking Error",
                            f"{float(analyzer.backtest_metrics.get('tracking_error', 0)):.2%}"
                        )
                    with col6:
                        st.metric(
                            "Information Ratio",
                            f"{float(analyzer.backtest_metrics.get('information_ratio', 0)):.2f}"
                        )
            else:
                st.info("Backtest requires daily frequency data. Please select '1d' interval.")
        
        # TAB 4: Monte Carlo Simulation
        with tab4:
            st.header("Monte Carlo Simulation")
            
            if analyzer.monte_carlo_results is not None:
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Expected Return",
                        f"{analyzer.monte_carlo_results['expected_return']:.2%}"
                    )
                with col2:
                    st.metric(
                        "Expected Volatility",
                        f"{analyzer.monte_carlo_results['expected_volatility']:.2%}"
                    )
                with col3:
                    st.metric(
                        "95% VaR",
                        f"{analyzer.monte_carlo_results['var_95']:.2%}"
                    )
                with col4:
                    st.metric(
                        "95% CVaR",
                        f"{analyzer.monte_carlo_results['cvar_95']:.2%}"
                    )
                
                # Fan chart
                st.subheader("📊 Simulation Fan Chart")
                
                fig_mc = go.Figure()
                
                # Add percentile bands
                percentiles = analyzer.monte_carlo_results['percentiles']
                days = np.arange(len(percentiles[0]))
                
                # Add filled areas for percentile bands
                fig_mc.add_trace(go.Scatter(
                    x=days, y=percentiles[0],
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=False
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=days, y=percentiles[4],
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0.3)'),
                    name='5-95 percentile'
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=days, y=percentiles[1],
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=False
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=days, y=percentiles[3],
                    fill='tonexty',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(0,100,255,0.3)'),
                    name='25-75 percentile'
                ))
                
                # Add median line
                fig_mc.add_trace(go.Scatter(
                    x=days, y=percentiles[2],
                    line=dict(color='blue', width=2),
                    name='Median'
                ))
                
                fig_mc.update_layout(
                    title=f"Monte Carlo Simulation ({n_simulations} paths)",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_mc, width="stretch")
                
                # Distribution of final values
                st.subheader("📊 Distribution of Final Values")
                
                final_values = analyzer.monte_carlo_results['simulations'][:, -1]
                
                fig_hist = px.histogram(
                    final_values,
                    nbins=50,
                    title=f"Distribution of Portfolio Values at Day {mc_days}",
                    labels={'value': 'Portfolio Value ($)', 'count': 'Frequency'}
                )
                
                # Add VaR and CVaR lines
                var_val = np.percentile(final_values, 5)
                cvar_val = final_values[final_values <= var_val].mean()
                
                fig_hist.add_vline(x=var_val, line_dash="dash", line_color="red", 
                                   annotation_text=f"VaR (95%): ${var_val:.0f}")
                fig_hist.add_vline(x=cvar_val, line_dash="dash", line_color="darkred",
                                   annotation_text=f"CVaR (95%): ${cvar_val:.0f}")
                
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, width="stretch")
                
                # Probability table
                st.subheader("📋 Probability Analysis")
                
                prob_data = {
                    'Outcome': [
                        'Gain > 0%',
                        'Gain > 10%',
                        'Gain > 20%',
                        'Loss > 10%',
                        'Loss > 20%'
                    ],
                    'Probability': [
                        (final_values > 100).mean(),
                        (final_values > 110).mean(),
                        (final_values > 120).mean(),
                        (final_values < 90).mean(),
                        (final_values < 80).mean()
                    ]
                }
                
                prob_df = pd.DataFrame(prob_data)
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(prob_df, width="stretch")
            else:
                st.info("Run portfolio optimization first to enable Monte Carlo simulation.")
        
        # TAB 5: Advanced Analytics
        with tab5:
            st.header("Advanced Analytics")
            st.subheader("🌡️ Market Regime Analysis")
            
            # Check if regime data exists and is not empty
            if analyzer.regime is not None and not analyzer.regime.empty:
                fig_regime_timeline = go.Figure()
                
                # Define colors with opacity
                regime_colors = {
                    'Bull': 'rgba(0, 128, 0, 0.2)', 
                    'Bear': 'rgba(255, 0, 0, 0.2)', 
                    'High Vol': 'rgba(255, 165, 0, 0.2)', 
                    'Normal': 'rgba(0, 0, 255, 0.2)', 
                    'Unknown': 'rgba(128, 128, 128, 0.2)'
                }
                
                # Create a DataFrame for grouping consecutive regimes
                df_regime = analyzer.regime.to_frame(name='regime')
                df_regime['group'] = (df_regime['regime'] != df_regime['regime'].shift()).cumsum()
                
                # Add colored background rectangles
                for _, group_data in df_regime.groupby('group'):
                    regime_type = group_data['regime'].iloc[0]
                    # Ensure we have valid dates
                    if len(group_data) > 0:
                        start_date = group_data.index[0]
                        # Extend the rectangle to the next available date or add 1 day
                        end_date = group_data.index[-1] + pd.Timedelta(days=1)
                        
                        fig_regime_timeline.add_vrect(
                            x0=start_date, 
                            x1=end_date, 
                            fillcolor=regime_colors.get(regime_type, 'gray'), 
                            opacity=1, 
                            layer="below", 
                            line_width=0
                        )
                
                # Calculate and Plot SPY (Benchmark)
                spy_rets = analyzer.benchmark_returns.fillna(0)
                
                # Reindex to match the regime timeline exactly
                spy_rets = spy_rets.reindex(analyzer.regime.index).fillna(0)
                
                # Calculate cumulative performance
                spy_norm = (1 + spy_rets).cumprod() * 100
                
                # Normalize to start at 100
                if len(spy_norm) > 0:
                    spy_norm = spy_norm / spy_norm.iloc[0] * 100
                
                fig_regime_timeline.add_trace(go.Scatter(
                    x=spy_norm.index, 
                    y=spy_norm.values, 
                    name='Market Index (SPY)', 
                    line=dict(color='black', width=2)
                ))
                
                fig_regime_timeline.update_layout(
                    title="Market Regime Timeline", 
                    xaxis_title="Date", 
                    yaxis_title="Normalized Price", 
                    height=400, 
                    showlegend=True,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_regime_timeline, width="stretch")
                
                # Regime Stats Columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Regime Distribution**")
                    regime_stats = analyzer.regime.value_counts(normalize=True)
                    for regime, pct in regime_stats.items():
                        st.write(f"• {regime}: {pct:.1%}")
                with col2:
                    st.markdown("**Current Regime**")
                    if not analyzer.regime.empty:
                        current_regime = analyzer.regime.iloc[-1]
                        color_map = {'Bull': 'green', 'Bear': 'red', 'High Vol': 'orange', 'Normal': 'blue'}
                        regime_color = color_map.get(current_regime, 'gray')
                        st.markdown(f"<h3 style='color: {regime_color}'>{current_regime}</h3>", unsafe_allow_html=True)
            else:
                st.warning("Not enough data to calculate Market Regimes. Try increasing the date range.")
            
            # Efficient Frontier Section
            st.markdown("---")
            st.subheader("📈 Efficient Frontier")
            
            # Efficient Frontier Calculations
            returns_mean = analyzer.returns.mean() * analyzer.freq_scaler
            cov_matrix = analyzer.returns.cov() * analyzer.freq_scaler
            
            # Vectorized Efficient Frontier
            n_portfolios = 10000
            n_assets = len(analyzer.returns.columns)
            
            np.random.seed(10)
            
            # 1. Generate all random weights at once (N x M matrix)
            weights = np.random.random((n_portfolios, n_assets))
            # 2. Normalize each portfolio (each row sums to 1)
            weights /= weights.sum(axis=1, keepdims=True)
            
            # 3. Calculate returns for all portfolios (N,)
            port_returns = np.dot(weights, returns_mean.values)
            
            # 4. Calculate volatility for all portfolios (N,)
            # Formula: sqrt(w @ Cov @ w^T) for each portfolio
            # Efficient: sum((w @ Cov) * w, axis=1) then sqrt
            port_vols = np.sqrt(np.sum((weights @ cov_matrix.values) * weights, axis=1))
            
            # 5. Calculate Sharpe ratios for all portfolios (N,)
            port_sharpe = np.divide(
                port_returns - rf_rate,
                port_vols,
                out=np.zeros(n_portfolios),
                where=port_vols != 0
            )
            
            results = np.vstack([port_returns, port_vols, port_sharpe])
            
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=results[1], 
                y=results[0], 
                mode='markers', 
                marker=dict(
                    size=5, 
                    color=results[2], 
                    colorscale='Viridis', 
                    showscale=True, 
                    colorbar=dict(title="Sharpe Ratio")
                ), 
                text=[f"Sharpe: {s:.2f}" for s in results[2]], 
                hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}", 
                name='Random Portfolios'
            ))
            
            if analyzer.weights is not None:
                # Align optimal weights to the returns columns
                curr_weights_series = analyzer.weights.set_index('Ticker')['Weight']
                curr_weights_aligned = np.array([curr_weights_series.get(t, 0.0) for t in analyzer.returns.columns])
                
                curr_return = np.sum(curr_weights_aligned * returns_mean.values)
                curr_std = np.sqrt(np.dot(curr_weights_aligned.T, np.dot(cov_matrix.values, curr_weights_aligned)))
                
                fig_frontier.add_trace(go.Scatter(
                    x=[curr_std], 
                    y=[curr_return], 
                    mode='markers', 
                    marker=dict(size=15, color='red', symbol='star'), 
                    name='Optimal Portfolio'
                ))
            
            fig_frontier.update_layout(
                title="Efficient Frontier", 
                xaxis_title="Risk (Standard Deviation)", 
                yaxis_title="Expected Return", 
                height=500, 
                hovermode='closest'
            )
            st.plotly_chart(fig_frontier, width="stretch")
            
            # Risk Decomposition
            if analyzer.weights is not None and analyzer.attribution is not None:
                st.subheader("🎯 Risk Decomposition")
                risk_contrib = analyzer.attribution['Risk Contribution']
                # Filter out negligible negative risks for pie chart
                risk_contrib = risk_contrib[risk_contrib > 0.0001]
                
                fig_risk = px.pie(
                    values=risk_contrib.values, 
                    names=risk_contrib.index, 
                    title="Portfolio Risk Contribution by Asset", 
                    hole=0.3
                )
                fig_risk.update_layout(height=400)
                st.plotly_chart(fig_risk, width="stretch")
        
        # --- Download Section ---
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            excel_data = analyzer.get_excel()
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"Portfolio_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )
        
        with col2:
            # Create summary text report
            # Robust float casting to prevent formatting errors
            try:
                exp_ret_val = float(analyzer.attribution['Return'].dot(analyzer.attribution['Weight'])) if analyzer.attribution is not None else 0.0
                port_risk_val = float(np.sqrt(analyzer.weights['Weight'].values @ analyzer.returns[analyzer.weights['Ticker']].cov().values @ analyzer.weights['Weight'].values) * np.sqrt(analyzer.freq_scaler)) if analyzer.weights is not None else 0.0
                total_ret_val = float(analyzer.backtest_metrics.get('total_return', 0.0))
                sharpe_val = float(analyzer.backtest_metrics.get('sharpe_ratio', 0.0))
                max_dd_val = float(analyzer.backtest_metrics.get('max_drawdown', 0.0))
            except:
                exp_ret_val = port_risk_val = total_ret_val = sharpe_val = max_dd_val = 0.0

            summary_report = f"""
PORTFOLIO ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
=============
Tickers: {', '.join(analyzer.tickers)}
Period: {start_date} to {end_date}
Optimization Method: {opt_method}
Risk Model: {risk_model}
Return Model: {ret_model}

PORTFOLIO COMPOSITION
====================
{analyzer.weights.to_string() if analyzer.weights is not None else 'Not optimized'}

KEY METRICS
===========
Expected Return: {exp_ret_val:.2%}
Portfolio Risk: {port_risk_val:.2%}

BACKTEST RESULTS
===============
Total Return: {total_ret_val:.2%}
Sharpe Ratio: {sharpe_val:.2f}
Max Drawdown: {max_dd_val:.2%}
"""
            st.download_button(
                label="📄 Download Text Report",
                data=summary_report,
                file_name=f"Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width="stretch"
            )
        
        with col3:
            if st.button("🔄 Reset Analysis", width="stretch"):
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Portfolio Optimizer
    </div>
    """,
    unsafe_allow_html=True
)
