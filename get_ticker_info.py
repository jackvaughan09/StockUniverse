import yfinance as yf
import pandas as pd
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import numpy as np


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


######### Helper Functions: #########
def convert_to_thousands(value):
    if isinstance(value, (int, float)):
        return f"{value/1000:.0f}"
    return value


def calculate_ratio(numerator, denominator):
    if (
        isinstance(numerator, (int, float))
        and isinstance(denominator, (int, float))
        and denominator != 0
    ):
        return f"{numerator / denominator:.2f}"
    return "No result"


def convert_to_percent(value):
    if isinstance(value, float):
        return f"{value * 100:.2f}%"
    return value


def get_latest_value(df, metric):
    if metric in df.index:
        for col in df.columns:
            value = df.at[metric, col]
            if not pd.isna(value):
                return value
    return "Not Found"


######### Metric Calculation Functions #########
def calculate_nwc_change(balance_sheet):
    # NWC (most broad interpretation) = Current Assets - Current Liabilities
    # DeltaNWC = Prior Period NWC - Current Period NWC
    cur_nwc = get_latest_value(balance_sheet, "Working Capital")
    prior_balance = balance_sheet.iloc[:, 1:]
    prior_nwc = get_latest_value(prior_balance, "Working Capital")
    if prior_nwc - cur_nwc == 0:
        print(f"prior: {prior_nwc}, current: {cur_nwc}")
    return prior_nwc - cur_nwc


def _calculate_non_cash_wc_for_period(balance_sheet):
    ar = get_latest_value(balance_sheet, "Accounts Receivable")
    inv = get_latest_value(balance_sheet, "Inventory")
    ap = get_latest_value(balance_sheet, "Accounts Payable")
    return ar + inv - ap


def calculate_non_cash_wc(balance_sheet):
    current = _calculate_non_cash_wc_for_period(balance_sheet)
    prior_bs = balance_sheet.iloc[:, 1:]
    prior = _calculate_non_cash_wc_for_period(prior_bs)
    return prior - current


def calculate_ufcf(cash_flow, income_statement, balance_sheet):
    # Unlevered Free Cash Flow
    # EBIT*(1-tax rate) + D&A – ∆NWC – CAPEX
    nopat = get_latest_value(income_statement, "EBIT") * (
        1 - get_latest_value(income_statement, "Tax Rate For Calcs")
    )  # Net Operating Profit After Tax
    dep_am = get_latest_value(cash_flow, "Depreciation And Amortization")
    change_in_non_cash_wc = calculate_non_cash_wc(balance_sheet)
    capex = get_latest_value(cash_flow, "Capital Expenditure")
    # Calculate UFCF
    ufcf = nopat + dep_am - change_in_non_cash_wc - capex
    return ufcf


def calculate_sfcf(cash_flow):
    # Simple Free Cash Flow
    operating_cash = get_latest_value(cash_flow, "Operating Cash Flow")
    capex = get_latest_value(cash_flow, "Capital Expenditure")
    return operating_cash - capex


def calculate_net_net_wc_to_price(balance_sheet, market_cap):
    # ((Cash + Short Term Marketable Investments + (Accounts Receivable * 75%) + (Inventory * 50%)  - Total Liabilities)
    # / # of shares) / Price
    # convert /shares/price -> /market_cap
    ccesti = get_latest_value(
        balance_sheet, "Cash Cash Equivalents And Short Term Investments"
    )
    disc_AR = 0.75 * get_latest_value(balance_sheet, "Accounts Receivable")
    disc_INV = 0.5 * get_latest_value(balance_sheet, "Inventory")
    tot_liab = get_latest_value(balance_sheet, "Total Assets") - get_latest_value(
        balance_sheet, "Stockholders Equity"
    )
    return float(f"{(ccesti + disc_AR + disc_INV - tot_liab) / market_cap:.2f}")


######### YFinance Fetcher Function #########
# NOTE: Need to adjust dictionary keys to match source sheet headers


def fetch_financial_data(ticker, session):
    def safe_get(func, default="No result"):
        try:
            return func()
        except Exception:
            return default

    stock = yf.Ticker(ticker, session)
    info = stock.info
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow

    financial_data = {
        "Ticker": ticker,
        "rev (ttm)": safe_get(lambda: convert_to_thousands(info.get("totalRevenue"))),
        "roa (ttm)": safe_get(lambda: convert_to_percent(info.get("returnOnAssets"))),
        "roe (ttm)": safe_get(lambda: convert_to_percent(info.get("returnOnEquity"))),
        "ebitda (s&p)": safe_get(lambda: convert_to_thousands(info.get("ebitda"))),
        "ev": safe_get(lambda: convert_to_thousands(info.get("enterpriseValue"))),
        "div": safe_get(lambda: info.get("dividendRate")),
        "cash, equiv & sti": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(
                    balance_sheet, "Cash Cash Equivalents And Short Term Investments"
                )
            )
        ),
        "debt": safe_get(lambda: convert_to_thousands(info.get("totalDebt"))),
        "eps": safe_get(lambda: info.get("trailingEps")),
        "b": safe_get(lambda: info.get("bookValue")),
        "lfcf (ttm)": safe_get(lambda: convert_to_thousands(info.get("freeCashflow"))),
        "cfo (ttm)": safe_get(
            lambda: convert_to_thousands(info.get("operatingCashflow"))
        ),
        "cfi (ttm)": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(cash_flow, "Investing Cash Flow")
            )
        ),
        "cff (ttm)": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(cash_flow, "Financing Cash Flow")
            )
        ),
        "tb": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Tangible Book Value")
            )
        ),
        "pre-tx inc": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(income_statement, "Pretax Income")
            )
        ),
        "inc tax exp": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(income_statement, "Pretax Income")
                * get_latest_value(income_statement, "Tax Rate For Calcs")
            )
        ),
        "intrst exp": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(income_statement, "Interest Expense")
            )
        ),
        "ebit": safe_get(
            lambda: convert_to_thousands(get_latest_value(income_statement, "EBIT"))
        ),
        "capex": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(cash_flow, "Capital Expenditure")
            )
        ),
        "curr assets": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Current Assets")
            )
        ),
        "cash": safe_get(lambda: convert_to_thousands(info.get("totalCash"))),
        "shortterm investment": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Other Short Term Investments")
            )
        ),
        "cur inventory": safe_get(
            lambda: convert_to_thousands(get_latest_value(balance_sheet, "Inventory"))
        ),
        "cur liab": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Current Liabilities")
            )
        ),
        "net receivables": safe_get(
            lambda: convert_to_thousands(get_latest_value(balance_sheet, "Receivables"))
        ),
        "tot assets": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Total Assets")
            )
        ),
        "tot liab": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Total Assets")
                - get_latest_value(balance_sheet, "Stockholders Equity")
            )
        ),
        "ret ear": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Retained Earnings")
            )
        ),
        "∆NWC ttm": safe_get(
            lambda: convert_to_thousands(calculate_nwc_change(balance_sheet))
        ),
        "work cap": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(balance_sheet, "Working Capital")
            )
        ),
        "eff tax rate": safe_get(
            lambda: get_latest_value(income_statement, "Tax Rate For Calcs")
        ),
        "tax shield": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(income_statement, "Interest Expense")
                * get_latest_value(income_statement, "Tax Rate For Calcs")
            )
        ),
        "fcf": safe_get(lambda: convert_to_thousands(info.get("freeCashflow"))),
        "ufcf": safe_get(
            lambda: convert_to_thousands(
                calculate_ufcf(cash_flow, income_statement, balance_sheet)
            )
        ),
        "sfcf": safe_get(lambda: convert_to_thousands(calculate_sfcf(cash_flow))),
        "ncf": safe_get(
            lambda: convert_to_thousands(
                get_latest_value(cash_flow, "Operating Cash Flow")
                + get_latest_value(cash_flow, "Investing Cash Flow")
                + get_latest_value(cash_flow, "Financing Cash Flow")
            )
        ),
        "ncash": safe_get(
            lambda: convert_to_thousands(
                info.get("totalCash", 0) - info.get("totalDebt", 0)
            )
        ),
        "Z1": safe_get(
            lambda: calculate_ratio(
                get_latest_value(balance_sheet, "Working Capital"),
                get_latest_value(balance_sheet, "Total Assets"),
            )
        ),
        "Z2": safe_get(
            lambda: calculate_ratio(
                get_latest_value(balance_sheet, "Retained Earnings"),
                get_latest_value(balance_sheet, "Total Assets"),
            )
        ),
        "Z3": safe_get(
            lambda: calculate_ratio(
                get_latest_value(income_statement, "EBIT"),
                get_latest_value(balance_sheet, "Total Assets"),
            )
        ),
        "Z4": safe_get(
            lambda: calculate_ratio(
                info.get("marketCap", 0),
                get_latest_value(balance_sheet, "Total Assets")
                - get_latest_value(balance_sheet, "Stockholders Equity"),
            )
        ),
        "Z5": safe_get(
            lambda: calculate_ratio(
                get_latest_value(income_statement, "Total Revenue"),
                get_latest_value(balance_sheet, "Total Assets"),
            )
        ),
        "s/ p": safe_get(
            lambda: calculate_ratio(info.get("totalRevenue"), info.get("marketCap"))
        ),
        "ebitda/ ev": safe_get(
            lambda: calculate_ratio(1, info.get("enterpiseToEbitda"))
        ),
        "tb/ p": safe_get(
            lambda: calculate_ratio(
                get_latest_value(balance_sheet, "Net Tangible Assets"),
                info.get("marketCap"),
            )
        ),
        "b/ p": safe_get(lambda: calculate_ratio(1, info.get("priceToBook"))),
        "e/ p": safe_get(
            lambda: calculate_ratio(info.get("trailingEps"), info.get("marketCap"))
        ),
        "cfo/ p": safe_get(
            lambda: calculate_ratio(
                info.get("operatingCashflow"), info.get("marketCap")
            )
        ),
        "sfcf/ p": safe_get(
            lambda: calculate_ratio(calculate_sfcf(cash_flow), info.get("marketCap"))
        ),
        "ncf/ p": safe_get(
            lambda: calculate_ratio(
                get_latest_value(cash_flow, "Operating Cash Flow")
                + get_latest_value(cash_flow, "Investing Cash Flow")
                + get_latest_value(cash_flow, "Financing Cash Flow"),
                info.get("marketCap"),
            )
        ),
        "div/ p": safe_get(
            lambda: calculate_ratio(info.get("dividendRate"), info.get("marketCap"))
        ),
        "cash/ p": safe_get(
            lambda: calculate_ratio(info.get("totalCash"), info.get("marketCap"))
        ),
        "ncash/ p": safe_get(
            lambda: calculate_ratio(
                info.get("totalCash", 0) - info.get("totalDebt", 0),
                info.get("marketCap"),
            )
        ),
        "nn/ p*": safe_get(
            lambda: calculate_net_net_wc_to_price(balance_sheet, info.get("marketCap"))
        ),
        "asset/ p": safe_get(
            lambda: calculate_ratio(
                get_latest_value(balance_sheet, "Total Assets"), info.get("marketCap")
            )
        ),
        "retear/ mc": safe_get(
            lambda: calculate_ratio(
                get_latest_value(balance_sheet, "Retained Earnings"),
                info.get("marketCap"),
            )
        ),
        "revs/ debt": safe_get(
            lambda: calculate_ratio(info.get("totalRevenue"), info.get("totalDebt"))
        ),
        "mc/ ev": safe_get(
            lambda: calculate_ratio(info.get("marketCap"), info.get("enterpriseValue"))
        ),
    }
    return financial_data


########## Create Session #########
def create_session():
    session = CachedLimiterSession(
        limiter=Limiter(
            RequestRate(2, Duration.SECOND * 5)
        ),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    session.headers["User-agent"] = "my-program/1.0"
    return session
