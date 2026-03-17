"""Tests for financial_algo.portfolio."""

import pytest

from financial_algo.portfolio import Portfolio


class TestPortfolioInit:
    def test_default_cash(self):
        p = Portfolio()
        assert p.cash == 10_000.0

    def test_custom_cash(self):
        p = Portfolio(cash=5_000.0)
        assert p.cash == 5_000.0

    def test_negative_cash_raises(self):
        with pytest.raises(ValueError):
            Portfolio(cash=-1.0)

    def test_empty_positions(self):
        assert Portfolio().positions == {}


class TestPortfolioBuy:
    def test_buy_reduces_cash(self):
        p = Portfolio(cash=1_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        assert p.cash == pytest.approx(500.0)

    def test_buy_adds_position(self):
        p = Portfolio(cash=1_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        assert p.positions["AAPL"] == pytest.approx(10.0)

    def test_buy_accumulates_shares(self):
        p = Portfolio(cash=2_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        p.buy("AAPL", shares=5, price=50.0)
        assert p.positions["AAPL"] == pytest.approx(15.0)

    def test_buy_insufficient_cash_raises(self):
        p = Portfolio(cash=100.0)
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.buy("AAPL", shares=10, price=50.0)

    def test_buy_zero_shares_raises(self):
        with pytest.raises(ValueError):
            Portfolio().buy("AAPL", shares=0, price=10.0)

    def test_buy_negative_price_raises(self):
        with pytest.raises(ValueError):
            Portfolio().buy("AAPL", shares=1, price=-1.0)


class TestPortfolioSell:
    def _funded_portfolio(self) -> Portfolio:
        p = Portfolio(cash=1_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        return p

    def test_sell_increases_cash(self):
        p = self._funded_portfolio()
        p.sell("AAPL", shares=5, price=60.0)
        assert p.cash == pytest.approx(500.0 + 300.0)

    def test_sell_reduces_position(self):
        p = self._funded_portfolio()
        p.sell("AAPL", shares=4, price=60.0)
        assert p.positions["AAPL"] == pytest.approx(6.0)

    def test_sell_all_removes_ticker(self):
        p = self._funded_portfolio()
        p.sell("AAPL", shares=10, price=55.0)
        assert "AAPL" not in p.positions

    def test_sell_too_many_raises(self):
        p = self._funded_portfolio()
        with pytest.raises(ValueError, match="Insufficient position"):
            p.sell("AAPL", shares=20, price=50.0)

    def test_sell_unowned_ticker_raises(self):
        with pytest.raises(ValueError):
            Portfolio().sell("TSLA", shares=1, price=100.0)

    def test_sell_zero_shares_raises(self):
        p = self._funded_portfolio()
        with pytest.raises(ValueError):
            p.sell("AAPL", shares=0, price=50.0)


class TestPortfolioMarketValue:
    def test_no_positions(self):
        p = Portfolio(cash=500.0)
        assert p.market_value({}) == pytest.approx(500.0)

    def test_with_positions(self):
        p = Portfolio(cash=1_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        assert p.market_value({"AAPL": 60.0}) == pytest.approx(500.0 + 600.0)

    def test_missing_price_treated_as_zero(self):
        p = Portfolio(cash=1_000.0)
        p.buy("AAPL", shares=10, price=50.0)
        assert p.market_value({}) == pytest.approx(500.0)
