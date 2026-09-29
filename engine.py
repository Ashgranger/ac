"""Level 7 Quantitative Market-Making Engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

from config import Config
from market import Market, MarketData
from ledger import Ledger
from utils import BPS, BUY, SELL, ZERO, ONE, clamp, q_down, q_up, fmt


@dataclass
class QuoteTarget:
    pair_index: int
    side: str
    price: Decimal
    qty: Decimal
    expected_value_bps: Decimal
    fill_probability: float
    is_exit_quote: bool


class MarketMakingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_fair_value(self, md: MarketData, now: float, ledger: Optional[Ledger] = None) -> Decimal:
        base_mid = md.mid
        if base_mid is None:
            return ZERO

        if not self.cfg.enable_orderbook_intel:
            return md.micro if (self.cfg.use_micro and md.micro) else base_mid

        micro = md.micro if md.micro else base_mid
        half_spr = (md.ask - md.bid) / Decimal("2") if (md.bid and md.ask) else ZERO

        l = ledger.learner if (ledger and hasattr(ledger, "learner") and self.cfg.enable_online_learning) else None
        alpha = l.obi_alpha if l else self.cfg.obi_alpha
        beta = l.tfi_beta if l else self.cfg.tfi_beta

        obi = md.obi
        obi_shift = half_spr * obi * alpha

        tfi = md.trade_flow_imbalance(10.0, now)
        tfi_shift = half_spr * tfi * beta

        fair_val = micro + obi_shift + tfi_shift
        if md.bid and md.ask and md.bid < md.ask:
            fair_val = clamp(fair_val, md.bid, md.ask)
        return fair_val

    def fill_probability(self, distance_bps: Decimal, ledger: Optional[Ledger] = None) -> float:
        d = float(max(ZERO, distance_bps))
        l = ledger.learner if (ledger and hasattr(ledger, "learner") and self.cfg.enable_online_learning) else None
        kappa = float(l.fill_prob_kappa if l else self.cfg.fill_prob_kappa)
        return math.exp(-kappa * d)

    def expected_adverse_move(self, side: str, md: MarketData, ledger: Ledger, now: float) -> Decimal:
        base_tox = ledger.side_tox_bps(side)
        
        ret_5s = md.ret_bps(self.cfg.trend_window_s, now)
        momentum_risk = ZERO
        if side == BUY and ret_5s < 0:
            momentum_risk = abs(ret_5s) * self.cfg.trend_widen
        elif side == SELL and ret_5s > 0:
            momentum_risk = ret_5s * self.cfg.trend_widen

        tfi = md.trade_flow_imbalance(10.0, now)
        flow_risk = ZERO
        if side == BUY and tfi < Decimal("-0.2"):
            flow_risk = abs(tfi) * Decimal("1.5")
        elif side == SELL and tfi > Decimal("0.2"):
            flow_risk = tfi * Decimal("1.5")

        total_adverse = base_tox + momentum_risk + flow_risk
        return total_adverse

    def compute_reservation_price(self, fair_value: Decimal, position_usd: Decimal,
                                  vol_bps: Decimal, ledger: Optional[Ledger] = None) -> Decimal:
        if self.cfg.max_position_usd <= 0:
            return fair_value
        q = clamp(position_usd / self.cfg.max_position_usd, Decimal("-1"), Decimal("1"))
        
        q_eff = Decimal(str(math.copysign(math.pow(abs(float(q)), 1.3), float(q))))
        l = ledger.learner if (ledger and hasattr(ledger, "learner") and self.cfg.enable_online_learning) else None
        skew_rate = l.skew_bps if l else self.cfg.skew_bps
        gamma_rate = l.gamma_risk_aversion if l else self.cfg.gamma_risk_aversion

        inv_skew_bps = q_eff * skew_rate
        if vol_bps > 0:
            inv_skew_bps += q_eff * vol_bps * gamma_rate

        res_price = fair_value * (ONE - inv_skew_bps / BPS)
        return res_price

    def generate_ladder_quotes(
        self,
        m: Market,
        md: MarketData,
        ledger: Ledger,
        now: float,
        buy_blocked: bool,
        sell_blocked: bool,
        existing_slots: Optional[set] = None
    ) -> List[QuoteTarget]:
        if not md.bid or not md.ask or md.bid >= md.ask or not md.mid:
            return []

        mid = md.mid
        tick = m.tick_for(mid)
        step = m.step
        l = ledger.learner if (ledger and hasattr(ledger, "learner") and self.cfg.enable_online_learning) else None
        min_edge = l.min_edge_bps if l else self.cfg.min_edge_bps
        max_edge = l.max_edge_bps if l else self.cfg.max_edge_bps
        vol_k = l.vol_k if l else self.cfg.vol_k
        tox_mult = l.tox_mult if l else self.cfg.tox_mult
        tox_spread_mult = l.regime_toxic_spread_mult if l else self.cfg.regime_toxic_spread_mult
        spacing = l.level_spacing_bps if l else self.cfg.level_spacing_bps
        size_mult_base = l.level_size_mult if l else self.cfg.level_size_mult
        min_ev_base = l.min_ev_bps if l else self.cfg.min_ev_bps

        fair_val = self.compute_fair_value(md, now, ledger=ledger)
        pos_usd = ledger.position * mid
        res_price = self.compute_reservation_price(fair_val, pos_usd, md.vol_bps, ledger=ledger)

        regime = md.detect_regime(now, ledger.tox_bps)
        is_toxic = (regime == "REGIME_D_TOXIC")
        
        base_edge_bps = min_edge + vol_k * md.vol_bps
        if is_toxic:
            base_edge_bps = base_edge_bps * tox_spread_mult
        base_edge_bps = clamp(base_edge_bps, min_edge, max_edge)

        # Microstructure Flow & Asymmetric Quote Shading (Stoikov & Cartea-Jaimungal)
        obi = md.obi
        tfi = md.trade_flow_imbalance(10.0, now)
        flow_bias = Decimal("0.6") * obi + Decimal("0.4") * tfi
        half_spr = (md.ask - md.bid) / Decimal("2") if (md.bid and md.ask) else ZERO
        obi_alpha = l.obi_alpha if l else self.cfg.obi_alpha

        bid_asym_shift = max(ZERO, -flow_bias) * obi_alpha * half_spr
        ask_asym_shift = -max(ZERO, -flow_bias) * Decimal("0.3") * half_spr
        if flow_bias > 0:
            ask_asym_shift = flow_bias * obi_alpha * half_spr
            bid_asym_shift = -flow_bias * Decimal("0.3") * half_spr

        buy_tox = ledger.side_tox_bps(BUY)
        sell_tox = ledger.side_tox_bps(SELL)
        max_tox_addon = Decimal("2.5")
        buy_tox_penalty = min(max_tox_addon, buy_tox * tox_mult) if self.cfg.enable_online_learning else ZERO
        sell_tox_penalty = min(max_tox_addon, sell_tox * tox_mult) if self.cfg.enable_online_learning else ZERO

        quotes: List[QuoteTarget] = []
        stress_loss_limit = l.stress_loss_bps if l else self.cfg.stress_loss_bps
        max_hold_time = l.max_hold_s if l else self.cfg.max_hold_s

        is_stressed = (pos_usd != 0 and (
            ledger.hold_s(now) > max_hold_time or
            (ledger.unrealized(mid) / abs(pos_usd) * BPS < -stress_loss_limit)
        ))
        ret_5s = md.ret_bps(self.cfg.trend_window_s, now)
        trend_pull = l.trend_pull_bps if l else self.cfg.trend_pull_bps
        chasing_top = (ret_5s > Decimal("0.8") and (is_toxic or flow_bias > Decimal("0.3")))
        chasing_bottom = (ret_5s < Decimal("-0.8") and (is_toxic or flow_bias < Decimal("-0.3")))

        total_levels = 1 + max(0, self.cfg.extra_levels)

        remaining_buy_usd = max(ZERO, self.cfg.max_position_usd - pos_usd)
        remaining_sell_usd = max(ZERO, self.cfg.max_position_usd + pos_usd)

        for k in range(total_levels):
            k_spacing = Decimal(str(k)) * spacing
            if is_toxic:
                k_spacing = k_spacing * Decimal("2.0")
            level_edge = base_edge_bps + k_spacing
            
            size_mult = Decimal(str(math.pow(float(size_mult_base), k)))
            level_usd = max(self.cfg.order_usd * size_mult, m.min_notional)

            # --- BUY SIDE --- #
            is_unwind_buy = (pos_usd < 0)
            if is_unwind_buy:
                # UNWIND SHORT: Exit short with minimum profit target or breakeven shading
                if k == 0:
                    min_profit_bps = max(self.cfg.exit_min_profit_bps, Decimal("1.0"))
                    hold_time = ledger.hold_s(now)
                    if hold_time > 180.0:
                        min_profit_bps = Decimal("0.2")
                    elif hold_time > 60.0:
                        min_profit_bps = Decimal("0.8")

                    min_profit_px = ledger.avg_cost * (ONE - min_profit_bps / BPS) if ledger.avg_cost > ZERO else md.bid

                    if is_stressed and (ledger.unrealized(mid) / abs(pos_usd) * BPS < -self.cfg.stress_loss_bps):
                        cand_px = md.bid
                        if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                            cand_px = md.bid + tick
                    else:
                        cand_px = min(md.bid, min_profit_px)
                        if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick and (md.bid + tick) <= min_profit_px:
                            cand_px = md.bid + tick

                    cand_px = min(cand_px, md.ask - tick)
                    cand_px = q_down(cand_px, tick)
                    if md.ask and cand_px >= md.ask:
                        cand_px = md.ask - tick
                    qty = q_down(abs(ledger.position), step)
                    if cand_px > ZERO and qty >= m.min_size:
                        quotes.append(QuoteTarget(
                            pair_index=0, side=BUY, price=cand_px, qty=qty,
                            expected_value_bps=Decimal("1.0"), fill_probability=0.9,
                            is_exit_quote=True
                        ))
            else:
                # ADDING LONG: Quote as long as inventory has room and side is not blocked
                severe_sell_pressure = (flow_bias < Decimal("-0.50") or (is_toxic and md.obi < Decimal("-0.55")))
                toxic_extra_level = (is_toxic and k > 0)
                # INVENTORY ROTATION & ANTI-CHASING: If already long, suppress touch L0 buy!
                already_long = (pos_usd >= self.cfg.order_usd * Decimal("0.5"))
                suppress_buy = (already_long and k == 0) or (chasing_top and k == 0)
                can_add = (not buy_blocked) and (not severe_sell_pressure) and (not toxic_extra_level) and (not suppress_buy) and (remaining_buy_usd >= level_usd)
                if can_add:
                    if k == 0 and self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick and not is_toxic and flow_bias >= Decimal("-0.2"):
                        cand_px = md.bid + tick
                    else:
                        cand_px = (res_price - bid_asym_shift) * (ONE - (level_edge + buy_tox_penalty) / BPS)
                    
                    cand_px = min(cand_px, md.ask - tick)
                    cand_px = q_down(cand_px, tick)
                    if md.ask and cand_px >= md.ask:
                        cand_px = md.ask - tick
                    
                    if cand_px > ZERO:
                        qty = q_down(level_usd / cand_px, step)
                        if qty >= m.min_size:
                            capture_bps = (fair_val - cand_px) / fair_val * BPS
                            adv_bps = self.expected_adverse_move(BUY, md, ledger, now)
                            dist_bps = (md.ask - cand_px) / mid * BPS
                            p_fill = self.fill_probability(dist_bps, ledger=ledger)
                            fee_bps = self.cfg.maker_fee_bps
                            skew_rate = l.skew_bps if l else self.cfg.skew_bps
                            inv_cost_bps = max(ZERO, pos_usd / self.cfg.max_position_usd) * skew_rate
                            
                            ev_bps = Decimal(str(p_fill)) * (capture_bps - adv_bps) - fee_bps - inv_cost_bps
                            
                            min_ev = max(ZERO, min_ev_base - self.cfg.ev_hysteresis_bps) if (existing_slots and (k, BUY) in existing_slots) else min_ev_base

                            if (not self.cfg.enable_adaptive_ev) or (ev_bps >= min_ev):
                                quotes.append(QuoteTarget(
                                    pair_index=k, side=BUY, price=cand_px, qty=qty,
                                    expected_value_bps=ev_bps, fill_probability=p_fill,
                                    is_exit_quote=False
                                ))
                                remaining_buy_usd -= (qty * cand_px)

            # --- SELL SIDE --- #
            is_unwind_sell = (pos_usd > 0)
            if is_unwind_sell:
                # UNWIND LONG: Exit long with minimum profit target or breakeven shading
                if k == 0:
                    min_profit_bps = max(self.cfg.exit_min_profit_bps, Decimal("1.0"))
                    hold_time = ledger.hold_s(now)
                    if hold_time > 180.0:
                        min_profit_bps = Decimal("0.2")
                    elif hold_time > 60.0:
                        min_profit_bps = Decimal("0.8")

                    min_profit_px = ledger.avg_cost * (ONE + min_profit_bps / BPS) if ledger.avg_cost > ZERO else md.ask

                    if is_stressed and (ledger.unrealized(mid) / abs(pos_usd) * BPS < -self.cfg.stress_loss_bps):
                        cand_px = md.ask
                        if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                            cand_px = md.ask - tick
                    else:
                        cand_px = max(md.ask, min_profit_px)
                        if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick and (md.ask - tick) >= min_profit_px:
                            cand_px = md.ask - tick

                    cand_px = max(cand_px, md.bid + tick)
                    cand_px = q_up(cand_px, tick)
                    if md.bid and cand_px <= md.bid:
                        cand_px = md.bid + tick
                    qty = q_down(abs(ledger.position), step)
                    if cand_px > ZERO and qty >= m.min_size:
                        quotes.append(QuoteTarget(
                            pair_index=0, side=SELL, price=cand_px, qty=qty,
                            expected_value_bps=Decimal("1.0"), fill_probability=0.9,
                            is_exit_quote=True
                        ))
            else:
                # ADDING SHORT: Quote as long as inventory has room and side is not blocked
                severe_buy_pressure = (flow_bias > Decimal("0.50") or (is_toxic and md.obi > Decimal("0.55")))
                toxic_extra_level = (is_toxic and k > 0)
                # INVENTORY ROTATION & ANTI-CHASING: If already short, suppress touch L0 sell!
                already_short = (-pos_usd >= self.cfg.order_usd * Decimal("0.5"))
                suppress_sell = (already_short and k == 0) or (chasing_bottom and k == 0)
                can_add = (not sell_blocked) and (not severe_buy_pressure) and (not toxic_extra_level) and (not suppress_sell) and (remaining_sell_usd >= level_usd)
                if can_add:
                    if k == 0 and self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick and not is_toxic and flow_bias <= Decimal("0.2"):
                        cand_px = md.ask - tick
                    else:
                        cand_px = (res_price + ask_asym_shift) * (ONE + (level_edge + sell_tox_penalty) / BPS)
                    
                    cand_px = max(cand_px, md.bid + tick)
                    cand_px = q_up(cand_px, tick)
                    if md.bid and cand_px <= md.bid:
                        cand_px = md.bid + tick
                    
                    if cand_px > ZERO:
                        qty = q_down(level_usd / cand_px, step)
                        if qty >= m.min_size:
                            capture_bps = (cand_px - fair_val) / fair_val * BPS
                            adv_bps = self.expected_adverse_move(SELL, md, ledger, now)
                            dist_bps = (cand_px - md.bid) / mid * BPS
                            p_fill = self.fill_probability(dist_bps, ledger=ledger)
                            fee_bps = self.cfg.maker_fee_bps
                            skew_rate = l.skew_bps if l else self.cfg.skew_bps
                            inv_cost_bps = max(ZERO, -pos_usd / self.cfg.max_position_usd) * skew_rate
                            
                            ev_bps = Decimal(str(p_fill)) * (capture_bps - adv_bps) - fee_bps - inv_cost_bps
                            
                            min_ev = max(ZERO, min_ev_base - self.cfg.ev_hysteresis_bps) if (existing_slots and (k, SELL) in existing_slots) else min_ev_base

                            if (not self.cfg.enable_adaptive_ev) or (ev_bps >= min_ev):
                                quotes.append(QuoteTarget(
                                    pair_index=k, side=SELL, price=cand_px, qty=qty,
                                    expected_value_bps=ev_bps, fill_probability=p_fill,
                                    is_exit_quote=False
                                ))
                                remaining_sell_usd -= (qty * cand_px)

        return quotes
