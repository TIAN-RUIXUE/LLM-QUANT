"""
rules/engine.py
Symbolic reasoning layer: validates LLM recommendations against hard rules.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Action(Enum):
    KEEP = "keep"
    DOWNGRADE_CONFIDENCE = "downgrade_confidence"
    OVERRIDE_TO_HOLD = "override_to_hold"
    OVERRIDE_TO_SELL = "override_to_sell"
    OVERRIDE_TO_BUY = "override_to_buy"
    FLAG = "flag"


@dataclass
class RuleResult:
    rule_name: str
    triggered: bool
    action: Action
    description: str
    original_value: Optional[float] = None
    new_value: Optional[float] = None


@dataclass
class ValidationLog:
    rules_fired: List[RuleResult] = field(default_factory=list)
    final_recommendation: str = ""
    final_confidence: float = 0.0
    uncertainty: float = 0.0
    conflicts: List[str] = field(default_factory=list)
    effective_risk_flags: List[str] = field(default_factory=list)


class SymbolicEngine:
    """
    Validates LLM output against symbolic financial rules.
    """

    def __init__(self,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 confidence_threshold: float = 0.4,
                 confidence_downgrade_step: float = 0.2):
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.confidence_threshold = confidence_threshold
        self.confidence_downgrade_step = confidence_downgrade_step

    def validate(self,
                 llm_output: Dict,
                 indicators: Dict) -> ValidationLog:
        """
        Main entry point. Takes LLM JSON output + indicator summary,
        runs all rules, returns validation log.
        """
        log = ValidationLog()
        rec = llm_output.get('recommendation', 'HOLD').upper()
        conf = llm_output.get('confidence')
        conf = 0.5 if conf is None else float(conf)
        sentiment = llm_output.get('sentiment_score')
        sentiment = 0.0 if sentiment is None else float(sentiment)

        signals = indicators.get('signals', {})

        # --- Rule 1: RSI overbought + BUY -> HOLD ---
        if signals.get('rsi_state') == 'OVERBOUGHT' and rec == 'BUY':
            log.rules_fired.append(RuleResult(
                rule_name="RSI_Overbought_Block",
                triggered=True,
                action=Action.OVERRIDE_TO_HOLD,
                description="RSI > 70 indicates overbought; BUY overridden to HOLD",
                original_value=conf
            ))
            rec = 'HOLD'
            conf = min(conf, self.confidence_threshold) 
            log.conflicts.append("RSI overbought contradicts BUY recommendation")

        # --- Rule 2: RSI oversold + SELL -> HOLD ---
        if signals.get('rsi_state') == 'OVERSOLD' and rec == 'SELL':
            log.rules_fired.append(RuleResult(
                rule_name="RSI_Oversold_Block",
                triggered=True,
                action=Action.OVERRIDE_TO_HOLD,
                description="RSI < 30 indicates oversold; SELL overridden to HOLD",
                original_value=conf
            ))
            rec = 'HOLD'
            conf = min(conf, self.confidence_threshold)
            log.conflicts.append("RSI oversold contradicts SELL recommendation")

        # --- Rule 3: Negative sentiment + BUY -> downgrade ---
        if sentiment < -0.5 and rec == 'BUY':
            new_conf = max(0.0, conf - self.confidence_downgrade_step)
            log.rules_fired.append(RuleResult(
                rule_name="Sentiment_Contradiction",
                triggered=True,
                action=Action.DOWNGRADE_CONFIDENCE,
                description=f"Negative sentiment ({sentiment}) contradicts BUY; confidence {conf:.2f} -> {new_conf:.2f}",
                original_value=conf,
                new_value=new_conf
            ))
            conf = new_conf
            log.conflicts.append("Negative news sentiment contradicts BUY")

        # --- Rule 4: Positive sentiment + SELL -> downgrade ---
        if sentiment > 0.5 and rec == 'SELL':
            new_conf = max(0.0, conf - self.confidence_downgrade_step)
            log.rules_fired.append(RuleResult(
                rule_name="Sentiment_Contradiction",
                triggered=True,
                action=Action.DOWNGRADE_CONFIDENCE,
                description=f"Positive sentiment ({sentiment}) contradicts SELL; confidence {conf:.2f} -> {new_conf:.2f}",
                original_value=conf,
                new_value=new_conf
            ))
            conf = new_conf
            log.conflicts.append("Positive news sentiment contradicts SELL")

        # --- Rule 5: Bullish tech (MA + MACD) + SELL -> HOLD ---
        if (signals.get('macd') == 'POSITIVE' and
            signals.get('ma_cross') == 'BULLISH' and
            rec == 'SELL'):
            # Both trend indicators oppose the call -> block it (symmetric with the RSI rules:
            # a counter-trend call justified only by the RSI extreme is not accepted)
            log.rules_fired.append(RuleResult(
                rule_name="Technical_Bullish_Contradiction",
                triggered=True,
                action=Action.OVERRIDE_TO_HOLD,
                description="MACD positive + MA bullish contradicts SELL; SELL overridden to HOLD",
                original_value=conf
            ))
            rec = 'HOLD'
            conf = min(conf, self.confidence_threshold)
            log.conflicts.append("Technical indicators bullish but LLM says SELL")

        # --- Rule 5b: Bearish tech (MA + MACD) + BUY -> HOLD (mirror of Rule 5) ---
        if (signals.get('macd') == 'NEGATIVE' and
            signals.get('ma_cross') == 'BEARISH' and
            rec == 'BUY'):
            # Both trend indicators oppose the call -> block it (symmetric with the RSI rules:
            # a counter-trend call justified only by the RSI extreme is not accepted)
            log.rules_fired.append(RuleResult(
                rule_name="Technical_Bearish_Contradiction",
                triggered=True,
                action=Action.OVERRIDE_TO_HOLD,
                description="MACD negative + MA bearish contradicts BUY; BUY overridden to HOLD",
                original_value=conf
            ))
            rec = 'HOLD'
            conf = min(conf, self.confidence_threshold)
            log.conflicts.append("Technical indicators bearish but LLM says BUY")

        # --- Rule 5c: Mixed trend signals (MA vs MACD disagree) + directional call -> downgrade ---
        # The LLM is instructed to prefer HOLD when indicators conflict; enforce a penalty if it doesn't.
        ma, macd = signals.get('ma_cross'), signals.get('macd')
        mixed = ((ma == 'BULLISH' and macd == 'NEGATIVE') or
                 (ma == 'BEARISH' and macd == 'POSITIVE'))
        if mixed and rec in ('BUY', 'SELL'):
            step = self.confidence_downgrade_step / 2
            new_conf = max(0.0, conf - step)
            log.rules_fired.append(RuleResult(
                rule_name="Mixed_Trend_Signals",
                triggered=True,
                action=Action.DOWNGRADE_CONFIDENCE,
                description=f"MA cross {ma} vs MACD {macd} disagree; {rec} confidence {conf:.2f} -> {new_conf:.2f}",
                original_value=conf,
                new_value=new_conf
            ))
            conf = new_conf
            log.conflicts.append(f"Mixed trend signals (MA {ma}, MACD {macd}) but LLM says {rec}")

        # --- Rule 6: Risk flags + BUY -> downgrade ---
        # (must run BEFORE the low-confidence check, otherwise a BUY can end
        #  below the confidence threshold without being forced to HOLD)
        risk_flags = list(llm_output.get('risk_flags') or [])
        # Drop volatility flags that the measured volatility does not support
        if signals.get('volatility_state') in ('LOW', 'NORMAL'):
            unsupported = [f for f in risk_flags if 'volatil' in str(f).lower()]
            if unsupported:
                risk_flags = [f for f in risk_flags if f not in unsupported]
                log.rules_fired.append(RuleResult(
                    rule_name="Unsupported_Risk_Flag",
                    triggered=True,
                    action=Action.FLAG,
                    description=(f"Ignored {unsupported}: volatility_state is {signals['volatility_state']} "
                                 f"({signals.get('volatility_annualized_pct')}% annualised)")
                ))
        log.effective_risk_flags = risk_flags
        if len(risk_flags) > 0 and rec == 'BUY':
            new_conf = max(0.0, conf - 0.1 * len(risk_flags))
            log.rules_fired.append(RuleResult(
                rule_name="Risk_Flag_Downgrade",
                triggered=True,
                action=Action.DOWNGRADE_CONFIDENCE,
                description=f"{len(risk_flags)} risk flag(s) present; confidence reduced by {0.1 * len(risk_flags):.2f}",
                original_value=conf,
                new_value=new_conf
            ))
            conf = new_conf

        # --- Rule 7: Confidence <= threshold -> force HOLD ---
        # Round away float noise (0.6 - 0.2 = 0.3999...) so a value printed as 0.40
        # is not treated as "below 0.4"
        conf = round(conf, 4)
        # At or below the threshold is not enough conviction for a directional call
        if conf <= self.confidence_threshold and rec != 'HOLD':
            log.rules_fired.append(RuleResult(
                rule_name="Low_Confidence_Force_HOLD",
                triggered=True,
                action=Action.OVERRIDE_TO_HOLD,
                description=f"Confidence {conf:.2f} at/below threshold {self.confidence_threshold}; forced HOLD",
                original_value=conf
            ))
            rec = 'HOLD'

        # --- Compute uncertainty ---
        # Normalise by the number of signals actually checked for this recommendation
        # (previously divided by a fixed 5, so uncertainty could never drop below 20-60%).
        agree_count, total_signals = self._count_agreeing_signals(rec, signals, sentiment)
        log.uncertainty = 1.0 - (agree_count / total_signals) if total_signals else 1.0

        log.final_recommendation = rec
        log.final_confidence = round(conf, 4)

        return log

    def _count_agreeing_signals(self, rec: str, signals: Dict, sentiment: float) -> Tuple[int, int]:
        """Returns (agreeing signals, signals checked)."""
        agree = 0
        if rec == 'BUY':
            if signals.get('ma_cross') == 'BULLISH':
                agree += 1
            if signals.get('macd') == 'POSITIVE':
                agree += 1
            if signals.get('rsi_state') in ('OVERSOLD', 'NORMAL'):
                agree += 1
            if sentiment > 0:
                agree += 1
        elif rec == 'SELL':
            if signals.get('ma_cross') == 'BEARISH':
                agree += 1
            if signals.get('macd') == 'NEGATIVE':
                agree += 1
            if signals.get('rsi_state') in ('OVERBOUGHT', 'NORMAL'):
                agree += 1
            if sentiment < 0:
                agree += 1
        elif rec == 'HOLD':
            if signals.get('rsi_state') == 'NORMAL':
                agree += 1
            if -0.2 <= sentiment <= 0.2:
                agree += 1
            # Mixed trend signals (MA and MACD disagree) support HOLD
            ma_bull = signals.get('ma_cross') == 'BULLISH'
            macd_bull = signals.get('macd') == 'POSITIVE'
            if ma_bull != macd_bull:
                agree += 1
            # Contrarian RSI extreme against an aligned trend (e.g. OVERSOLD in a bearish
            # trend) is a genuine conflict that also supports HOLD
            elif ((signals.get('rsi_state') == 'OVERSOLD' and not ma_bull) or
                  (signals.get('rsi_state') == 'OVERBOUGHT' and ma_bull)):
                agree += 1
            return agree, 3
        return agree, 4

    def format_validation_report(self, log: ValidationLog) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("SYMBOLIC REASONING VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Final Recommendation: {log.final_recommendation}")
        lines.append(f"Final Confidence: {log.final_confidence:.2%}")
        lines.append(f"Uncertainty Estimate: {log.uncertainty:.2%}")
        lines.append("")

        if log.conflicts:
            lines.append("Conflicts Detected:")
            for c in log.conflicts:
                lines.append(f"  WARNING  {c}")
            lines.append("")

        if log.rules_fired:
            lines.append("Rules Fired:")
            for r in log.rules_fired:
                lines.append(f"  * [{r.rule_name}] {r.description}")
            lines.append("")
        else:
            lines.append("  OK No rules triggered. LLM output passed validation.")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    engine = SymbolicEngine()

    # Test case: LLM says BUY but RSI overbought + negative sentiment
    llm_output = {
        'recommendation': 'BUY',
        'confidence': 0.75,
        'sentiment_score': -0.6,
        'risk_flags': ['high_volatility'],
        'reasoning': 'Model predicts upward movement'
    }

    indicators = {
        'signals': {
            'ma_cross': 'BULLISH',
            'rsi': 78.5,
            'rsi_state': 'OVERBOUGHT',
            'macd': 'POSITIVE'
        },
        'raw_indicators': {'rsi': 78.5}
    }

    log = engine.validate(llm_output, indicators)
    print(engine.format_validation_report(log))
    print(f"Final rec: {log.final_recommendation}")
    print(f"Final conf: {log.final_confidence}")