# llm/prompt.py

def build_system_prompt() -> str:
    return """You are a professional quantitative investment analyst, specializing in making investment decisions based on technical indicators and market data.
Your task is to analyze the given trading data and market conditions, provide a clear investment recommendation, and explain your reasoning.

**Important Rules**:
1. Prioritize using content from the "Retrieved Reference Information" as evidence.
2. If the reference information conflicts with the indicators, explain the contradiction in the reasoning.
3. Cite specific sources for each conclusion (e.g., "[source1]").
4. If retrieved information is insufficient, make conservative judgments based on technical indicators.

Please strictly output in the following JSON format, without adding any other content:

{
    "action": "BUY/SELL/HOLD",
    "reason": "Explain why you made this decision, citing specific data",
    "risk_level": "Low/Medium/High",
    "confidence": number between 0.0-1.0 indicating confidence level
}"""


def build_user_prompt(context: dict) -> str:
    """
    Build user prompt based on the output of decision_engine.build_prompt_context()
    """
    trade = context["trade"]
    summary = context.get("portfolio_summary") or {}

    # build_prompt_context() provides exit_price / profit_pct, not current_price / unrealized_pnl
    current_price = trade.get('current_price', trade.get('exit_price'))
    pnl = trade.get('unrealized_pnl', trade.get('profit_pct', 0.0))
    price_line = f"${current_price:.2f}" if current_price is not None else "Unknown"

    if 'total_trades' in summary:
        summary_text = (
            f"- Total Trades: {summary['total_trades']}\n"
            f"- Win Rate: {summary.get('win_rate', 0):.1f}%\n"
            f"- Average Return per Trade: {summary.get('avg_profit_pct', 0):.2f}%\n"
            f"- Cumulative Total Return: {summary.get('total_return', 0):.2f}%"
        )
    else:
        summary_text = f"- {summary.get('message', 'No portfolio data available')}"

    return f"""Please analyze the following trading data and provide an investment recommendation:

[Current Trade Information]
- Entry Date: {trade['entry_date']}
- Exit/Current Date: {trade.get('exit_date', 'Holding')}
- Entry Price: ${trade['entry_price']:.2f}
- Exit/Current Price: {price_line}
- Shares Held: {trade['shares']}
- PnL: {pnl:+.2f}%

[Portfolio Historical Performance]
{summary_text}

[Technical State] (if available)
{trade.get('technical_state', 'No technical state data')}

Please provide your investment recommendation based on the above information."""


if __name__ == "__main__":
    from strategy.decision_engine import DecisionEngine
    import pandas as pd

    # Example usage
    print("Prompt builder module loaded.")