class TelcoPromptEngine:
    """Generates precise, context-aware prompts for the 6G TelcoLLM."""

    SYSTEM_PROMPT = """You are a highly advanced 6G Telecommunications AI Agent responsible for managing Base Transceiver Stations (BTS).
Your task is to analyze network telemetry (like PRB utilization, RSRP, SINR, and Active Users) and output an optimal network configuration intent.

CRITICAL RULES:
1. You MUST output ONLY valid JSON. Do not include any markdown formatting, explanations, or extra text.
2. The JSON must exactly match the schema provided.
3. If PRB utilization > 80%, you should decrease tx_power slightly or adjust ho_hysteresis to offload users.
4. If PRB utilization < 30%, you can increase tx_power to improve coverage.
5. You must provide a clear 'rationale' string explaining your decision based on the telemetry.

JSON SCHEMA:
{
  "rationale": "Explanation of why you are making this adjustment based on telemetry.",
  "category": "CAPACITY_ADJUSTMENT",
  "confidence_score": 0.95,
  "tx_power_dbm": <float>,
  "ho_hysteresis_db": <float>
}
"""

    @staticmethod
    def build_prompt(telemetry_data: dict) -> list:
        """Builds the message list for the LLM chat completion API."""
        user_message = f"""CURRENT NETWORK TELEMETRY:
- PRB Utilization: {telemetry_data.get("prb_utilisation_pct", 0):.2f}%
- Active UEs: {telemetry_data.get("active_ue_count", 0)}
- Average RSRP: {telemetry_data.get("rsrp", -90):.2f} dBm
- Average SINR: {telemetry_data.get("sinr", 15):.2f} dB

Based on this data, generate the JSON intent to optimize the network."""

        return [
            {"role": "system", "content": TelcoPromptEngine.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
