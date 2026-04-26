from datetime import datetime
import json
from lib.utils.environment import get_conf_for
from lib.utils.connectors import URLCaller
from lib.finance.finance import get_current_week_data
from lib.utils.cache import JSONFileCache

from lib.utils.log import logger

log = logger.get_logger()


def get_headers():
    conf = get_conf_for("open_router")

    return {
        "Authorization": f"Bearer {conf.get('api_key')}",
        "Content-Type": "application/json"
    }

def get_alert_payload(model: str):
    """Generate the payload for getting financial alerts."""
    transactions_json = get_current_week_data()

    return {
  "model": model,
  "temperature": 0,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "alerts_payload",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "alerts": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "id": { "type": "integer" },
                "type": { "type": "string", "enum": ["critical", "warning", "info"] },
                "heading": { "type": "string" },
                "subtext": { "type": "string" },
                "user_id": { "type": "string" },
                "timestamp": { "type": "string", "format": "date-time" }
              },
              "required": ["id", "type", "heading", "subtext", "user_id", "timestamp"]
            }
          }
        },
        "required": ["alerts"],
        "additionalProperties": False
      }
    }
  },
  "messages": [
    {
      "role": "system",
      "content": "You are a financial alert engine and should behave like a concise, practical finance advisor. Follow these rules exactly and return ONLY a single JSON object that validates the supplied json_schema (no markdown, no explanations, no code fences):\n\nOVERVIEW:\n- You will be given a Transactions JSON array. Do NOT aggregate or sum multiple transactions. Process each transaction as an individual item and decide whether to emit an alert for that single transaction only.\n\nINPUT FORMAT:\n- Each transaction object contains: idx, id, description, amount (string or number, euros), transaction_date (YYYY-MM-DD), category_id (integer), created_at (ISO 8601 with timezone or space-separated +offset), user_id.\n\nTHRESHOLDS (FIXED):\n- Use this category mapping (category_id → (name, threshold €)):\n  1: Groceries, 80\n  2: Transportation, 30\n  3: Clothing, 25\n  4: Restaurants, 40\n  5: Family, 35\n  6: Office/Business, 25\n  7: Others, 40\n  17: Subscriptions, 20\n- If a transaction's category_id is not listed above, treat it as 'Others' with a threshold of €50.\n\nALERT RULE (PER-ITEM, STRICT):\n- For each transaction, compare the transaction.amount (as a decimal in euros) to the threshold for that transaction's category.\n- Emit an alert for that transaction **only if transaction.amount > threshold** (strictly greater). Do NOT emit alerts for transactions ≤ threshold.\n- Do NOT sum or group transactions for alerting purposes.\n\nALERT CONTENT & FORMATTING (PER-ITEM):\n- id: sequential integer starting at 1 (see ORDER below).\n- type: use \"critical\" for item-level budget-exceeded alerts.\n- heading: concise Title Case describing the alert. Example: \"Family Budget Exceeded\".\n- subtext: one sentence that references the transaction.description and states where the expense exceeded the threshold and by how much. It must include the transaction amount and the overage with € prefix and two decimals (dot decimal separator), category name, and a short mention of the transaction_date. Example: \"A €120.00 charge at 'Dad' on 2025-10-27 exceeded the Family threshold by €85.00.\"\n- user_id: use the transaction.user_id for that alert.\n\nTIMESTAMP (PER-ITEM):\n- For each alert set timestamp = the transaction.created_at converted to UTC with a Z suffix (no offset), trimmed to seconds precision. Example: transaction.created_at '2025-10-27T15:13:15.55967+00:00' → '2025-10-27T15:13:15Z'.\n\nORDER & IDS:\n- Collect all item-level alerts and then order them by transaction_date ascending, then by transaction.description ascending. Assign ids sequentially starting at 1 according to that order.\n\nOUTPUT STRICTNESS:\n- Return ONLY the JSON object that matches the json_schema. No markdown, no extra fields, no explanations, no code fences.\n- If there are no alerts, return {\"alerts\":[]}.\n\nCURRENCY & ROUNDING:\n- Always format euro amounts with a € prefix and two decimals. Round HALF-UP to two decimals.\n\nDETERMINISM:\n- Use temperature = 0 and follow the instructions exactly. Do not invent categories or thresholds beyond the rules above."
    },
    {
      "role": "user",
      "content": f"Transactions JSON: {transactions_json}"
    }
  ]
}

def get_alerts():
    conf = get_conf_for("open_router")
    base_url = conf.get("base_url")
    ai_model = conf.get("ai_model")

    url_caller = URLCaller(headers=get_headers())

    response = url_caller.perform_single_call(
        url= base_url,
        verb="post",
        json= get_alert_payload(model=ai_model),
    )
    cache = JSONFileCache(name="open_router_response.json")
    log.debug(f"Open Router response: {response.status}")

    cache.save(response.json)

def test_clean_up_result():

    alerts_json_string = "{\"alerts\":[{\"id\":1,\"type\":\"critical\",\"heading\":\"Family Budget Exceeded\",\"subtext\":\"A €120.00 charge at 'Dad' on 2025-10-27 exceeded the Family threshold by €85.00.\",\"user_id\":\"1b9db96d-0871-42ca-bba9-d5d4e9a5d381\",\"timestamp\":\"2025-10-27T15:13:15Z\"},{\"id\":2,\"type\":\"critical\",\"heading\":\"Others Budget Exceeded\",\"subtext\":\"A €722.18 charge at 'Air india(sehaj)' on 2025-10-30 exceeded the Others threshold by €682.18.\",\"user_id\":\"1b9db96d-0871-42ca-bba9-d5d4e9a5d381\",\"timestamp\":\"2025-10-30T21:22:53Z\"},{\"id\":3,\"type\":\"critical\",\"heading\":\"Others Budget Exceeded\",\"subtext\":\"A €219.58 charge at 'Ryanair (dad)' on 2025-10-30 exceeded the Others threshold by €179.58.\",\"user_id\":\"1b9db96d-0871-42ca-bba9-d5d4e9a5d381\",\"timestamp\":\"2025-10-30T19:37:02Z\"}]}"
    alerts_dict = json.loads(alerts_json_string)

    JSONFileCache(name="alerts.json").save(alerts_dict)
if __name__ == "__main__":
    # test_clean_up_result()
    get_alerts()
