"""Utils for running the agreement gate."""
import os
import time
from typing import Any, Dict, Tuple
import logging
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
    """Extract the agreement gate state and the reasoning from the GPT-4 API response.

    Args:
        api_response: Agreement gate response from GPT-4.
    Returns:
        is_open: Whether the agreement gate is open.
        reason: The reasoning for why the agreement gate is open or closed.
        decision: The decision of the status of the gate in string form.
    """
    try:
        lines = api_response.strip().split("\n")
        reason = lines[0]
        decision = lines[1].split("Therefore:")[-1].strip()
        is_open = "disagrees" in decision
    except Exception as e:
        reason = f"Failed to parse. Error: {str(e)}"
        decision = None
        is_open = False

    return is_open, reason, decision


def run_agreement_gate(
    claim: str,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    num_retries: int = 5,
) -> Dict[str, Any]:
    """Checks if a provided evidence contradicts the claim given a query.

    Checks if the answer to a query using the claim contradicts the answer using the
    evidence. If so, we open the agreement gate, which means that we allow the editor
    to edit the claim. Otherwise the agreement gate is closed.

    Args:
        claim: Text to check the validity of.
        query: Query to guide the validity check.
        evidence: Evidence to judge the validity of the claim against.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        gate: A dictionary with the status of the gate and reasoning for decision.
    """
    client = OpenAI()
    gpt_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()


    for _ in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": gpt_input}
                ],
                temperature=0.0,
                max_tokens=512,
                stop=["\n\n"],
            )
            api_response = response.choices[0].message.content
            logging.info("Agreement gate response: %s", api_response.strip())
            logging.info("Usage: %s", response.usage.total_tokens)

            is_open, reason, decision = parse_api_response(api_response)
            gate = {"is_open": is_open, "reason": reason, "decision": decision}
            logging.info(f"Parsed agreement gate output: {gate}")
            return gate
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error: {str(e)}. Retrying...")
            time.sleep(2)

    return {"is_open": False, "reason": "Max retries exceeded", "decision": None}
