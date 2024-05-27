"""Utils for running the editor."""
import os
import time
from typing import Dict, Union
import logging
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> str:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns a reason for the edit and the edit in two consecutive lines.
    Only extract out the edit from the second line.

    Args:
        api_response: Editor response from GPT-3.
    Returns:
        edited_claim: The edited claim.
    """
    try:
        lines = api_response.strip().split("\n")
        edited_claim = None
        for line in lines:
            if line.startswith("Revised:"):
                edited_claim = line.split("Revised:")[1].strip()
                break
        return edited_claim
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return None


def run_ragr_editor(
    claim: str,
    query: str,
    context: str,
    evidence: str,
    reason: str,
    model: str,
    prompt: str,
    num_retries: int = 5,
) -> Dict[str, str]:
    """Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

    Args:
        claim: Text to edit.
        query: Query to guide the editing.
        evidence: Evidence to base the edit on.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        edited_claim: The edited claim.
    """
    client = OpenAI()
    gpt_input = prompt.format(
        claim=claim, 
        query=context
        if context
        else query, 
        reason=reason,
        evidence=evidence).strip()

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
            logging.info("Edited claim: %s", response.choices[0].message.content.strip())
            logging.info("Usage: %s", response.usage.total_tokens)
            break

        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)

    edited_claim = parse_api_response(response.choices[0].message.content.strip())
    logging.info(f"Parsed editor response: {edited_claim}")
    # If there was an error in GPT-4 generation, return the original claim.
    if not edited_claim:
        edited_claim = claim
    output = {"text": edited_claim}
    return output
