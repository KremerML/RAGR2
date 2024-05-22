"""Utils for running question generation."""
import os
import time
from typing import List
import logging
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the GPT-4 API response.

    Args:
        api_response: Question generation response from GPT-4.
    Returns:
        questions: A list of questions.
    """
    questions = []
    lines = api_response.split("\n")
    for line in lines:
        if "I asked:" in line:
            question = line.split("I asked:")[1].strip()
            questions.append(question)
    return questions


def ragr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float,
    num_rounds: int,
    num_retries: int = 5,
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3/4 model to use.
        prompt: The prompt template to query GPT-3/4 with.
        temperature: Temperature to use for sampling questions. 0 represents greedy decoding.
        num_rounds: Number of times to sample questions.
        context: Additional context to provide in the prompt.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        questions: A list of questions.
    """
    client = OpenAI()
    gpt_input = prompt.format(claim=claim).strip()

    questions = set()
    for _ in range(num_rounds):
        for _ in range(num_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": gpt_input}
                    ],
                    temperature=temperature,
                    max_tokens=256,
                )
                logging.info("Generated query: %s", response.choices[0].message.content.strip())
                logging.info("Usage: %s", response.usage.total_tokens)
                cur_round_questions = parse_api_response(
                    response.choices[0].message.content.strip()
                )
                questions.update(cur_round_questions)
                break
            except openai.OpenAIError as exception:
                print(f"{exception}. Retrying...")
                time.sleep(1)

    questions = list(sorted(questions))
    logging.info(f"Collected Questions: {questions}")
    return questions
