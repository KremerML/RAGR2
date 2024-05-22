import argparse
import json
import os
from typing import Any, Dict
import logging

import jsonlines
import Levenshtein
import tqdm
import time

from prompts import ragr_prompts
from utils import (
    agreement_gate,
    editor,
    evidence_selection,
    question_generation,
)
from narrative_qa_rag import DocumentLoader, retrieve_evidence
from langchain_community.vectorstores import FAISS


def run_editor_one_instance(
    claim: str,
    model: str = "gpt-3.5-turbo-1106", # "text-davinci-003",
    temperature_qgen: float = 0.7,
    num_rounds_qgen: int = 2,
    max_evidences_per_question: int = 1,
    max_edit_ratio: float = 100,
    vecdb: FAISS = None,
) -> Dict[str, Any]:
    """Runs query generation, search, agreement gating, and editing on a claim.

    Args:
        claim: Text to check the validity of.
        model: Name of the OpenAI GPT-3 model to use.
        temperature_qgen: Sampling temperature to use for query generation.
        num_rounds_qgen: Number of times to sample questions.
        max_evidences_per_question: Maximum number of evidences to return per question.
        max_edit_ratio: Maximum edit ratio between claim and edit for each round.
    Returns:
        result: All revision information, including the queries generated, search
            results, agreement gate information, and each revision step done on the
            claim.
    """
    # Timing variables
    # time_question_generation = 0
    # time_evidence_retrieval = 0
    # time_agreement_gate = 0
    # time_editing = 0

    original_claim = claim
    agreement_gates = []

    # start_time = time.time()

    # Generate questions for the claim
    questions = question_generation.ragr_question_generation(
        claim=claim,
        model=model,
        prompt=ragr_prompts.RAG_QGEN_PROMPT,
        temperature=temperature_qgen,
        num_rounds=num_rounds_qgen,
    )
    # time_question_generation = time.time() - start_time

    # Retrieve evidence from vector store based on the generated queries
    evidences_for_questions = [
        retrieve_evidence(query=query, vecdb=vecdb, top_k=max_evidences_per_question)
        for query in questions
    ]
    # TODO: Figure out how to deduplicate evidences
    seen_evidences = set()
    deduplicated_evidences = []
    for evidence_list in evidences_for_questions:
        for evidence in evidence_list:
            evidence_tuple = (evidence["text"], frozenset(evidence["metadata"].items()))
            if evidence_tuple not in seen_evidences:
                seen_evidences.add(evidence_tuple)
                deduplicated_evidences.append(evidence)


    # time_evidence_retrieval = time.time() - time_question_generation - start_time

    # Flatten and deduplicate the evidences per question into a single list.
    # used_evidences = [
    #     e
    #     for e in deduplicated_evidences 
    #     for e in cur_evids
    # ]
    # logging.info(f"All evidence: {used_evidences}")
    logging.info(f"All evidence: {deduplicated_evidences}")

    # Iterative editing over each evidence
    revision_steps = []
    for evid in deduplicated_evidences:
        # Run the agreement gate on the current (claim, context, query, evidence) tuple
        gate = agreement_gate.run_agreement_gate(
            claim=claim,
            query=evid["query"],
            evidence=evid["text"],
            model=model,
            prompt=ragr_prompts.RAG_AGREEMENT_GATE_PROMPT,
        )
        # time_agreement_gate = time.time() - time_evidence_retrieval - time_question_generation - start_time
        agreement_gates.append(gate)

        # Run the editor gate if the agreement gate is open
        if gate["is_open"]:
            edited_claim = editor.run_ragr_editor(
                claim=claim,
                query=evid["query"],
                # evidence=evid["text"],
                reason=gate['reason'],
                model=model,
                prompt=ragr_prompts.RAG_EDITOR_PROMPT,
            )["text"]

            # Don't keep the edit if the editor makes a huge change
            if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                claim = edited_claim

        # time_editing = time.time() - time_agreement_gate - time_evidence_retrieval - time_question_generation - start_time
        revision_steps.append({"text": claim})

    result = {
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revisions": [
            {
                "original_text": original_claim,
                "revised_text": revision_steps[-1]["text"],
                "evidences": deduplicated_evidences,
                "agreement_gates": agreement_gates,
                "revision_steps": revision_steps,
            }
        ],
    }
    selected_evidences = evidence_selection.select_evidences(result)
    result["selected_evidences"] = selected_evidences

    # print(f"Question Generation Time: {time_question_generation:.2f} seconds")
    # print(f"Evidence Retrieval Time: {time_evidence_retrieval:.2f} seconds")
    # print(f"Agreement Gate Time: {time_agreement_gate:.2f} seconds")
    # print(f"Editing Time: {time_editing:.2f} seconds")
    # print(f"Total Time: {time.time() - start_time:.2f} seconds")
    
    return result


def get_args() -> argparse.Namespace:
    """Gets command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSONLines file of claims to run RARR on.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="JSONLines file to write revisions to.",
    )
    parser.add_argument(
        "--claim_field",
        default="model_outputs_explanation",
        type=str,
        help="Field of the JSONL file to run the claim editing on.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo-0125",
        type=str,
        help="OpenAI GPT-3 model to use.",
    )
    parser.add_argument(
        "--temperature_qgen",
        default=0.7,
        type=float,
        help="Sampling temperature to use for query generation.",
    )
    parser.add_argument(
        "--num_rounds_qgen",
        default=3,
        type=int,
        help="Number of times to re-sample queries for a claim.",
    )
    parser.add_argument(
        "--max_evidences_per_question",
        default=1,
        type=int,
        help="Maximum number of evidences to consider per question.",
    )
    parser.add_argument(
        "--max_edit_ratio",
        default=100,
        type=float,
        help="Maximum edit ratio between claim and edit for each round.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes the editing process if broken by loading the output file.",
    )
    args = parser.parse_args()

    # Write all args to file
    with open(args.output_file + "_args", "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, indent=4)
    return args


def main() -> None:
    """Loads a RAGR2 evaluation set and runs GPT-4 RAGR2 editing."""
    args = get_args()

    config = {
        'qaps_path': '/Users/kremerr/Documents/GitHub/RAGR2/archive/narrative_qa/qaps.csv',
        'summaries_path': '/Users/kremerr/Documents/GitHub/RAGR2/archive/narrative_qa/summaries.csv',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'chunk_size': 600,
        'chunk_overlap': 0,
        'vecdb_type': 'FAISS',
        'num_questions': 500,
        'min_len': 40,
        'max_len': 100
    }
    doc_loader = DocumentLoader(config=config)
    doc_loader.load_data()  # Ensures data is loaded
    vecdb = doc_loader.create_vecdb()

    if not vecdb:
        print("Could not initialize vector store. Exiting.")
        return

    # Load the finished results by mapping from the claim name to the results.
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming with results from {args.output_file}")
        finished_results = {
            l["input_info"][args.claim_field]: l["result"]
            for l in jsonlines.open(args.output_file)
        }
        print(f"Found {len(finished_results)} finished lines.")
    else:
        finished_results = None

    with open(args.output_file, "w", encoding="utf-8") as writer:
        lines = list(jsonlines.open(args.input_file))
        for line in tqdm.tqdm(lines):
            claim = line["input_info"][args.claim_field]

            # Search for finished result
            if finished_results and claim in finished_results:
                line["result"] = finished_results[claim]
            else:
                line["result"] = run_editor_one_instance(
                    model=args.model,
                    claim=claim,
                    temperature_qgen=args.temperature_qgen,
                    num_rounds_qgen=args.num_rounds_qgen,
                    max_evidences_per_question=args.max_evidences_per_question,
                    max_edit_ratio=args.max_edit_ratio,
                    vecdb=vecdb
                )
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
