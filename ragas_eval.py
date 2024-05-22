import os
import pandas as pd
import ast
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness
)
from langchain_community.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper



def extract_data(file_path, chunk_size):
    # Extract data in chunks for memory optimization.
    return pd.read_csv(file_path, chunksize=chunk_size)

def load_data(df, file_path, mode='append'):
    if mode == 'append' and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

@retry(wait=wait_fixed(60), stop=stop_after_attempt(5))
def evaluate_with_retry(dataset,gpt_wrapper):
    result = evaluate(
        dataset=dataset, 
        metrics=[
            # context_precision,
            # context_recall,
            # faithfulness,
            answer_relevancy,
            answer_correctness
        ],
        llm=gpt_wrapper,
        is_async=False
    )
    return result 

def chunk_to_dict(chunk):
    # question,response,ragr_evidence,ragr_answer,answer1,answer2
    questions = chunk['question'].to_list()
    ragr_evidences = chunk['ragr_evidence'].to_list()
    evidences = [ast.literal_eval(evid) for evid in ragr_evidences]
    ground_truths = chunk['answer1'].to_list()
    ragr_answers = chunk['response'].to_list()

    data = { 
        "question": questions,
        "contexts": evidences,
        "ground_truth": ground_truths,
        "answer": ragr_answers
    }
    dataset = Dataset.from_dict(data)
    return dataset

def main():
    output_path = '/Users/kremerr/Documents/GitHub/RAGR2/archive/ragas_eval/RAG_long_ans_500_ans_metrics.csv'
    input_path = '/Users/kremerr/Documents/GitHub/RAGR2/archive/benchmarks/long_ans_500_final_merged_benchmark.csv'
    
    gpt3_5 = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt3_5)
    # llm = load_huggingface_model()

    first_chunk = True
    for chunk in extract_data(input_path, chunk_size=30):
        dataset = chunk_to_dict(chunk)
        result = evaluate_with_retry(dataset,gpt_wrapper)
        ragas_eval_results = result.to_pandas()
        if first_chunk:
            load_data(ragas_eval_results, output_path, mode='replace')
            first_chunk = False
        else:
            load_data(ragas_eval_results, output_path, mode='append')
        time.sleep(15)

if __name__ == "__main__":
    main()
