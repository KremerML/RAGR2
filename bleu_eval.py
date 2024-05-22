import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Function to tokenize sentences
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def compute_bleu_scores(response, ragr_answer, answer1, answer2):
    """
    Computes BLEU scores between the response/ragr_answer and the answer1/answer2.

    :param response: The generated response.
    :param ragr_answer: The revised response using RAGR.
    :param answer1: Correct answer version 1.
    :param answer2: Correct answer version 2.
    :return: Dictionary with BLEU scores.
    """
    # Tokenize the sentences
    response_tokens = simple_tokenize(response)
    ragr_answer_tokens = simple_tokenize(ragr_answer)
    answer1_tokens = simple_tokenize(answer1)
    answer2_tokens = simple_tokenize(answer2)
    
    reference_answers = [answer1_tokens, answer2_tokens]
    
    # Smoothing function to handle short sentences
    smoothing = SmoothingFunction().method1
    
    # Compute BLEU scores
    response_bleu1 = sentence_bleu(reference_answers, response_tokens, smoothing_function=smoothing)
    response_bleu2 = sentence_bleu(reference_answers, ragr_answer_tokens, smoothing_function=smoothing)
    
    return {
        'response_bleu1': response_bleu1,
        'response_bleu2': response_bleu2
    }

def main():
    filepath = '/Users/kremerr/Documents/GitHub/RAGR2/archive/benchmarks/final_merged_benchmark_500.csv'
    data = pd.read_csv(filepath)
    # Apply the function to the dataframe
    data['bleu_scores'] = data.apply(lambda row: compute_bleu_scores(row['response'], row['ragr_answer'], row['answer1'], row['answer2']), axis=1)

    # Extract the BLEU scores into separate columns for easier analysis
    data['response_bleu1'] = data['bleu_scores'].apply(lambda x: x['response_bleu1'])
    data['response_bleu2'] = data['bleu_scores'].apply(lambda x: x['response_bleu2'])

    # Display the resulting dataframe with BLEU scores
    data.to_csv('/Users/kremerr/Documents/GitHub/RAGR2/archive/bleu_eval/ragr_bleu_eval_500.csv')
    # data[['question', 'response', 'ragr_answer', 'answer1', 'answer2', 'response_bleu1', 'response_bleu2']].head()
