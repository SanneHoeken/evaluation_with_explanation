import numpy as np
from sklearn.metrics import average_precision_score
import json, argparse
from transformers import BertTokenizer
from tqdm import tqdm

# INSPIRED BY: ERASER Benchmark (https://github.com/jayded/eraserbenchmark) and XAI Benchmark code (https://github.com/copenlu/xai-benchmark)
# FOR NOW THIS CODE WORKS ONLY FOR BERT CLASSIFICATION MODELS

def gold_to_token_level(gold_explanations, texts, tokenizer, special_tokens_idx):
    
    token_level_explanations = []
    
    for text, gold_explanation in zip(texts, gold_explanations):
        
        if gold_explanation:
            token_explanation = []
            for word, exp in zip(text.split(' '), gold_explanation):
                for _ in range(len(tokenizer.tokenize(word))):
                    token_explanation.append(exp)
            for idx in special_tokens_idx:
                if idx < 0:
                    idx = len(token_explanation) + idx + 1
                token_explanation.insert(idx, 0)
            token_level_explanations.append(token_explanation)
        else:
            token_level_explanations.append(gold_explanation)
    
    return token_level_explanations


def compute_plausibility(truth, preds):
   
    aps = []
    for t, p in tqdm(zip(truth, preds)):
        if t and p:
            aps.append(average_precision_score(t, p))
    map = np.mean(aps)

    return map   


def main(test_filepath, explanations_filepath, tokenizer_dir):

    # get input, labels and gold rationales
    with open(test_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    texts = [i['text'].lower() for i in instances]
    gold_explanations = [i['rationale'] for i in instances]
    
    # get explantions
    with open(explanations_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    explanations = [i['explanation'] for i in instances]

    # convert gold explanations and compute plausibility
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    special_tokens_idx = [0, -1] # postitions of [CLS] and [SEP] for Bert
    token_level_gold = gold_to_token_level(gold_explanations, texts, tokenizer, special_tokens_idx)
    plausibility_score = compute_plausibility(token_level_gold, explanations)
    print(plausibility_score)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filepath",
                        help="""Path to test data file. This file must be in json-format with an object in which
                         the member 'instances' maps to an array of instance objects each containing the members
                          'text', 'post_id' and 'label', mapping to a string, and 'rationale' which maps to an array.""",
                        default="../data/HateXplain-test.json",
                        type=str)
    parser.add_argument("--explanations_filepath",
                        help="""Path to file with explanations for test data predictions. This file must be in json-format with an object in which
                         the member 'instances' maps to an array of instance objects each containing the members
                          'text' and 'post_id' mapping to a string, and 'explanation' which maps to an array.""",
                        default="../data/explanations/HateXplain-test-random-explanations.json",
                        type=str)
    parser.add_argument("--tokenizer",
                        help="Name of tokenizer available in Hugging Face library or path to the directory with a tokenizer configuration",
                        default="bert-base-uncased",
                        type=str)
    
    args = parser.parse_args()

    main(args.test_filepath, args.explanations_filepath, args.tokenizer)