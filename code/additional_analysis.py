import numpy as np
import json, csv, argparse
from transformers import BertTokenizer
from collections import defaultdict
from explanation_plausibility import gold_to_token_level

def error_per_token(tokenized_texts, truth, preds, output_file):

    token_error_dict = defaultdict(dict)
    list_token_dicts = []

    for tokenized_text, gold, pred in zip(tokenized_texts, truth, preds):
        if gold and pred:
            pred_normalized = [(p-min(pred))/(max(pred)-min(pred)) for p in pred]
            for t, g, p in zip(tokenized_text, gold, pred_normalized):
                if 'truth' not in token_error_dict[t]:
                    token_error_dict[t]['truth'] = []
                    token_error_dict[t]['pred'] = []
                    token_error_dict[t]['delta'] = []
                token_error_dict[t]['truth'].append(g)
                token_error_dict[t]['pred'].append(p)
                token_error_dict[t]['delta'].append(g-p)
                
    for token in token_error_dict:
        token_freq = len(token_error_dict[token]['delta'])
        mean_error = np.mean(token_error_dict[token]['delta'])
        mean_truth = np.mean(token_error_dict[token]['truth'])
        mean_pred = np.mean(token_error_dict[token]['pred'])

        instance = {'token': token, 'mean_error': mean_error, 'token_freq': token_freq, 'mean_gold': mean_truth, 'mean_explanation': mean_pred}
        list_token_dicts.append(instance)

    keys = list_token_dicts[0].keys()
    with open(output_file, 'w')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_token_dicts)

    
def main(test_filepath, explanations_filepath, tokenizer_dir, output_file):

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
    
    encodings = [tokenizer.encode(t) for t in texts]
    tokenized_texts = [[tokenizer.ids_to_tokens[e] for e in encoding] for encoding in encodings]
    error_per_token(tokenized_texts, token_level_gold, explanations, output_file)
 

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
    parser.add_argument("--output_filepath",
                        help="Path to csv-file to store the output",
                        default="../data/plausibility_error_analysis.csv",
                        type=str)
    parser.add_argument("--tokenizer",
                        help="Name of tokenizer available in Hugging Face library or path to the directory with a tokenizer configuration",
                        default="bert-base-uncased",
                        type=str)
    
    args = parser.parse_args()

    main(args.test_filepath, args.explanations_filepath, args.tokenizer, args.output_filepath)