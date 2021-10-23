import torch, json, argparse
import os.path
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from prediction_performance import predict
from explain_model import explain_input
from explanation_faithfulness import compute_faithfulness
from explanation_plausibility import compute_plausibility, gold_to_token_level

# FOR NOW THIS CODE WORKS ONLY FOR BERT CLASSIFICATION MODELS

def main(test_filepath, model_dir, tokenizer_dir, cls_encoder, 
        ablator_name, explanations_filepath, exclude_cls, summarize):

    print('\nLoading data and model...\n')
    # GET DATA FROM TESTSET
    with open(test_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    texts = [i['text'].lower() for i in instances]
    post_ids = [i['post_id'] for i in instances]
    labels = [i['label'] for i in instances]
    gold_explanations = [i['rationale'] for i in instances]

    # INITIALIZE MODEL AND TOKENIZER AND ENCODE INPUT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    mask_id = tokenizer.mask_token_id
    encodings = [tokenizer.encode(t) for t in texts]
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    # COMPUTE PREDICTIONS
    cls_decoder = {value: key for key, value in cls_encoder.items()}
    print('\nPredicting test instances...\n')
    predictions = [cls_decoder[predict(model, e)] for e in tqdm(encodings)]
    print('\nPerformance:\n', classification_report(labels, predictions))

    # GET EXPLANATIONS
    if os.path.isfile(explanations_filepath):
        with open(explanations_filepath, 'r') as infile:
            instances = json.load(infile)['instances']
        explanations = [i['explanation'] for i in instances]
    else:
        # COMPUTE EXPLANATIONS
        print('\nComputing explanations...\n')
        explanations = []
        for input in tqdm(texts):
            attributions = explain_input(input, tokenizer_dir, model_dir, ablator_name, exclude_cls, summarize)
            explanations.append(attributions)

        # SAVE EXPLANATIONS
        print('\nSaving explanations...')
        explanations_file = {"instances": []}
        for id, text, rat in zip(post_ids, texts, explanations):
            instance = {"post_id": id, "text": text, "explanation": rat}
            explanations_file["instances"].append(instance)
        with open(explanations_filepath, 'w') as outfile:
            json.dump(explanations_file, outfile)
    
    # COMPUTE FAITHFULNESS
    print('\nComputing faitfulness scores...\n')
    labels = [cls_encoder[label] for label in labels]
    faithfulness = compute_faithfulness(encodings, explanations, labels, model, mask_id)
    print('\nFaithfulness:\n', faithfulness)

    # COMPUTE PLAUSIBILITY
    print('\nComputing plausibility scores...\n')
    special_tokens_idx = [0, -1] # postitions of [CLS] and [SEP] for Bert, same can be done for padding
    token_level_gold = gold_to_token_level(gold_explanations, texts, tokenizer, special_tokens_idx)
    plausibility_score = compute_plausibility(token_level_gold, explanations)
    print('\nPlausibility:\n', plausibility_score)


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
                          'text' and 'post_id' mapping to a string, and 'explanation' which maps to an array.
                          OR
                          Path to json-file (in existing directory) where explanations for test data predictions to be generated should be stored.""",
                        default="../data/explanations/HateXplain-test-random-explanations.json",
                        type=str)
    parser.add_argument("--model_dir",
                        help="Path to the direcory with the trained model",
                        type=str)
    parser.add_argument("--tokenizer",
                        help="Name of tokenizer available in Hugging Face library or path to the directory with a tokenizer configuration",
                        default="bert-base-uncased",
                        type=str)
    parser.add_argument("--cls_encoder",
                        help="string containing JSON mapping label names to classes of type int",
                        default='{"normal": 0, "hatespeech": 1, "offensive": 2}',
                        type=json.loads)
    parser.add_argument("--explanation_method",
                        help="if no explanations filepath is provided: name of explanation method to use",
                        choices=['random', 'attention', 'deep_lift', 'feature_ablation', 'guided_backprop', 'input_x_gradient', 'integrated_gradients', 'saliency', 'shapley_value'],
                        default='random',
                        type=str)
    parser.add_argument("--exclude_cls",
                        help="if no explanations filepath is provided: class(es) (encoded to integers) to exclude for explaining",
                        nargs='*',
                        default=[],
                        type=int)
    parser.add_argument("--gradient_aggregation",
                        help="if no explanations filepath is provided and explanation method is gradient-based: way to aggregate attribution values",
                        choices=['mean', 'l2'],
                        default='mean',
                        type=str)

    args = parser.parse_args()

    main(args.test_filepath, args.model_dir, args.tokenizer, args.cls_encoder, 
        args.explanation_method, args.explanations_filepath, args.exclude_cls, args.gradient_aggregation)