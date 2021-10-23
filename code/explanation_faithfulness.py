from sklearn.metrics import auc
import torch, json, argparse
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np

# INSPIRED BY: ERASER Benchmark (https://github.com/jayded/eraserbenchmark) and XAI Benchmark (https://github.com/copenlu/xai-benchmark)
# FOR NOW THIS CODE WORKS ONLY FOR BERT CLASSIFICATION MODELS

def predict(encodings, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = torch.tensor([[int(i > 0) for i in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
    
    # get model output
    output = model(input, masks)
    pred_label = torch.argmax(output[0], dim=1).item()
    pred_probs = torch.softmax(output[0], dim=1)[0]

    return pred_label, pred_probs


def preds_with_ablations(input, mask_id, original_pred, explanation, thresholds, model):

    pred_scores = []
    
    # sort input with original index based on explanation score
    input_with_index = [[i, t, e] for i, (t, e) in enumerate(zip(input, explanation))]
    sorted_input = sorted(input_with_index, key=lambda lis: lis[2], reverse=True)

    for thres in thresholds:
        n_ablation = int(thres * len(sorted_input))

        # replace tokens to ablate by mask token id
        for i, input in enumerate(sorted_input):
            if i < n_ablation:
                input[1] = mask_id

        # restore original order
        ablated_input = sorted(sorted_input, key=lambda lis: lis[0])
        ablated_input = [a[1] for a in ablated_input]
        
        # predict ablated input
        pred_label, pred_probs = predict(ablated_input, model)
        prob_original_pred = pred_probs[original_pred].item()
        pred_scores.append((pred_label, prob_original_pred))
    
    return pred_scores


def calculate_AUC_TCPD(preds_ablations, original_preds, thresholds):
    # mean difference in probability compared to original prediction
    dataset_scores = []
    for text_preds, original_pred in zip(preds_ablations, original_preds):
        text_scores = []
        for thres_pred in text_preds:
            delta = original_pred[1] - thres_pred[1]
            text_scores.append(delta)
        dataset_scores.append(text_scores)

    avg_dataset_scores = np.mean(dataset_scores, axis=0)
    auc_score = auc(thresholds, avg_dataset_scores)
    
    return auc_score     


def compute_faithfulness(encodings, explanations, labels, model, mask_id):
    
    original_preds = []
    ablated_preds = []
    gold_labels = []
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for i, encoding in tqdm(enumerate(encodings)):
        
        explanation = list(explanations[i])
        if len(explanation) == 0:
            continue

        pred_label, pred_probs = predict(encoding, model)
        prob_original_pred = pred_probs[pred_label].item()
        preds_ablations = preds_with_ablations(encoding, mask_id, pred_label, explanation, thresholds, model)
        
        original_preds.append((pred_label, prob_original_pred))
        ablated_preds.append(preds_ablations)
        gold_labels.append(labels[i])

    score = calculate_AUC_TCPD(ablated_preds, original_preds, thresholds)
        
    return score
     

def main(test_filepath, explanations_filepath, model_dir, tokenizer_dir, cls_encoder):

    # get input, labels and gold rationales
    with open(test_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    texts = [i['text'].lower() for i in instances]
    labels = [cls_encoder[i['label']] for i in instances]
    
    # get explantions
    with open(explanations_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    explanations = [i['explanation'] for i in instances]

    # encode input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    mask_id = tokenizer.mask_token_id
    encodings = [tokenizer.encode(t) for t in texts]
    
    # initialize model
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    # compute faithfulness
    faithfulness = compute_faithfulness(encodings, explanations, labels, model, mask_id)
    print(faithfulness)


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
    
    args = parser.parse_args()

    main(args.test_filepath, args.explanations_filepath, args.model_dir, args.tokenizer, args.cls_encoder)
