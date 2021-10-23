from sklearn.metrics import classification_report
import torch, json, argparse
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# FOR NOW THIS CODE WORKS ONLY FOR BERT CLASSIFICATION MODELS

def predict(model, encodings):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = torch.tensor([[int(e > 0) for e in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
    
    # get model output
    output = model(input, masks)
    pred_label = torch.argmax(output[0], dim=1).item()

    return pred_label


def main(test_filepath, model_dir, tokenizer_dir, cls_decoder):

    # get input and labels
    with open(test_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    texts = [i['text'].lower() for i in instances]
    labels = [i['label'] for i in instances]

    # encode input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    encodings = [tokenizer.encode(t) for t in texts]
    
    # predict testset
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    predictions = [cls_decoder[predict(model, e)] for e in tqdm(encodings)]
    print(classification_report(labels, predictions))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filepath",
                        help="""Path to test data file. This file must be in json-format with an object in which
                         the member 'instances' maps to an array of instance objects each containing the members
                          'text', 'post_id' and 'label', mapping to a string, and 'rationale' which maps to an array.""",
                        default="../data/HateXplain-test.json",
                        type=str)
    parser.add_argument("--model_dir",
                        help="Path to the direcory with the trained model",
                        type=str)
    parser.add_argument("--tokenizer",
                        help="Name of tokenizer available in Hugging Face library or path to the directory with a tokenizer configuration",
                        default="bert-base-uncased",
                        type=str)
    parser.add_argument("--cls_encoder",
                        help="string containing JSON, mapping label names to classes of type int",
                        default='{"normal": 0, "hatespeech": 1, "offensive": 2}',
                        type=json.loads)
    
    args = parser.parse_args()

    main(args.test_filepath, args.model_dir, args.tokenizer, args.cls_encoder)