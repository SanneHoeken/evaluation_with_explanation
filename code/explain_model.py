from captum.attr import ShapleyValueSampling, DeepLift, GuidedBackprop, InputXGradient, Saliency, FeatureAblation, IntegratedGradients
from transformers import BertTokenizer, BertForSequenceClassification
import torch, json, argparse
import numpy as np
from tqdm import tqdm

# INSPIRED BY: XAI Benchmark (https://github.com/copenlu/xai-benchmark) and Captum Tutorials (https://captum.ai/tutorials/)
# FOR NOW THIS CODE WORKS ONLY FOR BERT CLASSIFICATION MODELS

class ShapleyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ShapleyModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(input, attention_mask=mask)[0]

class GradientModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GradientModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(inputs_embeds=input, attention_mask=mask)[0]


def get_ablator(model, ablator_name):
    # to implement: GradientShap, GuidedGradCam, KernelShap, LimeBase, LRP, NoiseTunnel, FeaturePermutation, Occlusion
    ablators = {'deep_lift': DeepLift(model), 'feature_ablation': FeatureAblation(model),
            'guided_backprop': GuidedBackprop(model), 'input_x_gradient': InputXGradient(model),
            'integrated_gradients': IntegratedGradients(model), 
                'saliency': Saliency(model), 'shapley_value': ShapleyValueSampling(model)}
    ablator = ablators[ablator_name]
    return ablator


def explain_input(input, tokenizer_dir, model_dir, ablator_name, exclude_cls, summarize):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare inputs
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    encodings = tokenizer.encode(input)
    masks = torch.tensor([[int(i > 0) for i in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
    
    # prepare model
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    
    # get model output
    output = model(input, masks)
    pred_label = torch.argmax(output[0], dim=1)

    if pred_label.item() in exclude_cls:
            return []
    
    if ablator_name == 'random':
        return [np.random.rand() for _ in range(len(encodings))]
    
    if ablator_name == 'attention':
        attention_maps = np.array([x.detach().cpu().numpy()[0] for x in output[1]])
        # select last encoder block map and average over heads, select [CLS] only
        attention_maps = np.mean(attention_maps[-1], axis=0)[:1, :]
        attentions = attention_maps / np.max(attention_maps)
        return attentions[0].tolist()
    elif ablator_name == 'shapley_value':
      model = ShapleyModelWrapper(model)
    else:
      model = GradientModelWrapper(model)
      input = model.model.bert.embeddings(input)

    # get attributions
    ablator = get_ablator(model, ablator_name)
    attributions = ablator.attribute(input, target=pred_label, additional_forward_args=masks)[0]
    if ablator_name != 'shapley_value':
        if summarize == 'l2':
            attributions = attributions.norm(p=1, dim=-1).squeeze(0)
        else:
            # summarize attributions by taking the mean
            attributions = attributions.mean(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)

    return attributions.tolist()


def main(input_filepath, output_filepath, model_dir, tokenizer_dir, ablator_name, exclude_cls, summarize):

    # GET INPUT
    with open(input_filepath, 'r') as infile:
        instances = json.load(infile)['instances']
    texts = [i['text'].lower() for i in instances]
    post_ids = [i['post_id'] for i in instances]

    # COMPUTE ATTRIBUTIONS
    explanations = []
    for input in tqdm(texts):
        attributions = explain_input(input, tokenizer_dir, model_dir, ablator_name, exclude_cls, summarize)
        explanations.append(attributions)

    # SAVE EXPLANATIONS
    explanations_file = {"instances": []}
    for id, text, rat in zip(post_ids, texts, explanations):
        instance = {"post_id": id, "text": text, "explanation": rat}
        explanations_file["instances"].append(instance)
    with open(output_filepath, 'w') as outfile:
        json.dump(explanations_file, outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filepath",
                        help="""Path to test data file. This file must be in json-format with an object in which
                         the member 'instances' maps to an array of instance objects each containing the members
                          'text', 'post_id' and 'label', mapping to a string, and 'rationale' which maps to an array.""",
                        default="../data/HateXplain-test.json",
                        type=str)
    parser.add_argument("--explanations_filepath",
                        help="Path to json-file (in existing directory) where explanations for test data predictions to be generated should be stored.",
                        default="../data/explanations/HateXplain-test-random-explanations.json",
                        type=str)
    parser.add_argument("--model_dir",
                        help="Path to the direcory with the trained model",
                        type=str)
    parser.add_argument("--tokenizer",
                        help="Name of tokenizer available in Hugging Face library or path to the directory with a tokenizer configuration",
                        default="bert-base-uncased",
                        type=str)
    parser.add_argument("--explanation_method",
                        help="name of explanation method to use",
                        choices=['random', 'attention', 'deep_lift', 'feature_ablation', 'guided_backprop', 'input_x_gradient', 'integrated_gradients', 'saliency', 'shapley_value'],
                        default='random',
                        type=str)
    parser.add_argument("--exclude_cls",
                        help="class(es) (encoded to integers) to exclude for explaining",
                        nargs='*',
                        default=[],
                        type=int)
    parser.add_argument("--gradient_aggregation",
                        help="if explanation method is gradient-based: way to aggregate attribution values",
                        choices=['mean', 'l2'],
                        default='mean',
                        type=str)

    args = parser.parse_args()

    main(args.test_filepath, args.explanations_filepath, args.model_dir, args.tokenizer, args.explanation_method, args.exclude_cls, args.gradient_aggregation)
