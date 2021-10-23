# Evaluation of NLP models with Explainability

This is the repository for the paper "Towards new evaluation standards for Natural Language Processing models in response to an increasing demand for explainability", written for the master course Advanced NLP Seminar, which is part of the Humanities Research master at VU University Amsterdam.
October 2021.

### Project

In this project I proposed new evaluation standards for Natural Language Processing models in which explainability is integrated. The code base in this repository serves these new evaluation practices.

## Getting started

### Requirements

This codebase is written entirely in Python 3.7. requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:

```
pip install -r requirements.txt
```

### Structure

The following list describes the folders in the project:

- **/code**: contains all code of this project
- **/data**: contains data used for project illustration
- **/models**: containing placeholders for models used in this project

## Using the main programs

To run the main programs, change the working directory of the execution environment to the **/code** directory. Examples of running each of the four components of the proposed new evaluation standards are:

1. **Explain model**

  ```
    python explain_model.py --test_filepath "../data/HateXplain-test.json" --explanations_filepath "../data/explanations/Bert-HateXplain-test-input_x_gradient-l2-explanations.json" --model_dir "../models/Bert_ft_HateXplain" --explanation_method "input_x_gradient" --exclude_cls 0 --gradient_aggregation "l2"
  ```

2. **Prediction Performance**

  ```
    python prediction_performance.py --test_filepath "../data/HateXplain-test.json" --model_dir "../models/Bert_ft_HateXplain" --cls_encoder {"normal": 0, "hatespeech": 1, "offensive": 2}
  ```

3. **Explanation Faithfulness**

  ```
    python explanation_faithfulness.py --test_filepath "../data/HateXplain-test.json" --explanations_filepath "../data/explanations/Bert-HateXplain-test-saliency-explanations.json" --model_dir "../models/Bert_ft_HateXplain"
  ```

4. **Explanation Plausibility**

  ```
    python explanation_plausiblity.py --test_filepath "../data/HateXplain-test.json" --explanations_filepath "../data/explanations/HateXplain-test-random-explanations.json"
  ```

**Running the whole pipeline**

  All components can also be run at once, by calling for example:

  ```
    python evaluate_model.py --test_filepath "../data/HateXplain-test.json" --explanations_filepath "../data/explanations/Bert-HateXplain-test-shapley_value-explanations.json" --model_dir "../models/Bert_ft_HateXplain" --explanation_method "shapley" --exclude_cls 0
  ```
  
  Or in case of already generated explanations, this program can be run by calling for example:

  ```
    python evaluate_model.py --test_filepath "../data/HateXplain-test.json" --explanations_filepath "../data/explanations/HateBert-HateXplain-test-shapley_value-explanations.json" --model_dir "../models/HateBert_ft_HateXplain" --tokenizer "../models/HateBert-pt"
  ```

## Author
- Sanne Hoeken (student number: 2710599)
