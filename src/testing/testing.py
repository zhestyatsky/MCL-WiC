import json

from src.util.util import get_accuracy


def get_results(probas, thld, dataset):
    ids = dataset.ids
    sentences = dataset.sentences
    gold_labels = dataset.labels

    test_accuracy = get_accuracy(gold_labels, probas, thld)

    results = [
        {"id": id, "sentence1": s1, "sentence2": s2, "proba": str(proba.item()), "tag": "T" if proba > thld else "F",
         "gold_tag": "T" if gold_label == 1 else "F"}
        for
        id, proba, (s1, s2), gold_label in
        zip(ids, probas, sentences, gold_labels)]

    return {"test_accuracy": str(test_accuracy), "results": results}


def write_outputs(results, model_description, output_dir, validation_accuracy, threshold):
    outputs_path = output_dir + model_description + '_outputs.json'

    test_accuracy = results["test_accuracy"]
    outputs = {"model": model_description, "validation_accuracy": str(validation_accuracy),
               "test_accuracy": test_accuracy, "threshold": str(threshold), "results": results["results"]}

    with open(outputs_path, 'w') as f:
        f.write(json.dumps(outputs, indent=4))

    return outputs
