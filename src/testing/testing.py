import json


def _get_predictions(probas, thld):
    return probas > thld


def get_results(probas, thld, dataset):
    ids = dataset.ids
    sentences = dataset.sentences
    results = [{"id": id, "proba": str(proba), "tag": "T" if proba > thld else "F", "sentence1": s1, "sentence2": s2}
               for
               id, proba, (s1, s2) in
               zip(ids, probas, sentences)]
    return results


def write_outputs(results, model_description, output_dir, threshold=0.5):
    outputs_path = output_dir + model_description + '_outputs.json'

    outputs = {"model": model_description, "threshold": threshold, "results": results}

    with open(outputs_path, 'w') as f:
        f.write(json.dumps(outputs))


def convert_to_gold(results):
    return [{"id": sample["id"], "tag": sample["tag"]} for sample in results]
