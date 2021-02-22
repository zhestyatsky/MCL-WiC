import json


def _get_predictions(probas, thld):
    return probas > thld


def get_results(probas, thld, ids):
    results = [{"id": id.item(), "proba": proba, "tag": "T" if proba > thld else "F"} for id, proba in
               zip(ids, probas)]
    return results


def write_outputs(results, model_description, output_dir, threshold=0.5):
    outputs_path = output_dir + model_description + '_outputs.json'

    outputs = {"model": model_description, "threshold": threshold, "results": results}

    with open(outputs_path, 'w') as f:
        f.write(json.dumps(outputs))


def convert_to_gold(results):
    return [{"id": sample["id"], "tag": sample["tag"]} for sample in results]
