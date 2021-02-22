import json


def _get_predictions(probas, thld):
    return probas > thld


def get_outputs(probas, thld, ids):
    preds = _get_predictions(probas, thld)
    probas_output = [{"id": id, "proba": proba} for id, proba in zip(ids, probas)]
    preds_output = [{"id": id, "tag": "T" if pred else "F"} for id, pred in zip(ids, preds)]
    return probas_output, preds_output


def dump_outputs(probas, thld, ids, output_dir):
    probas_path = output_dir + 'proba.json'
    preds_path = output_dir + 'preds.json'

    probas_output, preds_output = get_outputs(probas, thld, ids)

    with open(probas_path, 'w') as f:
        f.write(json.dumps(probas_output))

    with open(preds_path, 'w') as f:
        f.write(json.dumps(preds_output))
