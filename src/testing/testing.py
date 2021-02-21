import json


def get_predictions(proba, thld):
    return proba > thld


def generate_output(predictions, ids, path):
    result = [{"id": id, "tag": "T"} if pred else {"id": id, "tag": "F"} for id, pred in
              zip(ids, predictions)]
    with open(path, 'w') as f:
        f.write(json.dumps(result))
