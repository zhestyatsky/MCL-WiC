EMBEDDINGS = {
    "bert-base": "bert-base-cased",
    "bert-large": "bert-large-cased",
    "roberta-base": "xlm-roberta-base",
    "roberta-large": "xlm-roberta-large"
}

TOP = {
    "linear": "linear",
    "cos_sim": "cosine_similarity"
}

VALID_DESCRIPTIONS = {
    "bert-base-cos_sim-sigmoid",
    "bert-base-cos_sim-relu",
    "bert-base-linear-no_cls",
    "bert-base-linear-cls",
    "bert-large-cos_sim-sigmoid",
    "bert-large-cos_sim-relu",
    "bert-large-linear-no_cls",
    "bert-large-linear-cls",
    "roberta-base-cos_sim-sigmoid",
    "roberta-base-cos_sim-relu",
    "roberta-base-linear-no_cls",
    "roberta-base-linear-cls",
    "roberta-large-cos_sim-sigmoid",
    "roberta-large-cos_sim-relu",
    "roberta-large-linear-no_cls",
    "roberta-large-linear-cls"
}


def get_config(model_description):
    if model_description not in VALID_DESCRIPTIONS:
        raise RuntimeError("Invalid description")

    model_description = model_description.split("-")

    model_config = {"embeddings": EMBEDDINGS["-".join(model_description[:2])], "top": TOP[model_description[2]]}

    feature = model_description[3]

    if feature == "cls":
        model_config["use_cls"] = True
    elif feature == "no_cls":
        model_config["use_cls"] = False
    else:
        model_config["activation"] = feature

    return model_config
