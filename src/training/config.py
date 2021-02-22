EMBEDDINGS = {
    "roberta-large": "xlm-roberta-large",
    "bert-large": "bert-large-cased",
    "bert-base": "bert-base-cased"
}

TOP = {
    "linear": "linear",
    "cos_sim": "cosine_similarity"
}

VALID_DESCRIPTIONS = {
    "roberta-large-linear-no_cls",
    "roberta-large-linear-cls",
    "bert-base-linear-no_cls",
    "bert-base-linear-cls",
    "bert-large-linear-no_cls",
    "bert-large-linear-cls",
    "roberta-large-cos_sim-sigmoid",
    "roberta-large-cos_sim-relu",
    "bert-base-cos_sim-sigmoid",
    "bert-base-cos_sim-relu",
    "bert-large-cos_sim-sigmoid",
    "bert-large-cos_sim-relu",
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
