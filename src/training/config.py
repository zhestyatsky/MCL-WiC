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
    "roberta-large-linear-cls",
    "bert-base-cos_sim-sigmoid-only_wic",
    "bert-base-cos_sim-relu-only_wic",
    "bert-base-linear-no_cls-only_wic",
    "bert-base-linear-cls-only_wic",
    "bert-large-cos_sim-sigmoid-only_wic",
    "bert-large-cos_sim-relu-only_wic",
    "bert-large-linear-no_cls-only_wic",
    "bert-large-linear-cls-only_wic",
    "roberta-base-cos_sim-sigmoid-only_wic",
    "roberta-base-cos_sim-relu-only_wic",
    "roberta-base-linear-no_cls-only_wic",
    "roberta-base-linear-cls-only_wic",
    "roberta-large-cos_sim-sigmoid-only_wic",
    "roberta-large-cos_sim-relu-only_wic",
    "roberta-large-linear-no_cls-only_wic",
    "roberta-large-linear-cls-only_wic",
    "bert-large-cos_sim-relu-use_default_datasets"
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

    if len(model_description) == 4:
        model_config["only_wic"] = False
        model_config["use_default_datasets"] = False
        return model_config

    dataset_feature = model_description[4]

    if dataset_feature == "only_wic":
        model_config["only_wic"] = True
        model_config["use_default_datasets"] = False
    elif dataset_feature == "use_default_datasets":
        model_config["only_wic"] = False
        model_config["use_default_datasets"] = True

    return model_config
