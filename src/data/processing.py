def get_sentences(df):
    return [(s1, s2) for s1, s2 in zip(df["sentence1"], df["sentence2"])]


def get_word_ranges(df):
    return [((int(s1), int(e1)), (int(s2), int(e2))) for s1, e1, s2, e2 in
            zip(df["start1"], df["end1"], df["start2"], df["end2"])]


def get_labels(df):
    return df["tag"].tolist()


def get_ids(df):
    return df["id"].tolist()
