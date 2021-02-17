def _find_nth_word_start_end_indices_in_sentence(sentence, n):
    words = sentence.split(' ')
    start_idx = n
    for word in words[:n]:
        start_idx += len(word)
    return start_idx, start_idx + len(words[n])


def get_word_start_end_in_sentence(row):
    first_word_pos, second_word_pos = [int(idx) for idx in row['word_indices'].split('-')]
    start1, end1 = _find_nth_word_start_end_indices_in_sentence(row['sentence1'], first_word_pos)
    start2, end2 = _find_nth_word_start_end_indices_in_sentence(row['sentence2'], second_word_pos)
    return (start1, end1), (start2, end2)
