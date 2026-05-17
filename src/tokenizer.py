def build_tokenizer(docs: list[str]) -> tuple[list[str], int, int]:
    # collects every uniq char across all strings, sorted alpha
    # each chars index in the list is its token id, for ex:
    # a -> 0
    # b -> 1
    uchars = sorted(set("".join(docs)))

    # beginning of sequence token, gets first available id after all chars
    # doubles as stop EOS token; when model predicts BOS generation stops
    BOS = len(uchars)

    #total token count
    vocab_size = len(uchars) + 1
    return uchars, BOS, vocab_size
