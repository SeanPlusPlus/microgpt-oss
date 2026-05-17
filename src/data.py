import os
import random

random.seed(42)

# file fetch and load

def load_docs() -> list[str]:
    if not os.path.exists("input.txt"):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(url, "input.txt")
    docs = [line.strip() for line in open("input.txt") if line.strip()]
    random.shuffle(docs)
    return docs
