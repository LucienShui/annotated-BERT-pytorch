from bert_pytorch.__main__ import get_args, train
import sys
from bert_pytorch.dataset.vocab import WordVocab

embed_corpus: str = "Welcome to the \\t the jungle\nI can stay \\t here all night"
corpus_path: str = 'corpus.embed.txt'
vocab_path: str = 'vocab.embed'


def init_fake_data():
    with open(corpus_path, 'w') as f:
        f.write(embed_corpus)


def build_vocab():
    with open(corpus_path) as f:
        vocab = WordVocab(f)
    vocab.save_vocab(vocab_path)


def main():
    init_fake_data()

    build_vocab()

    sys.argv.extend(['-c', corpus_path, '-v', vocab_path, '-o', 'bert.model'])  # mock 传参
    args = get_args()

    train(args)


if __name__ == '__main__':
    main()
