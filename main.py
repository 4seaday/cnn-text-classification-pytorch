from utils import Params
from utils import build_tokenizer, build_vocab, padding_sentence, token_to_idx

def main():
    params = Params('config/params.json')
    tokenizer = build_tokenizer()
    vocab = build_vocab(sentence_list) # 전체 데이터셋에 대한 vocabulary

    if params.mode == 'train':
        sentence_list = padding_sentence(sentence_list)
        sentence_list = token_to_idx(sentence_list, vocab)

    return 0

if __name__ == '__main__':
    main()
    pass