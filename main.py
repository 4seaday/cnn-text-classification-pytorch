from utils import Params
from utils import build_tokenizer, build_vocab, padding_sentence, token_to_idx

def main():
    params = Params('config/params.json')
    build_tokenizer() # tokenizer pickle dump
    vocab_size, max_sequence_legnth = build_vocab() # 전체 데이터셋에 대한 vocabulary
    print(vocab_size, max_sequence_legnth)
    # vocabulary에 대해서 train 파일과 max_sequencelength를 반환받아야 함.


    return 0

if __name__ == '__main__':
    main()
    pass