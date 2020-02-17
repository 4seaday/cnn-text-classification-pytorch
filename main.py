from utils import Params
from utils import build_tokenizer, padding_sentence, make_iter
from train import Trainer

def main():
    build_tokenizer() # tokenizer pickle dump

    params = Params('config/params.json')
    print('start build tokenizer')
    print('finish build tokenizer')
    # max_seq, vocab_size = build_vocab() # 전체 데이터셋에 대한 vocabulary

    if params.mode == 'train':
        inputs, labels = padding_sentence(params)
        data_loader = make_iter(params, inputs, labels)
        trainer = Trainer(params)
        print('lets train')
        trainer.train(data_loader)

if __name__ == '__main__':
    main()
    pass