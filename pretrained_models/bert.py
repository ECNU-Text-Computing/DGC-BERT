from pretrained_models.base_bert import BaseBert


class BERT(BaseBert):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, **kwargs):
        super(BERT, self).__init__(vocab_size, embed_dim, num_class, pad_index, pad_size, word2vec, keep_prob,
                                   hidden_size, model_path, **kwargs)
        self.model_name = 'BERT'
        if kwargs['mode']:
            self.model_name += '_' + kwargs['mode']
            print(self.model_name)

        print(self.fc)

