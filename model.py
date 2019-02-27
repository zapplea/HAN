import tensorflow as tf
from layers import Layers

class HAN:
    def __init__(self,config):
        self.config = config
        self.layers = Layers(config)

    def build_HAN_net(self):
        X_id = self.layers.X_input()
        senti_Y = self.layers.senti_Y_input()
        table = self.layers.word_embedding_table()
        mask = self.layers.padded_word_mask(X_id)
        X = self.layers.lookup(X_id, table, mask)
        seq_len = self.layers.sequence_length(X_id)
        sent_repr_ls = []
        for i in range(self.config['model']['sentAtt_num']):
            name = '_layer%d'%i
            X = self.layers.biLSTM(X,seq_len,name)
            graph = tf.get_default_graph()
            tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
                graph.get_tensor_by_name('biLSTM%s/bidirectional_rnn/fw/lstm_cell/kernel:0'%name)))
            tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
                graph.get_tensor_by_name('biLSTM%s/bidirectional_rnn/bw/lstm_cell/kernel:0'%name)))
            sent_att = self.layers.sent_attention(X,X_id)
            # (batch size, max sent len)
            sent_repr = self.layers.sent_repr(sent_att, X)
            sent_repr_ls.append(sent_repr)
        # (batch size, sentAtt_num * max sent len)
        sent_repr = tf.concat(sent_repr_ls,axis=1)
        senti_score = self.layers.score(sent_repr)
        pred = self.layers.senti_prediction(senti_score)
        loss = self.layers.senti_loss(senti_score, senti_Y)
        train_step = tf.train.AdamOptimizer(self.config['model']['lr']).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        return {'loss': loss, 'pred': pred, 'graph': tf.get_default_graph(), 'train_step': train_step, 'saver': saver}