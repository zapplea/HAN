import tensorflow as tf

class Layers:
    def __init__(self,config):
        self.config = config

    def X_input(self):
        X = tf.placeholder(shape=(None,self.config['model']['max_sent_len']),dtype='int32')
        tf.add_to_collection('senti_X_id',X)
        return X

    def attr_Y_input(self):
        attr_Y = tf.placeholder(shape=(None,self.config['model']['attr_num']),dtype='int32')
        tf.add_to_collection('attr_Y',attr_Y)
        return attr_Y

    def senti_Y_input(self):
        senti_Y = tf.placeholder(shape=(None,self.config['model']['attr_num'],self.config['model']['senti_num']),dtype='int32')
        tf.add_to_collection('senti_Y',senti_Y)
        return senti_Y

    def padded_word_mask(self,X_id):
        """

        :param X_id: (batch size, max sent len)
        :return: (batch size, max sent len, word dim)
        """
        X_id = tf.cast(X_id, dtype='float32')
        padding_id = tf.ones_like(X_id, dtype='float32') * self.config['model']['padding_word_index']
        is_padding = tf.equal(X_id, padding_id)
        mask = tf.where(is_padding, tf.zeros_like(X_id, dtype='float32'), tf.ones_like(X_id, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.config['model']['word_dim']])
        return mask

    def word_embedding_table(self):
        table = tf.placeholder(shape=(self.config['model']['vocab_size'], self.config['model']['word_dim']),dtype="float32")
        tf.add_to_collection('table', table)
        embedding = tf.Variable(table)
        return embedding

    def lookup(self,X_id,table,mask):
        """

        :param X_id: (batch size, max sent len)
        :param mask: used to prevent update of padded words
        :return:
        """
        X = tf.nn.embedding_lookup(table, X_id, partition_strategy='mod', name='lookup_table')
        X = tf.multiply(X,mask)
        return X

    def parameter_initializer(self,shape,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
        return init

    def sequence_length(self, X_id):
        """

        :param X_id: (batch size, max sentence len)
        :return:
        """
        padding_id = tf.ones_like(X_id, dtype='int32') * self.config['model']['padding_word_index']
        condition = tf.equal(padding_id, X_id)
        seq_len = tf.reduce_sum(
            tf.where(condition, tf.zeros_like(X_id, dtype='int32'), tf.ones_like(X_id, dtype='int32')),
            axis=1, name='seq_len')
        return seq_len

    def biLSTM(self,X,seq_len,name=''):
        """

        :param X: (batch size, max sent len, word dim)
        :param seq_len: (batch size,)
        :param name:
        :return: (batch size, max sent len, rnn dim)
        """
        with tf.variable_scope('biLSTM'+name, reuse=tf.AUTO_REUSE):
            # define parameters
            fw_cell = tf.contrib.rnn.LSTMCell(
                self.config['model']['biLSTM']['rnn_dim'] / 2
            )
            bw_cell = tf.contrib.rnn.LSTMCell(
                self.config['model']['biLSTM']['rnn_dim'] / 2
            )

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=X,
                sequence_length=seq_len,
                dtype=tf.float32)

            outputs = tf.concat(outputs, axis=-1)
        return outputs

    def sent_attention(self,X, X_id):
        """

        :param X: (batch size, sent len, rnn dim)
        :param X_id: (batch size, sent len)
        :return:(batch size, max sent len)
        """
        X_id = tf.cast(X_id, dtype='float32')
        padding_id = tf.ones_like(X_id, dtype='float32') * self.config['model']['padding_word_index']
        is_padding = tf.equal(X_id, padding_id)
        # (batch size, max sentence len)
        mask = tf.where(is_padding,
                        tf.zeros_like(X_id, dtype='float32'),
                        tf.ones_like(X_id, dtype='float32'))
        # (rnn dim, mlp_dim)
        W = tf.get_variable(name = 'mlp_W',
                            initializer=self.parameter_initializer(shape=(self.config['model']['biLSTM']['rnn_dim'],self.config['model']['mlp_dim']),dtype='float32'))
        tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(W))
        b = tf.get_variable(name='mlp_bias',initializer=tf.zeros(shape=(self.config['model']['mlp_dim'],),dtype='float32'))
        # (batch size, sent len, mlp dim)
        u = tf.nn.tanh(tf.add(tf.tensordot(X,W,axes=[[2],[0]]),b))
        # (mlp dim,)
        uW = tf.get_variable(name='word_context_vec',initializer=self.parameter_initializer(shape=(self.config['model']['mlp_dim']),dtype='float32'))
        tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(uW))
        # (batch size, sent len)
        temp = tf.clip_by_value(tf.reduce_sum(tf.multiply(u,uW),axis=-1),
                                clip_value_min=tf.constant(-self.config['model']['clip_value']),
                                clip_value_max=tf.constant(self.config['model']['clip_value']))
        # (batch size, sent len)
        temp = tf.multiply(temp,mask)
        # (batch size, 1)
        denominator = tf.reduce_sum(temp, axis=-1, keepdims=True)
        # (batch size, max sent len)
        denominator = tf.tile(denominator, multiples=[1, self.config['model']['max_sent_len']])
        # (batch size, max sent len)
        att = tf.truediv(temp, denominator)

        return att

    def sent_repr(self,att,X):
        """

        :param att: (batch size, max sent len)
        :param X: (batch size, max sent len, rnn dim)
        :return: (batch size, max sent len)
        """
        # (batch size, max sent len, rnn_dim)
        att = tf.tile(tf.expand_dims(att,axis=2),multiples=[1,1,self.config['model']['biLSTM']['rnn_dim']])
        # (batch size, max sent len)
        sent_repr = tf.reduce_sum(tf.multiply(att,X),axis=-1)
        return sent_repr

    def score(self,sent_repr):
        """

        :param sent_repr: (batch size, max sent len)
        :return:
        """
        # (attr num , senti num, max sent len)
        A = tf.get_variable(name='attr_matrix',
                            initializer=self.parameter_initializer(shape=(self.config['model']['attr_num'],
                                                                          self.config['model']['senti_num'],
                                                                          self.config['model']['max_sent_len']),
                                                                   dtype='float32'))
        tf.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(A))
        # (attr num, senti num)
        b = tf.get_variable(name='mlp_bias',initializer=tf.zeros(shape=(self.config['model']['attr_num'],self.config['model']['senti_num']),
                                                                 dtype='float32'))
        # (batch size, attr num, senti num)
        score = tf.add(tf.tensordot(sent_repr,A,axes=[[1],[2]]),b)
        return score

    def senti_prediction(self,score):
        """

        :param score: (batch size, attr num, senti num)
        :return:
        """
        # (batch size, attr num, senti num)
        temp = tf.nn.softmax(score,axis=-1)
        senti_pred = tf.where(tf.equal(tf.reduce_max(temp, axis=2, keep_dims=True), temp), tf.ones_like(temp),
                        tf.zeros_like(temp))
        return senti_pred

    def senti_loss(self,logits,labels):
        """

        :param logits: (batch size, attr num, senti num)
        :param labels: (batch size, attr num, senti num)
        :return:
        """
        reg = tf.get_collection('reg')
        loss = tf.reduce_mean(tf.add(
            tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=-1), axis=1),
            tf.reduce_sum(reg)))
        return loss