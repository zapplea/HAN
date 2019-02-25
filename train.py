import tensorflow as tf
import numpy as np
from metrics import Metrics
from pathlib import Path

class SentiTrain:
    def __init__(self,config,df):
        self.config = config
        self.df = df
        self.mt = Metrics(config['train'])

        dir_ls = ['report_filePath', 'sr_path']
        for name in dir_ls:
            path = Path(self.config['train'][name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.config['train']['report_filePath'] = self.config['train']['report_filePath'] + '/base_report_reg%s_lr%s.info' % \
                                               (str(self.config['model']['reg_rate']), str(self.config['model']['lr']))
        self.config['train']['sr_path'] = self.config['train']['sr_path'] + '/base_ckpt_reg%s_lr%s/model.ckpt'%\
                                          (str(self.config['model']['reg_rate']), str(self.config['model']['lr']))

        self.outf = open(self.config['train']['report_filePath'], 'w+')

    def train(self,model):
        graph = model['graph']
        early_stop_count = 0
        best_f1_score = 0
        with graph.as_default():
            table = tf.get_collection('table')[0]
            senti_X = tf.get_collection('senti_X_id')[0]
            senti_Y = tf.get_collection('senti_Y')[0]
            senti_loss = model['loss']
            senti_pred = model['pred']
            train_step = model['train_step']
            saver = model['saver']

            init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph,config=config) as sess:
            sess.run(init, feed_dict={table:self.df.table})
            self.mt.report('\n#####sentiment metrics#####\n', self.outf, 'report')
            for epoch in range(self.config['train']['epoch_num']):
                dataset = self.df.data_generator('train')
                print('epoch: %d'%epoch)
                print('Start Training ...')
                for _, senti_labels_data, sentences_data in dataset:
                    sess.run(train_step,feed_dict={senti_Y:senti_labels_data,senti_X:sentences_data})
                print('Done!')

                if epoch%self.config['train']['mod']==0:
                    dataset = self.df.data_generator('val')
                    senti_loss_vec = []
                    senti_TP_vec = []
                    senti_FP_vec = []
                    senti_FN_vec = []
                    print('Start Testing ...')
                    for _, senti_labels_data, sentences_data in dataset:
                        senti_loss_value,senti_pred_value = sess.run([senti_loss,senti_pred],feed_dict={senti_X:sentences_data,senti_Y:senti_labels_data})

                        senti_labels_data = self.mt.caliberate(senti_labels_data)
                        senti_pred_data = self.mt.caliberate(senti_pred_value)

                        TP_data = self.mt.TP(senti_labels_data, senti_pred_data)
                        FP_data = self.mt.FP(senti_labels_data, senti_pred_data)
                        FN_data = self.mt.FN(senti_labels_data, senti_pred_data)
                        senti_TP_vec.append(TP_data)
                        senti_FP_vec.append(FP_data)
                        senti_FN_vec.append(FN_data)
                        senti_loss_vec.append(senti_loss_value)
                    print('Done!')
                    TP_vec = np.sum(senti_TP_vec, axis=0)
                    FP_vec = np.sum(senti_FP_vec, axis=0)
                    FN_vec = np.sum(senti_FN_vec, axis=0)
                    loss_value = np.mean(senti_loss_vec)
                    print('loss value: ',loss_value)
                    self.mt.report('\nepoch:%d\n'%epoch, self.outf, 'report')
                    self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                    _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,
                                                                outf=self.outf,
                                                                id_to_aspect_dic=self.df.id_to_aspect_dic, mod='senti')
                    print(_f1_score)
                    if best_f1_score >= _f1_score:
                        early_stop_count += 1
                    else:
                        early_stop_count = 0
                        best_f1_score = _f1_score
                        print('save path: %s' % self.config['train']['sr_path'])
                        saver.save(sess, self.config['train']['sr_path'])
                    if early_stop_count >= self.config['train']['early_stop_limit']:
                        break