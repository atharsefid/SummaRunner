import tensorflow as tf
from tensorflow.python import debug as tfdbg
tf.autograph.set_verbosity(0)
import sys
from params import *
from batch_data_utils import *

current_time = str(time.time())
log_dir = 'logs/' + current_time
session_conf = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
sess = tf.compat.v1.Session(config=session_conf)


class SummaRuNNer(object):
    def __init__(self, vocabulary_size, embedding_size, pretrained_embedding):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self.batch_size = None
        self.sent_len = max_inp_seq_len
        self.hidden_size = 200
        self.doc_len = doc_size
        with tf.compat.v1.variable_scope('inputs'):
            self.x = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, self.doc_len,self.sent_len ], name="x_input")
            self.y = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, self.doc_len])
            self.sequence_length = tf.reduce_sum(tf.sign(self.x), axis=1)
            self.doc_length = tf.reduce_sum(tf.sign(self.sequence_length), axis=0)
            
        with tf.compat.v1.variable_scope('embedding_layer'):
            self.embeddings = tf.compat.v1.get_variable(name='embeddings',
                                              initializer=tf.convert_to_tensor(self.pretrained_embedding),
                                              dtype=tf.float32)
            
                    
            document_placeholder_flat = tf.reshape(self.x, [-1])
            print('flat imput:', document_placeholder_flat)
            document_word_embedding = tf.nn.embedding_lookup(self.embeddings, document_placeholder_flat, name="Lookup")
            print('flat word embeded:', document_word_embedding)
            #documents_word_embedding = tf.reshape(document_word_embedding, [-1, self.doc_len, 
            #                                                              self.sent_len, self.embedding_size])
            
            all_sents_word_embedding = tf.reshape(document_word_embedding, [-1, self.sent_len, self.embedding_size])
            # shape: batch_size*doc_len, sent_len, embed_size
            # now it is 3 dimensional and suitable for applying GRU
    
        with tf.variable_scope("sent_level_BiGRU"):
            fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.sent_GRU_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, all_sents_word_embedding, scope='bi-GRU',
                                                                   dtype=tf.float32)
            self.sent_bigru = tf.concat([self.sent_GRU_out[0], self.sent_GRU_out[1]], 2) 
            self.sent_bigru_avg = tf.reduce_mean(self.sent_bigru, axis=1) 
            self.doc_sent_embed = tf.reshape(self.sent_bigru_avg, (params.batch_size, self.doc_len , 2 * self.hidden_size))
            
        with tf.variable_scope("doc_level_BiGRU"):
            fw_cell_2 = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            bw_cell_2 = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            doc_GRU_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_2, bw_cell_2, self.doc_sent_embed,
                                                             dtype=tf.float32)
            self.doc_bigru = tf.concat([doc_GRU_out[0], doc_GRU_out[1]], 2) 
            self.docs = tf.reduce_mean(self.doc_bigru, axis=1)         
            # document embedding
            Wf0 = tf.Variable(tf.random.uniform([2 * self.hidden_size, 100], -1.0, 1.0), name='W0')
            bf0 = tf.Variable(tf.zeros(shape=[100]), name='b0')
            self.docs_embeds = tf.nn.relu(tf.matmul(self.docs, Wf0) + bf0) 

            # position embedding
            Wpe = tf.Variable(tf.random.normal([500, self.sent_len]))
            # dense layers
            Wf = tf.Variable(tf.random.uniform([2 * self.hidden_size, 100], -1.0, 1.0), name='W')
            bf = tf.Variable(tf.zeros(shape=[100]), name='b')
            s = tf.Variable(tf.zeros(shape=(params.batch_size, 100), dtype=tf.float32))
            Wc = tf.Variable(tf.random.normal([1, 100]))
            Ws = tf.Variable(tf.random.normal([100, 100]))
            Wr = tf.Variable(tf.random.normal([100, 100]))
            Wp = tf.Variable(tf.random.normal([params.batch_size, self.sent_len]))
            bias = tf.Variable(tf.random.normal([1]), name="biases")
            
        scores = []
        with tf.variable_scope("score_layer"):

            for position, sent_hidden in enumerate(tf.unstack(self.doc_bigru, axis=(1))):
                #print('********* sent hidden', sent_hidden)  # The first sentences of all docs
                sy = tf.nn.relu(tf.matmul(sent_hidden, Wf) + bf)
                h = tf.transpose(sy, perm=(1, 0)) 
                pos_embed = tf.nn.embedding_lookup(Wpe, position) 
                p = tf.reshape(pos_embed, (1, -1)) 
                positions = tf.tile(p, tf.constant([params.batch_size,1]))
                content = tf.squeeze(tf.matmul(Wc, h)) 
                salience = tf.reduce_sum(tf.multiply(tf.matmul(sy,Ws), self.docs_embeds), axis=1)
                novelty = -1 * tf.reduce_sum(tf.multiply(tf.matmul(sy,Wr), tf.tanh(self.docs_embeds)), axis=1)            
                position_embeds = tf.reduce_sum(tf.multiply(Wp,positions), axis=1) 
                Prob = tf.sigmoid(content + salience + novelty + position_embeds + bias) 
                multiplier = tf.tile(tf.reshape(Prob,[-1,1]),tf.constant([1,100], tf.int32))
                s = s + tf.multiply(multiplier, sy) 
                scores.append(Prob)
            self.y_ = tf.transpose(tf.convert_to_tensor(scores, name="prediction"), perm=(1, 0)) 
          
                
        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):

            epsilon_one = 1e-37
            epsilon_zero = 1e-1
            target = self.y
            output = self.y_        
            self.loss = tf.reduce_sum(
                 -(target * tf.log(output + epsilon_one) + (1. - target) * tf.log(1. - output + epsilon_zero)))
            #weight = 0.7
            #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))
            # setting the pos_weight increases the loss if a positive sample is misclassified vs misclassification of a negative sample
            #self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=20))
