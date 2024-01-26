from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
# from model.HGNN_encoder import HGNNLayer
import time
from utils.evaluation import eval_type_hyperbolic
import torch
import torch.nn
from model.NYLONModel import NYLONModel
from model.graph_encoder import truncated_normal
torch.set_printoptions(precision=16)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class NYLON(torch.nn.Module):
    def __init__(self, ins_info, config):
        super(NYLON,self).__init__()
        #CONFIG SETTING
        self.config = config
        self.ins_node_num = ins_info["node_num"]       

        #INIT EMBEDDING    
        self.ins_node_embeddings = torch.nn.Embedding(self.ins_node_num, self.config['dim'])
        self.ins_node_embeddings.weight.data=truncated_normal(self.ins_node_embeddings.weight.data,std=0.02)


        ##GRAN_LAYER
        self.ins_config=dict()
        self.ins_config['num_hidden_layers']=self.config['num_hidden_layers']
        self.ins_config['num_attention_heads']=self.config['num_attention_heads']
        self.ins_config['hidden_size']=self.config['dim']
        self.ins_config['intermediate_size']=self.config['ins_intermediate_size']
        self.ins_config['hidden_dropout_prob']=self.config['hidden_dropout_prob']
        self.ins_config['attention_dropout_prob']=self.config['attention_dropout_prob']
        self.ins_config['vocab_size']=self.ins_node_num
        self.ins_config['num_relations']=ins_info["rel_num"]
        self.ins_config['num_edges']=self.config['num_edges']
        self.ins_config['max_arity']=ins_info['max_n']
        self.ins_config['device']=self.config['device']
        self.ins_granlayer=NYLONModel(self.ins_config,self.ins_node_embeddings).to(self.config['device'])



    def forward_E(self,ins_pos,ins_edge_labels, tag, confidence, correct_rate):
        # print(len(ins_pos))
        ins_input_ids, ins_input_mask, ins_mask_pos, ins_mask_label, ins_mask_type = ins_pos
        if tag == "normal":
            self.ins_triple_loss, self.ins_fc_out = self.ins_granlayer(ins_input_ids, ins_input_mask, ins_edge_labels,
                                                                       ins_mask_pos, ins_mask_label, ins_mask_type, tag,
                                                                       None,None, confidence, correct_rate)
            return self.ins_triple_loss , self.ins_fc_out
        else:
            self.ins_triple_loss, self.ins_fc_out, fc_out_vector, embeddings = self.ins_granlayer(ins_input_ids, ins_mask_label, ins_edge_labels,
                                                                       None, None, None, tag,
                                                                       ins_input_mask, ins_mask_pos, confidence, correct_rate)
            return self.ins_triple_loss, self.ins_fc_out, fc_out_vector, embeddings




