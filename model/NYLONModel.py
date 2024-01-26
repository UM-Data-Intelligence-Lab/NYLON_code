from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import torch.nn
from model.graph_encoder import encoder,truncated_normal

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class NYLONModel(torch.nn.Module):
    def __init__(self,config,node_embeddings):
        super(NYLONModel,self).__init__()

        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']

        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._max_arity = config['max_arity']
        self._max_seq_len = self._max_arity*2-1     

        self._device=config["device"]

        self.node_embedding=node_embeddings
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True)
        self.edge_embedding_k=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_k.weight.data=truncated_normal(self.edge_embedding_k.weight.data,std=0.02)
        self.edge_embedding_v=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_v.weight.data=truncated_normal(self.edge_embedding_v.weight.data,std=0.02)
        self.encoder_model=encoder( 
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout)
        self.fc1=torch.nn.Linear(self._emb_size, self._emb_size)
        self.fc1.weight.data=truncated_normal(self.fc1.weight.data,std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-7,elementwise_affine=True)
        self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._voc_size)), 0.0)
        self.fc3 = torch.nn.Linear(self._emb_size, self._emb_size)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        self.fc4 = torch.nn.Linear(self._emb_size, self._emb_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.constant_(self.fc4.bias, 0.0)
        self.fc5 = torch.nn.Linear(self._emb_size, self._max_seq_len)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.constant_(self.fc5.bias, 0.0)
        self.fc_tuple = torch.nn.Linear(self._emb_size*self._max_seq_len, self._emb_size)
        torch.nn.init.xavier_uniform_(self.fc_tuple.weight)
        torch.nn.init.constant_(self.fc_tuple.bias, 0.0)
        self.scaler = torch.nn.Parameter(torch.tensor(0.3))
        self.myloss = softmax_with_cross_entropy()
        self.loss_conf = torch.nn.BCELoss()
        
    def forward(self,input_ids, input_mask, edge_labels, mask_pos,mask_label, mask_type, mode, is_true, is_shown, confidence, correct_rate):
        emb_out = self.node_embedding(input_ids)
        batch_size = emb_out.shape[0]
        emb_out = torch.nn.Dropout(self._prepostprocess_dropout)(self.layer_norm1(emb_out))
        edges_key = self.edge_embedding_k(edge_labels)
        edges_value = self.edge_embedding_v(edge_labels)
        edge_mask = torch.sign(edge_labels).unsqueeze(2)
        edges_key = torch.mul(edges_key, edge_mask)
        edges_value = torch.mul(edges_value, edge_mask)
        input_mask=input_mask.unsqueeze(2)
        self_attn_mask = torch.matmul(input_mask,input_mask.transpose(1,2))
        self_attn_mask=1000000.0*(self_attn_mask-1.0)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)
        _enc_out = self.encoder_model(
            enc_input=emb_out,
            edges_key=edges_key,
            edges_value=edges_value,
            attn_bias=n_head_self_attn_mask)
        if mode == "normal":
            mask_pos = mask_pos.unsqueeze(1)
            mask_pos = mask_pos[:, :, None].expand(-1, -1, self._emb_size)
            h_masked = torch.gather(input=_enc_out, dim=1, index=mask_pos).reshape([-1, _enc_out.size(-1)])
            fc_out = self.fc1(h_masked)
            h_masked = torch.nn.GELU()(h_masked)
            h_masked = self.layer_norm2(h_masked)
            fc_out = torch.nn.functional.linear(h_masked, self.node_embedding.weight, self.fc2_bias)
            special_indicator = torch.empty(input_ids.size(0), 2).to(self._device)
            torch.nn.init.constant_(special_indicator, -1)
            relation_indicator = torch.empty(input_ids.size(0), self._n_relation).to(self._device)
            torch.nn.init.constant_(relation_indicator, -1)
            entity_indicator = torch.empty(input_ids.size(0), (self._voc_size - self._n_relation - 2)).to(self._device)
            torch.nn.init.constant_(entity_indicator, 1)
            type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
            mask_type = mask_type.unsqueeze(1)
            type_indicator = torch.mul(type_indicator, mask_type)
            type_indicator = torch.cat([special_indicator, type_indicator], dim=1)
            type_indicator = torch.nn.functional.relu(type_indicator)
            fc_out_mask = 1000000.0 * (type_indicator - 1.0)
            fc_out = torch.add(fc_out, fc_out_mask)
            one_hot_labels = torch.nn.functional.one_hot(mask_label, self._voc_size)  # _voc_size = node_num
            type_indicator = torch.sub(type_indicator, one_hot_labels)
            num_candidates = torch.sum(type_indicator, dim=1)
            soft_labels = ((1 + mask_type) * 0.9 +
                           (1 - mask_type) * 0.9) / 2.0
            soft_labels = soft_labels.expand(-1, self._voc_size)
            soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                          torch.mul(type_indicator, 1.0 / torch.unsqueeze(num_candidates, 1))
            mean_mask_lm_loss = self.myloss(logits=fc_out, label=soft_labels, weight=confidence)
            return mean_mask_lm_loss, fc_out

        else:
            _enc_out = _enc_out.view(batch_size, -1)
            out_embeddings = _enc_out.clone().detach()
            _enc_out = torch.nn.GELU()(self.fc_tuple(_enc_out))
            h_masked = self.fc3(_enc_out)
            h_masked = torch.nn.GELU()(h_masked)
            h_masked = self.layer_norm2(h_masked)
            h_masked = torch.nn.GELU()(self.fc4(h_masked))
            fc_out = self.fc5(h_masked)
            fc_out_vactor = torch.nn.Sigmoid()(fc_out)
            fc_out_vactor = 1 + fc_out_vactor * (input_mask.squeeze()) - input_mask.squeeze()
            fc_out = torch.min(fc_out_vactor, dim=1).values
            if len(input_mask.shape) == 3:
                is_true_sum = torch.sum(is_true * input_mask.squeeze(), dim=1)
                input_mask_sum = torch.sum(input_mask.squeeze(), dim=1)
            else:
                is_true_sum = torch.sum(is_true * input_mask, dim=1)
                input_mask_sum = torch.sum(input_mask, dim=1)
            tags = is_true_sum == input_mask_sum
            loss_conf = self.loss_conf(fc_out, tags.float())
            return loss_conf, fc_out, fc_out_vactor, out_embeddings

class softmax_with_cross_entropy(torch.nn.Module):
    def __init__(self):
        super(softmax_with_cross_entropy,self).__init__()

    def forward(self,logits, label, weight):
        logprobs=torch.nn.functional.log_softmax(logits,dim=1)
        loss=-1.0*torch.sum(torch.mul(label,logprobs),dim=1).squeeze()
        loss=torch.mean(loss*weight)
        return loss