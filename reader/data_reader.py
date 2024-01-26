
from __future__ import print_function
from __future__ import division

import random
from abc import abstractclassmethod

import logging
import collections


import json
import copy
import numpy as np
import torch


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

#读取三元组

def read_facts_new(file):
    facts_list = list()
    max_n=0
    entity_list = list()
    relation_list = list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            fact=list()
            # print(line)
            obj = json.loads(line)
            if obj['N']>max_n:
                max_n=obj['N']
            flag = 0
            for key in obj:
                if flag == 0:
                    fact.append(obj[key][0])
                    fact.append(key)
                    fact.append(obj[key][1])
                    relation_list.append(key)
                    entity_list.append(obj[key][0])
                    entity_list.append(obj[key][1])
                    break
            if obj['N']>2:
                for kv in obj.keys():
                    if kv!='N' and kv!=key:
                        if isinstance(obj[kv], list):
                            for item in obj[kv]:
                                fact.append(kv)
                                fact.append(item)
                                relation_list.append(kv)
                                entity_list.append(item)
                        else:
                            fact.append(kv)
                            fact.append(obj[kv])
                            relation_list.append(kv)
                            entity_list.append(obj[kv])
            facts_list.append(fact)
    return facts_list,max_n,relation_list,entity_list

def read_dict(ent_file,rel_file):
    dict_id=dict()
    dict_id['PAD']=0
    dict_id['MASK']=1
    dict_num=2
    rel_num=0
    with open(rel_file, 'r', encoding='utf8') as f:
        for line in f:
            line=line.strip('\n')
            dict_id[line]=dict_num
            dict_num+=1
            rel_num+=1

    with open(ent_file, 'r', encoding='utf8') as f:
        for line in f:
            line=line.strip('\n')
            dict_id[line]=dict_num
            dict_num+=1

    return dict_id,dict_num,rel_num


def read_dict_new(e_ls,r_ls):
    dict_id=dict()
    dict_id['PAD']=0
    dict_id['MASK']=1
    dict_num=2
    rel_num=0

    for item in r_ls:
        dict_id[item] = dict_num
        dict_num += 1
        rel_num += 1

    for item in e_ls:
        dict_id[item] = dict_num
        dict_num += 1

    return dict_id,dict_num,rel_num


def facts_to_id(facts,max_n,node_dict, is_true_out):
    id_facts=list()
    id_masks=list()
    mask_labels=list()
    mask_pos=list()
    mask_types=list()
    real_triples = list()
    is_true = list()
    is_shown = list()
    real_masks = list()
    replace_masks = list()
    for fact in facts:
        id_fact=list()
        id_mask=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
            id_mask.append(1.0)
        
        for j,mask_label in enumerate(id_fact):
            x=copy.copy(id_fact)        
            x[j]=1
            y=copy.copy(id_mask)
            z = copy.copy(id_mask)
            if j%2==0:
                mask_type=1
            else:
                mask_type=-1
            while len(x)<(2*max_n-1):
                x.append(0) 
                y.append(0.0)
                z.append(1.0)
            id_facts.append(x)
            id_masks.append(y)
            replace_masks.append(z)
            mask_pos.append(j)
            if j == 0:
                is_true.append(0)
                is_shown.append(0)
                x_copy = copy.copy(x)
                x_copy[0] = mask_label
                real_triples.append(x_copy)
                real_masks.append(y)
            mask_labels.append(mask_label)    
            mask_types.append(mask_type)
    return [id_facts,id_masks,mask_pos,mask_labels,mask_types], [real_triples, is_true_out.numpy().tolist(), torch.zeros(is_true_out.shape).numpy().tolist(), real_masks, is_true_out.numpy().tolist()]

def get_truth(all_facts,max_n,node_dict):
    max_aux=max_n-2
    max_seq_length = 2 * max_aux + 3
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    all_fact_ids=list()
    for fact in all_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item]) 
        all_fact_id=copy.copy(id_fact)
        all_fact_ids.append(all_fact_id)
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        for pos in range(max_seq_length):
            if id_fact[pos]==0:
                continue
            key = " ".join([
                str(id_fact[x]) for x in range(max_seq_length) if x != pos
            ])
            gt_dict[pos][key].append(id_fact[pos])          

    return gt_dict,all_fact_ids

def get_input(train_file, valid_file, test_file, noise_file, initiatial_amount):

    train_facts,max_train,train_r,train_e = read_facts_new(train_file)
    valid_facts,max_valid,valid_r,valid_e  = read_facts_new(valid_file)
    test_facts,max_test,test_r,test_e = read_facts_new(test_file)

    max_n = max(max_train, max_valid, max_test)
    e_list = list(set(train_e + valid_e + test_e))
    r_list = list(set(train_r + valid_r + test_r))
    noise_facts = []
    is_true = torch.ones([0, max_n*2-1])
    train_fact_num = len(train_facts)
    noise_amount = int(train_fact_num * initiatial_amount)
    for _ in range(noise_amount):
        temp_fact = copy.copy(random.choice(train_facts))
        is_true_temp = torch.ones([1, max_n*2-1])
        replace_num = int(random.randint(1, len(temp_fact) - 1) / 2)
        if replace_num == 0:
            replace_num = 1
        for j in range(replace_num):
            replace_index = random.randint(0, len(temp_fact) - 1)
            is_true_temp[0, replace_index] = 0
            if replace_index % 2 == 0:
                random_replace = random.choice(e_list)
                while True:
                    if random_replace != temp_fact[replace_index]:
                        break
                    random_replace = random.choice(e_list)
                temp_fact[replace_index] = random_replace
            else:
                random_replace = random.choice(r_list)
                while True:
                    if random_replace != temp_fact[replace_index]:
                        break
                    random_replace = random.choice(r_list)
                temp_fact[replace_index] = random_replace
        is_true = torch.cat([is_true, is_true_temp], dim=0)
        noise_facts.append(temp_fact)

    noise_test = []
    is_true_test = torch.ones([0, max_n * 2 - 1])
    test_fact_num = len(test_facts)
    noise_amount_test = int(test_fact_num * initiatial_amount)
    for _ in range(noise_amount_test):
        temp_fact = copy.copy(random.choice(test_facts))
        is_true_temp = torch.ones([1, max_n * 2 - 1])
        replace_num = int(random.randint(1, len(temp_fact) - 1) / 2)
        if replace_num == 0:
            replace_num = 1
        for j in range(replace_num):
            replace_index = random.randint(0, len(temp_fact) - 1)
            is_true_temp[0, replace_index] = 0
            if replace_index % 2 == 0:
                random_replace = random.choice(e_list)
                while True:
                    if random_replace != temp_fact[replace_index]:
                        break
                    random_replace = random.choice(e_list)
                temp_fact[replace_index] = random_replace
            else:
                random_replace = random.choice(r_list)
                while True:
                    if random_replace != temp_fact[replace_index]:
                        break
                    random_replace = random.choice(r_list)
                temp_fact[replace_index] = random_replace
        is_true_test = torch.cat([is_true_test, is_true_temp], dim=0)
        noise_test.append(temp_fact)
    all_facts = train_facts + valid_facts + test_facts + noise_facts
    node_dict, node_num, rel_num=read_dict_new(e_list,r_list)
    all_facts,all_fact_ids= get_truth(all_facts,max_n,node_dict)
    train_facts, train_real = facts_to_id(train_facts,max_n,node_dict, torch.ones([len(train_facts), 2*max_n-1]))
    valid_facts, _= facts_to_id(valid_facts,max_n,node_dict, torch.ones(is_true.shape))
    test_facts, test_real= facts_to_id(test_facts,max_n,node_dict, torch.ones([len(test_facts), 2*max_n-1]))
    noise_facts, noise_real = facts_to_id(noise_facts,max_n,node_dict, is_true)
    _, test_noise_real = facts_to_id(noise_test, max_n, node_dict, is_true_test)
    for i in range(len(train_facts)):
        train_facts[i] += noise_facts[i]
    for i in range(len(train_real)):
        train_real[i] += noise_real[i]
    for i in range(len(test_real)):
        test_real[i] = test_noise_real[i] + test_real[i]
    input_info=dict()
    input_info['all_facts']=all_facts
    input_info['all_fact_ids']=all_fact_ids
    input_info['train_facts']=train_facts
    input_info['train_real'] = train_real
    input_info['test_real'] = test_real
    input_info['valid_facts']=valid_facts
    input_info['test_facts']=test_facts
    input_info['node_dict']=node_dict
    input_info['node_num']=node_num
    input_info['rel_num']=rel_num
    input_info['max_n']=max_n
    return input_info

def truth_to_id(all_facts, ins_ent_ids, onto_ent_ids):
    typing=dict()
    for fact in all_facts:
        if fact[0] not in typing.keys():
            typing[fact[0]]=list()
        if onto_ent_ids[fact[2]] not in typing[fact[0]]:
            typing[fact[0]].append(onto_ent_ids[fact[2]])
    typing_id=dict()
    for key in typing.keys():
        typing_id[str(ins_ent_ids[key])]=typing[key]
    return typing_id

def read_input(folder, initiatial_amount):
    ins_info = get_input(folder +"/n-ary_train.json", folder + "/n-ary_valid.json", folder + "/n-ary_test.json", folder + "/n-ary_noise.json", initiatial_amount)

    logger.info("Number of ins_all fact_ids: "+str(len(ins_info['all_fact_ids'])))
    logger.info("Number of ins_train facts: "+str(len(ins_info['train_facts'][0])))
    logger.info("Number of ins_valid facts: "+str(len(ins_info['valid_facts'][0])))
    logger.info("Number of ins_test facts: "+str(len(ins_info['test_facts'][0])))
    logger.info("Number of ins nodes: "+str(ins_info['node_num']))
    logger.info("Number of ins relations: "+str(ins_info['rel_num']))
    logger.info("Number of ins max_n: "+str(ins_info['max_n']))
    logger.info("Number of ins max_seq_length: "+str(2*ins_info['max_n']-1))


    return ins_info








