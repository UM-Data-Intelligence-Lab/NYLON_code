import argparse
import ast
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.args import ArgumentGroup, print_arguments
import logging
from reader.data_reader import read_input
from reader.data_loader import prepare_EC_info, get_edge_labels
from model.NYLON import NYLON
import time
import math
import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import copy
import numpy as np
from itertools import cycle
from utils.evaluation import batch_evaluation, compute_metrics

torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

parser = argparse.ArgumentParser(description='HyperKE4TI')
NYLON_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
NYLON_g.add_arg('input', type=str, default='dataset/jf17k', help="")  # db
NYLON_g.add_arg('output', type=str, default='./', help="")

NYLON_g.add_arg('dim', type=int, default=256, help="")
NYLON_g.add_arg('onto_dim', type=int, default=256, help="")
NYLON_g.add_arg('ins_layer_num', type=int, default=3, help="")
NYLON_g.add_arg('onto_layer_num', type=int, default=3, help="")
NYLON_g.add_arg('neg_typing_margin', type=float, default=0.1, help="")
NYLON_g.add_arg('neg_triple_margin', type=float, default=0.2, help="")

NYLON_g.add_arg('nums_neg', type=int, default=30, help="")
NYLON_g.add_arg('mapping_neg_nums', type=int, default=30, help="")

NYLON_g.add_arg('learning_rate', type=float, default=1e-4, help="")
NYLON_g.add_arg('batch_size', type=int, default=1024, help="")
NYLON_g.add_arg('epochs', type=int, default=100, help="")

NYLON_g.add_arg('combine', type=ast.literal_eval, default=True, help="")
NYLON_g.add_arg('ent_top_k', type=list, default=[1, 3, 5, 10], help="")
NYLON_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")

NYLON_g.add_arg('ins_intermediate_size', type=int, default=512, help="")
NYLON_g.add_arg('onto_intermediate_size', type=int, default=512, help="")
NYLON_g.add_arg('num_hidden_layers', type=int, default=12, help="")
NYLON_g.add_arg('num_attention_heads', type=int, default=4, help="")
NYLON_g.add_arg('hidden_dropout_prob', type=float, default=0.1, help="")
NYLON_g.add_arg('attention_dropout_prob', type=float, default=0.1, help="")
NYLON_g.add_arg('num_edges', type=int, default=6, help="")

NYLON_g.add_arg('noise_level', type=float, default=1.0, help="")
NYLON_g.add_arg('active_sample_per_epoch', type=float, default=0.025, help="")
NYLON_g.add_arg('meta_lr', type=float, default=0.1, help="")
NYLON_g.add_arg('error_detection_every_x_epochs', type=int, default=1, help="")
NYLON_g.add_arg('aug_amount', type=int, default=0, help="")

args = parser.parse_args()


class EDataset(Dataset.Dataset):
    def __init__(self, triples1):
        self.triples1 = triples1

    def __len__(self):
        return len(self.triples1[0])

    def __getitem__(self, index):
        return self.triples1[0][index], self.triples1[1][index], self.triples1[2][index], self.triples1[3][index], \
               self.triples1[4][index]

class EDataset6(Dataset.Dataset):
    def __init__(self, triples1):
        self.triples1 = triples1

    def __len__(self):
        return len(self.triples1[0])

    def __getitem__(self, index):
        return self.triples1[0][index], self.triples1[1][index], self.triples1[2][index], self.triples1[3][index], \
               self.triples1[4][index], self.triples1[5][index]

def is_same(tensor1, tensor2, n_arity):
    list1 = tensor1.cpu().tolist()
    list2 = tensor2.cpu().tolist()
    for i in range(n_arity-1):
        if i == 0:
            tag1 = list1[0] + list1[1] + list1[2]
            tag2 = list2[0] + list2[1] + list2[2]
            tag1 = (tag1 == 3)
            tag2 = (tag2 == 3)
            if tag1 != tag2:
                return 0
        else:
            tag1 = list1[2 * i + 1] + list1[2 * i + 2]
            tag2 = list2[2 * i + 1] + list2[2 * i + 2]
            tag1 = (tag1 == 2)
            tag2 = (tag2 == 2)
            if tag1 != tag2:
                return 0
    return 1

            

def main(args):
    config = vars(args)
    if args.use_cuda:
        device = torch.device("cuda")
        config["device"] = "cuda"
    else:
        device = torch.device("cpu")
        config["device"] = "cpu"

    ins_info = read_input(args.input, args.noise_level)

    instance_info = prepare_EC_info(ins_info, device)
    ins_edge_labels = get_edge_labels(ins_info['max_n']).to(device)

    model_normal = NYLON(instance_info, config).to(device)

    # E_train_dataloader
    ins_train_facts = list()
    for ins_train_fact in ins_info['train_facts']:
        ins_train_fact = torch.tensor(ins_train_fact).to(device)
        ins_train_facts.append(ins_train_fact)
    ins_train_facts.append(torch.ones(ins_train_facts[0].shape[0]).to(device))
    train_data_E_reader = EDataset6(ins_train_facts)
    train_E_pyreader = DataLoader.DataLoader(train_data_E_reader, batch_size=args.batch_size, shuffle=True,
                                             drop_last=False)

    # train_information
    logging.info("train_ins_batch_size: " + str(args.batch_size))
    logging.info("train_onto_batch_size: " + str(args.batch_size))
    steps = math.ceil(len(ins_info['train_facts']) / args.batch_size)
    logging.info("train_steps_per_epoch: " + str(steps))

    # E_valid_dataloader
    ins_valid_facts = list()
    for ins_valid_fact in ins_info['valid_facts']:
        ins_valid_fact = torch.tensor(ins_valid_fact).to(device)
        ins_valid_facts.append(ins_valid_fact)
    valid_data_E_reader = EDataset(ins_valid_facts)
    valid_E_pyreader = DataLoader.DataLoader(
        valid_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    # E_valid_dataloader
    ins_test_reals = list()
    for ins_test_fact in ins_info['test_real']:
        ins_test_fact = torch.tensor(ins_test_fact).to(device)
        ins_test_reals.append(ins_test_fact)
    test_real_data_E_reader = EDataset(ins_test_reals)
    test_real_E_pyreader = DataLoader.DataLoader(
        test_real_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    ins_test_facts = list()
    for ins_test_fact in ins_info['test_facts']:
        ins_test_fact = torch.tensor(ins_test_fact).to(device)
        ins_test_facts.append(ins_test_fact)
    test_data_E_reader = EDataset(ins_test_facts)
    test_E_pyreader = DataLoader.DataLoader(
        test_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    ins_real_facts = list()
    for ins_real_fact in ins_info['train_real']:
        ins_real_fact = torch.tensor(ins_real_fact).to(device)
        ins_real_facts.append(ins_real_fact)
    real_data_E_reader = EDataset(ins_real_facts)
    real_E_pyreader = DataLoader.DataLoader(
        real_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    no_shuffle_reader = DataLoader.DataLoader(
        real_data_E_reader,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)
    samples = int(torch.sum(ins_real_facts[3]).int().item() * args.active_sample_per_epoch)
    print("sample amount : ", samples)

    real_indexing = dict()
    for index in range(ins_train_facts[0].shape[0]):
        key_real_indexing = ins_train_facts[0][index].cpu().numpy().tolist()
        key_real_indexing[ins_train_facts[2][index].item()] = ins_train_facts[3][index].item()
        key_real_indexing = [str(x) for x in key_real_indexing]
        key_real_indexing = "_".join(key_real_indexing)
        if key_real_indexing in real_indexing:
            real_indexing[key_real_indexing].append(index)
        else:
            real_indexing[key_real_indexing] = [index]


    # ECS_optimizers
    ins_optimizer_normal = torch.optim.Adam([{"params": model_normal.parameters()}], lr=config['learning_rate'])
    al_list = list()
    rel_num = ins_info["rel_num"]
    node_num = ins_info["node_num"]
    global_true_list = list()

    confidences = torch.empty([0, ]).to(device)
    positional_confidences = torch.empty([0, ]).to(device)
    model_normal.eval()
    all_reviewed_true = (torch.sum(ins_real_facts[2], dim=1) >= 1).int() + (torch.sum(ins_real_facts[1] * ins_real_facts[3], dim=1) == torch.sum(ins_real_facts[3], dim=1)).int() == 2
    all_reviewed_true = torch.nonzero(all_reviewed_true).cpu().numpy().tolist()
    for j, data in enumerate(no_shuffle_reader):
        [real_triples, is_true, is_shown, real_masks, _] = data
        ins_optimizer_normal.zero_grad()
        _, confidence, positional_confidence, _ = model_normal.forward_E(data, ins_edge_labels, "conf",
                                                                      None, 0)
        positional_confidence_temp = positional_confidence.clone().detach()
        positional_confidences = torch.cat([positional_confidences, positional_confidence_temp], dim=0)
        confidence_temp = confidence.clone().detach()
        confidences = torch.cat([confidences, confidence_temp])
    confidences = (confidences - 0.5) ** 2
    confidences = confidences.view(-1)
    ranks = torch.argsort(confidences)
    true_list = list()
    false_list = list()
    global_counter = 0
    for index in ranks:
        is_right = True
        if torch.sum(ins_real_facts[2][index]) > 0:
            continue
        positional_confidence_selected = positional_confidences[index, :]
        local_ranks = torch.argsort(positional_confidence_selected)
        local_counter = 0
        for position in local_ranks:
            if ins_real_facts[3][index, position] == 0:
                continue
            ins_real_facts[2][index, position] = 1
            local_counter += 1
            if ins_real_facts[1][index, position] == 0:
                is_right = False
                break
        global_counter += local_counter
        if torch.sum(ins_real_facts[1][index] * ins_real_facts[3][index]) == torch.sum(ins_real_facts[3][index]):
            true_list.append(index.item())
        else:
            false_list.append(index.item())
        if global_counter >= samples:
            break

    true_facts = ins_real_facts[0][true_list].clone()
    true_true = ins_real_facts[1][true_list].clone()
    true_shown = ins_real_facts[2][true_list].clone()
    true_mask = ins_real_facts[3][true_list].clone()
    true_place = ins_real_facts[4][true_list].clone()
    false_facts = ins_real_facts[0][false_list].clone()
    false_true = ins_real_facts[1][false_list].clone()
    false_shown = ins_real_facts[2][false_list].clone()
    false_mask = ins_real_facts[3][false_list].clone()
    false_place = ins_real_facts[4][false_list].clone()
    true_num = len(true_list)
    false_num = len(false_list)
    global_true_list += true_list
    al_facts = torch.cat([true_facts, false_facts], dim=0)
    al_true = torch.cat([true_true, false_true], dim=0)
    al_shown = torch.cat([true_shown, false_shown], dim=0)
    al_mask = torch.cat([true_mask, false_mask], dim=0)
    al_place = torch.cat([true_place, false_place], dim=0)
    al_list.append([al_facts.long(), al_true, al_shown, al_mask, al_true, true_list])
    if len(al_list) > 10:
        del al_list[0]
        torch.cuda.empty_cache()

    # Start Training
    iterations = 1

    for iteration in range(1, args.epochs // iterations + 1):
        logger.info("iteration " + str(iteration))
        model_normal.train()
        correct_rate = torch.sum(ins_real_facts[2] * ins_real_facts[1]) / torch.sum(ins_real_facts[2])
        for i in range(iterations):
            ins_epoch_loss = 0
            start = time.time()
            model_normal.train()
            if iteration % args.error_detection_every_x_epochs == 0:
                print("Start training the cross-grained confidence evaluator")
                tuple_embeddings = torch.empty(
                    [0, model_normal.ins_config["hidden_size"] * (2 * ins_info['max_n'] - 1)]).to(device)
                for j, data in enumerate(no_shuffle_reader):
                    _, _, _, temp_embeddings = model_normal.forward_E(data, ins_edge_labels, "conf", None, 0)
                    tuple_embeddings = torch.cat([tuple_embeddings, temp_embeddings.clone().detach()], dim=0)

                for j in range(len(al_list)):
                    id_facts_aug = torch.empty([0, al_facts.shape[1]]).to(device)
                    is_true_aug = torch.empty([0, al_true.shape[1]]).to(device)
                    is_shown_aug = torch.empty([0, al_shown.shape[1]]).to(device)
                    id_masks_aug = torch.empty([0, al_mask.shape[1]]).to(device)
                    id_facts_aug_false = torch.empty([0, al_facts.shape[1]]).to(device)
                    is_true_aug_false = torch.empty([0, al_true.shape[1]]).to(device)
                    is_shown_aug_false = torch.empty([0, al_shown.shape[1]]).to(device)
                    id_masks_aug_false = torch.empty([0, al_mask.shape[1]]).to(device)
                    last_loss = 0
                    true_list = al_list[j][5]
                    root_embeddings = tuple_embeddings[true_list]
                    belong_list = dict()
                    for embedding_index in range(tuple_embeddings.shape[0]):
                        closest_center = true_list[torch.argmin(
                            torch.norm(root_embeddings - tuple_embeddings[embedding_index].view(1, -1), dim=1))]
                        if closest_center in belong_list:
                            belong_list[closest_center].append(embedding_index)
                        else:
                            belong_list[closest_center] = [embedding_index]
                    correct_2 = 0
                    total_2 = 0
                    correct_5 = 0
                    total_5 = 0
                    correct_10 = 0
                    total_10 = 0
                    correct_20 = 0
                    total_20 = 0
                    correct_50 = 0
                    total_50 = 0
                    aug_amount = 0
                    for center in belong_list:
                        if torch.sum(ins_real_facts[1][center] * ins_real_facts[3][center]) != torch.sum(ins_real_facts[3][center]):
                            print("error occur!")
                            print(ins_real_facts[1][center])
                            continue
                        belong_embeddings = tuple_embeddings[belong_list[center]]
                        aug_distance = torch.norm(belong_embeddings - tuple_embeddings[center], dim=1)
                        aug_distance[torch.sum(ins_real_facts[2][belong_list[center]], dim=1) > 0] = 999999999
                        temp_ranks = torch.tensor(belong_list[center], device=device)[
                            torch.argsort(aug_distance, descending=False)]
                        correct_2 += torch.sum((torch.sum(ins_real_facts[1][temp_ranks[:2]] * ins_real_facts[3][temp_ranks[:2]], dim=1) == torch.sum(ins_real_facts[3][temp_ranks[:2]], dim=1)).int())
                        correct_5 += torch.sum((torch.sum(ins_real_facts[1][temp_ranks[:5]] * ins_real_facts[3][temp_ranks[:5]], dim=1) == torch.sum(ins_real_facts[3][temp_ranks[:5]], dim=1)).int())
                        correct_10 += torch.sum((torch.sum(ins_real_facts[1][temp_ranks[:10]] * ins_real_facts[3][temp_ranks[:10]], dim=1) == torch.sum(ins_real_facts[3][temp_ranks[:10]], dim=1)).int())
                        correct_20 += torch.sum((torch.sum(ins_real_facts[1][temp_ranks[:20]] * ins_real_facts[3][temp_ranks[:20]], dim=1) == torch.sum(ins_real_facts[3][temp_ranks[:20]], dim=1)).int())
                        correct_50 += torch.sum((torch.sum(ins_real_facts[1][temp_ranks[:50]] * ins_real_facts[3][temp_ranks[:50]], dim=1) == torch.sum(ins_real_facts[3][temp_ranks[:50]], dim=1)).int())
                        total_2 += 2
                        total_5 += 5
                        total_10 += 10
                        total_20 += 20
                        total_50 += 50
                        temp_ranks = temp_ranks.cpu().numpy().tolist()
                        # print(temp_ranks)
                        while True:
                            if len(temp_ranks) < args.aug_amount:
                                temp_ranks.append(random.choice(temp_ranks))
                            else:
                                break
                        temp_ranks = torch.tensor(temp_ranks, dtype=torch.long, device=device)
                        temp_ranks = temp_ranks[:args.aug_amount]
                        id_facts_aug = torch.cat([id_facts_aug, ins_real_facts[0][temp_ranks].clone().detach()], dim=0)
                        is_true_aug = torch.cat(
                            [is_true_aug, ins_real_facts[3][temp_ranks].clone().detach()], dim=0)
                        is_shown_aug = torch.cat(
                            [is_shown_aug, ins_real_facts[3][temp_ranks].clone().detach()], dim=0)
                        id_masks_aug = torch.cat([id_masks_aug, ins_real_facts[3][temp_ranks].clone().detach()], dim=0)
                        aug_amount += temp_ranks.shape[0] * 2
                        for temp_rank_index in temp_ranks:
                            temp_fact = ins_real_facts[0][temp_rank_index].clone().detach()
                            # print("temp_fact_before: ", temp_fact)
                            temp_true = ins_real_facts[3][temp_rank_index].clone().detach()
                            temp_shown = ins_real_facts[3][temp_rank_index].clone().detach()
                            temp_fact_mask = ins_real_facts[3][temp_rank_index].clone().detach()
                            replace_num = int(random.randint(1, torch.sum(temp_fact_mask).item() - 1) / 2)
                            if replace_num == 0:
                                replace_num = 1
                            for replace_index in range(replace_num):
                                replace_pos = random.randint(0, torch.sum(temp_fact_mask).item() - 1)
                                temp_true[replace_pos] = 0
                                if replace_pos % 2 == 0:
                                    random_replace = random.randint(rel_num + 2, node_num - 1)
                                    while True:
                                        if random_replace != temp_fact[replace_pos]:
                                            break
                                        random_replace = random.randint(rel_num + 2, node_num - 1)
                                    temp_fact[replace_pos] = random_replace
                                else:
                                    random_replace = random.randint(2, rel_num + 1)
                                    while True:
                                        if random_replace != temp_fact[replace_pos]:
                                            break
                                        random_replace = random.randint(2, rel_num + 1)
                                    temp_fact[replace_pos] = random_replace
                            id_facts_aug_false = torch.cat([id_facts_aug_false, temp_fact.view(1, -1)], dim=0)
                            is_true_aug_false = torch.cat([is_true_aug_false, temp_true.view(1, -1)], dim=0)
                            is_shown_aug_false = torch.cat([is_shown_aug_false, temp_shown.view(1, -1)], dim=0)
                            id_masks_aug_false = torch.cat([id_masks_aug_false, temp_fact_mask.view(1, -1)], dim=0)
                    correct_rate_j = torch.sum((torch.sum(al_list[j][1]*al_list[j][3], dim=1) == torch.sum(al_list[j][3], dim=1)).int()).item()
                    total_j = al_list[j][0].shape[0]
                    # print("id_facts_aug_false: ", id_facts_aug_false.shape[0])
                    # print("id_facts_aug: ", id_facts_aug.shape[0])
                    # print("correct_rate_j: ", correct_rate_j)
                    # print("total_j: ", total_j)
                    if correct_rate_j > 0.5 * total_j:
                        retain_count = int((total_j - correct_rate_j) * id_facts_aug_false.shape[0] / correct_rate_j)
                        retain_index = random.sample(range(id_facts_aug_false.shape[0]), retain_count)
                        # print(max(retain_index))
                        # print(id_facts_aug_false.shape[0])
                        id_facts_aug_false = id_facts_aug_false[retain_index]
                        is_true_aug_false = is_true_aug_false[retain_index]
                        is_shown_aug_false = is_shown_aug_false[retain_index]
                        id_masks_aug_false = id_masks_aug_false[retain_index]
                    else:
                        retain_count = int(correct_rate_j * id_facts_aug.shape[0] / (total_j - correct_rate_j))
                        retain_index = random.sample(range(id_facts_aug.shape[0]), retain_count)
                        # print(max(retain_index))
                        # print(id_facts_aug.shape[0])
                        id_facts_aug = id_facts_aug[retain_index]
                        is_true_aug = is_true_aug[retain_index]
                        is_shown_aug = is_shown_aug[retain_index]
                        id_masks_aug = id_masks_aug[retain_index]
                    id_facts_aug = torch.cat([id_facts_aug, id_facts_aug_false], dim=0)
                    is_true_aug = torch.cat([is_true_aug, is_true_aug_false], dim=0)
                    is_shown_aug = torch.cat([is_shown_aug, is_shown_aug_false], dim=0)
                    id_masks_aug = torch.cat([id_masks_aug, id_masks_aug_false], dim=0)
                    # print("closest 2: ", correct_2 / total_2)
                    # print("closest 5: ", correct_5 / total_5)
                    # print("closest 10: ", correct_10 / total_10)
                    # print("closest 20: ", correct_20 / total_20)
                    # print("closest 50: ", correct_50 / total_50)
                    # print("aug_amount: ", aug_amount)
                    # print("correct_rate: ", correct_rate_j/total_j)

                    # temp_states = list()
                    raw_state = model_normal.state_dict()
                    while True:
                        [id_facts, is_true, is_shown, id_masks, place_mask, true_list] = al_list[j]
                        loss_item = 0
                        # print(id_facts_aug.shape[0])
                        if id_facts.shape[0] == 0:
                            pass
                        else:
                            ins_pos_final = [id_facts_aug.long(), is_true_aug, is_shown_aug, id_masks_aug, is_true_aug]
                            for concat_index in range(len(ins_pos_final)):
                                ins_pos_final[concat_index] = torch.cat([ins_pos_final[concat_index], al_list[j][concat_index]], dim=0)
                            aug_dataset = EDataset(ins_pos_final)
                            aug_reader = DataLoader.DataLoader(
                                aug_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=False)
                            for aug_index, data in enumerate(aug_reader):
                                ins_optimizer_normal.zero_grad()
                                ins_loss_conf, _, fc_out_vector, _ = model_normal.forward_E(data,
                                                                                            ins_edge_labels,
                                                                                            "conf", None,
                                                                                            correct_rate)
                                ins_loss_pos_nong = torch.nn.BCELoss(reduction="none")(fc_out_vector * data[2],
                                                                                       data[1] * data[
                                                                                           2])
                                ins_loss_pos_nong = torch.sum(ins_loss_pos_nong, dim=1) / torch.sum(data[2],
                                                                                                    dim=1)
                                ins_loss_pos = torch.mean(ins_loss_pos_nong)
                                ins_loss = ins_loss_pos + ins_loss_conf
                                ins_loss.backward()
                                ins_optimizer_normal.step()
                                loss_item += ins_loss.item()
                            if j % 1 == 0:
                                logger.info(
                                    str(j) + ' , ins_loss_conf: ' + str(ins_loss.item()) + " with memory " + str(
                                        torch.cuda.memory_allocated(device=device) / 1024 / 1024))
                            if abs(loss_item - last_loss) / loss_item <= 0.05:
                                break
                            else:
                                last_loss = loss_item
                    temp_state = model_normal.state_dict()
                    for key in temp_state:
                        temp_state[key] = raw_state[key] + args.meta_lr * (temp_state[key] - raw_state[key])
                    model_normal.load_state_dict(temp_state)
                    del temp_state
                    del raw_state
                    torch.cuda.empty_cache()

                print("Start active learning with effort-efficient active labeler")
                confidences = torch.empty([0, ]).to(device)
                positional_confidences = torch.empty([0, ]).to(device)
                model_normal.eval()
                all_reviewed_true = (torch.sum(ins_real_facts[2], dim=1) >= 1).int() + (
                            torch.sum(ins_real_facts[1] * ins_real_facts[3], dim=1) == torch.sum(ins_real_facts[3],
                                                                                                 dim=1)).int() == 2
                for j, data in enumerate(no_shuffle_reader):
                    ins_optimizer_normal.zero_grad()
                    _, confidence, positional_confidence, _ = model_normal.forward_E(data, ins_edge_labels, "conf",
                                                                                  None, 0)
                    positional_confidence_temp = positional_confidence.clone().detach()
                    positional_confidences = torch.cat([positional_confidences, positional_confidence_temp], dim=0)
                    confidence_temp = confidence.clone().detach()
                    for update_index in range(data[0].shape[0]):
                        update_key = data[0][update_index].cpu().numpy().tolist()
                        update_key = [str(x) for x in update_key]
                        update_key = "_".join(update_key)
                        ins_train_facts[5][real_indexing[update_key]] = confidence_temp[update_index].clone()
                    confidences = torch.cat([confidences, confidence_temp])
                confidences = (confidences - 0.5) ** 2
                confidences = confidences.view(-1)
                ranks = torch.argsort(confidences)
                true_list = list()
                false_list = list()
                global_counter = 0
                for index in ranks:
                    is_right = True
                    if torch.sum(ins_real_facts[2][index]) > 0:
                        continue
                    positional_confidence_selected = positional_confidences[index, :]
                    local_ranks = torch.argsort(positional_confidence_selected)
                    local_counter = 0
                    for position in local_ranks:
                        if ins_real_facts[3][index, position] == 0:
                            continue
                        ins_real_facts[2][index, position] = 1
                        local_counter += 1
                        if ins_real_facts[1][index, position] == 0:
                            is_right = False
                            break
                    global_counter += local_counter
                    if torch.sum(ins_real_facts[1][index] * ins_real_facts[3][index]) == torch.sum(
                            ins_real_facts[3][index]):
                        true_list.append(index.item())
                    else:
                        false_list.append(index.item())
                    if global_counter >= samples:
                        break

                true_facts = ins_real_facts[0][true_list].clone()
                true_true = ins_real_facts[1][true_list].clone()
                true_shown = ins_real_facts[2][true_list].clone()
                true_mask = ins_real_facts[3][true_list].clone()
                true_place = ins_real_facts[4][true_list].clone()
                false_facts = ins_real_facts[0][false_list].clone()
                false_true = ins_real_facts[1][false_list].clone()
                false_shown = ins_real_facts[2][false_list].clone()
                false_mask = ins_real_facts[3][false_list].clone()
                false_place = ins_real_facts[4][false_list].clone()
                global_true_list += true_list

                al_facts = torch.cat([true_facts, false_facts], dim=0)
                al_true = torch.cat([true_true, false_true], dim=0)
                al_shown = torch.cat([true_shown, false_shown], dim=0)
                al_mask = torch.cat([true_mask, false_mask], dim=0)
                al_list.append([al_facts.long(), al_true, al_shown, al_mask, al_true, true_list])
                if len(al_list) > 10:
                    del al_list[0]
                    torch.cuda.empty_cache()

                batch_num = 0
                sum_cos = 0
                average_margin = 0
                right_sum = 0
                wrong_sum = 0
                right_cos = 0
                wrong_cos = 0

                right_right_fact = 0
                right_false_fact = 0
                false_false_fact = 0
                false_right_fact = 0

                right_right_element = 0
                right_false_element = 0
                false_false_element = 0
                false_right_element = 0

                model_normal.eval()
                for _, data in enumerate(test_real_E_pyreader):
                    _, confidences_tag, fc_out_vector, _ = model_normal.forward_E(data, ins_edge_labels, "conf", None,
                                                                               correct_rate)
                    tags = data[4]
                    fc_out_vector[fc_out_vector >= 0.5] = 1
                    fc_out_vector[fc_out_vector < 0.5] = 0
                    confidences_tag[confidences_tag >= 0.5] = 1
                    confidences_tag[confidences_tag < 0.5] = 0
                    input_masks = data[3].squeeze()
                    fc_out_vector = fc_out_vector * input_masks
                    tags = tags * input_masks
                    for i in range(fc_out_vector.shape[0]):
                        output = fc_out_vector[i, :]
                        tag = tags[i, :]
                        mask = input_masks[i, :]
                        conf_tag = confidences_tag[i]
                        if torch.sum(tag) == torch.sum(mask):
                            total_tag = 1
                        else:
                            total_tag = 0
                        batch_num += 1
                        is_same_num = is_same(output, tag, ins_info['max_n'])
                        sum_cos += is_same_num
                        if mask.equal(tag):
                            right_sum += 1
                            right_cos += is_same_num
                        else:
                            wrong_sum += 1
                            wrong_cos += is_same_num

                        if total_tag == 1 and conf_tag == 1:
                            right_right_fact += 1
                        if total_tag == 1 and conf_tag == 0:
                            right_false_fact += 1
                        if total_tag == 0 and conf_tag == 1:
                            false_right_fact += 1
                        if total_tag == 0 and conf_tag == 0:
                            false_false_fact += 1

                        for index_conf in range(torch.sum(mask).int().item()):
                            if tag[index_conf] == 1 and output[index_conf] == 1:
                                right_right_element += 1
                            if tag[index_conf] == 1 and output[index_conf] == 0:
                                right_false_element += 1
                            if tag[index_conf] == 0 and output[index_conf] == 1:
                                false_right_element += 1
                            if tag[index_conf] == 0 and output[index_conf] == 0:
                                false_false_element += 1

            model_normal.train()
            # conf_sum = 0
            print("Start training hyper-relational link predictor")
            for j, data in enumerate(train_E_pyreader):
                [id_facts, id_masks, mask_pos, mask_labels, mask_types, confidence] = data
                id_facts_temp = copy.deepcopy(id_facts)
                id_facts_len = id_facts_temp.shape[0]
                id_facts_temp[list(range(id_facts_len)), mask_pos] = mask_labels
                ins_optimizer_normal.zero_grad()
                bs = id_facts_temp.shape[0]
                ins_pos_normal = [id_facts, id_masks, mask_pos, mask_labels, mask_types]
                ins_loss, _ = model_normal.forward_E(ins_pos_normal, ins_edge_labels, "normal", 2 * confidence, correct_rate)
                ins_loss.backward()
                ins_optimizer_normal.step()
                ins_epoch_loss += ins_loss

                # print_ECS_loss_per_step
                if j % 100 == 0:
                    logger.info(str(j) + ' , ins_loss: ' + str(ins_loss.item()) + " with memory " + str(torch.cuda.memory_allocated(device=device) / 1024 / 1024))

            ins_epoch_loss /= steps
            end = time.time()
            t2 = round(end - start, 2)
            logger.info("ins_epoch_loss = {:.3f}, time = {:.3f} s".format(ins_epoch_loss, t2))


        # Start validation and testing
        with torch.no_grad():
            h2E = predict(
                model=model_normal,
                ins_test_pyreader=test_E_pyreader,
                ins_all_facts=ins_info['all_facts'],
                ins_edge_labels=ins_edge_labels,
                device=device)

            if iteration % args.error_detection_every_x_epochs == 0:
                print("accuracy_fact: ", (right_right_fact + false_false_fact) / (right_right_fact + false_false_fact + false_right_fact + right_false_fact))
                precision_fact = right_right_fact / (right_right_fact + false_right_fact)
                recall_fact = right_right_fact / (right_right_fact + right_false_fact)
                print("precision_fact: ", precision_fact)
                print("recall_fact: ", recall_fact)
                print("F1_fact: ", (2 * precision_fact * recall_fact) / (precision_fact + recall_fact))
                print("accuracy_element: ", (right_right_element + false_false_element) / (right_right_element + false_false_element + false_right_element + right_false_element))
                precision_element = right_right_element / (right_right_element + false_right_element)
                recall_element = right_right_element / (right_right_element + right_false_element)
                print("precision_element: ", precision_element)
                print("recall_element: ", recall_element)
                print("F1_element: ", (2 * precision_element * recall_element) / (precision_element + recall_element))

    logger.info("stop")


def predict(model, ins_test_pyreader,
            ins_all_facts,
            ins_edge_labels, device):
    start = time.time()

    step = 0
    ins_ret_ranks = dict()
    ins_ret_ranks['entity'] = torch.empty(0).to(device)
    ins_ret_ranks['relation'] = torch.empty(0).to(device)
    ins_ret_ranks['2-r'] = torch.empty(0).to(device)
    ins_ret_ranks['2-ht'] = torch.empty(0).to(device)
    ins_ret_ranks['n-r'] = torch.empty(0).to(device)
    ins_ret_ranks['n-ht'] = torch.empty(0).to(device)
    ins_ret_ranks['n-a'] = torch.empty(0).to(device)
    ins_ret_ranks['n-v'] = torch.empty(0).to(device)

    # while steps < max_train_steps:
    for i, data in enumerate(ins_test_pyreader):
        ins_pos = data
        length = data[0].shape[0]
        _, ins_np_fc_out = model.forward_E(ins_pos, ins_edge_labels, "normal", torch.ones(length).to("cuda:0"), 0)

        ins_ret_ranks = batch_evaluation(ins_np_fc_out, ins_pos, ins_all_facts, ins_ret_ranks, device)

        step += 1

    ins_eval_performance = compute_metrics(ins_ret_ranks)

    ins_all_entity = "ENTITY\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['entity']['mrr'],
        ins_eval_performance['entity']['hits1'],
        ins_eval_performance['entity']['hits3'],
        ins_eval_performance['entity']['hits5'],
        ins_eval_performance['entity']['hits10'])

    ins_all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['relation']['mrr'],
        ins_eval_performance['relation']['hits1'],
        ins_eval_performance['relation']['hits3'],
        ins_eval_performance['relation']['hits5'],
        ins_eval_performance['relation']['hits10'])

    ins_all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['ht']['mrr'],
        ins_eval_performance['ht']['hits1'],
        ins_eval_performance['ht']['hits3'],
        ins_eval_performance['ht']['hits5'],
        ins_eval_performance['ht']['hits10'])

    ins_all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['r']['mrr'],
        ins_eval_performance['r']['hits1'],
        ins_eval_performance['r']['hits3'],
        ins_eval_performance['r']['hits5'],
        ins_eval_performance['r']['hits10'])

    logger.info("\n-------- E Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join(["TASK\t", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        ins_all_ht, ins_all_r, ins_all_entity, ins_all_relation))

    end = time.time()
    logger.info("INS time: " + str(round(end - start, 3)) + 's')

    return ins_eval_performance['entity']['hits1']


if __name__ == '__main__':
    print_arguments(args)
    main(args)
