import json
import unicodedata
import string
import re
import random
import time
import math
import ast
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
import os
import pickle
from random import shuffle

from fix_label import *

# Note: hospital and police domais only occur in the training set so we can leave them out
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

SLOT_TO_NATURAL = {
    "leaveat" : "leave at",
    "pricerange" : "price range",
    "arriveby" : "arrive by"
}


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    for i in range(len(SLOTS)):
        domain, slot = SLOTS[i].split("-")
        if slot in SLOT_TO_NATURAL:
            slot = SLOT_TO_NATURAL[slot]
        SLOTS[i] = domain + "-" + slot
    return SLOTS


def read_file(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None, args = {"except_domain" : "", "only_domain" : ""}):
    """
    Reads examples from train / dev / test files

    Acknowledgement: most of this code is taken from the trade-dst repo (https://github.com/jasonwu0731/trade-dst)
    implementation of the function read_langs
    """
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {}
    with open(file_name) as f:
        dials = json.load(f)
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Note: this does nothing if only_domain = "" and except_domain = ""
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                agent_utt = ""
                user_utt = "[User]: {0}".format(turn["transcript"])
                if len(turn["system_transcript"]):
                    agent_utt = "[Agent]: {0}".format(turn["system_transcript"])
                    turn_uttr = agent_utt+" ; "+user_utt
                else:
                    turn_uttr = user_utt
                # user transcript is always second, sys transcript is always first
                turn_uttr_strip = turn_uttr.strip()
                # history contains all systranscript ; user transcripts
                dialog_history +=  (turn_uttr_strip + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                for i in range(len(turn_belief_list)):
                    domain, label, value = turn_belief_list[i].split("-")
                    if label in SLOT_TO_NATURAL:
                        label = SLOT_TO_NATURAL[label]
                    turn_belief_list[i] = domain + "-" + label + "-" + value

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])
                
                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "dialog_history":source_text, 
                    "turn_belief":turn_belief_list,
                    "turn_uttr":turn_uttr_strip
                    }
                data.append(data_detail)
                
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())
                
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp



def extract_dataset_instances(data_dir, task="dst"):
    """
    Returns a dictionary of train/dev/test instances for the given task
    """
    save_dir = os.path.join(data_dir, "splits")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_train = os.path.join(data_dir, 'train_dials.json')
    file_dev =  os.path.join(data_dir, 'dev_dials.json')
    file_test = os.path.join(data_dir, 'test_dials.json')

    ontology = json.load(open(os.path.join(data_dir, "multi-woz/MULTIWOZ2.1/ontology.json"), 'r'))
    ALL_SLOTS = get_slot_information(ontology)

    gating_dict = {"ptr":0, "dontcare": "dontcare", "none": "none"}
    args = {"except_domain" : "", "only_domain" : ""}
    pair_train, train_max_len, slot_train = read_file(file_train, gating_dict, ALL_SLOTS, "train", None, None, None, None, args=args)
    pair_dev, dev_max_len, slot_dev = read_file(file_dev, gating_dict, ALL_SLOTS, "dev", None, None, None, None, args=args)
    pair_test, test_max_len, slot_test = read_file(file_test, gating_dict, ALL_SLOTS, "test", None, None, None, None, args=args)

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])

    train_dict = {"split" : "train", "examples" : pair_train, "max_len" : train_max_len, "slots" : slot_train}
    dev_dict = {"split" : "dev", "examples" : pair_dev, "max_len" : dev_max_len, "slots" : slot_dev}
    test_dict = {"split" : "test", "examples" : pair_test, "max_len" : test_max_len, "slots" : slot_test}

    json.dump(train_dict, open(os.path.join(save_dir, "train.json"), "w"))
    json.dump(dev_dict, open(os.path.join(save_dir, "dev.json"), "w"))
    json.dump(test_dict, open(os.path.join(save_dir, "test.json"), "w"))


if __name__ == "__main__":
    extract_dataset_instances()
