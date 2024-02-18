# NYLON (Robust HKG Model)

NYLON is a robust link prediction model over noisy hyper-relational knowledge graphs. Trained with an active learning strategy, NYLON evaluates the confidence of facts and rebalances the loss for each fact with its confidence to alleviate the negative impact of less confident facts.  Please see the details in our paper below:
- Weijian Yu, Jie Yang and Dingqi Yang. 2024. Robust Link Prediction over Noisy Hyper-Relational Knowledge Graphs via Active Learning. In Proceedings of the ACM Web Conference 2024 (WWW'24), May 13-17, 2024, Singapore.

## How to run the code
###### Train and evaluate model (suggested parameters for JF17k, WikiPeople and WD50K dataset)
```
python3 -u run.py --input "dataset/jf17k"

python3 -u run.py --input "dataset/wikipeople"

python3 -u run.py --input "dataset/wd50k"
```

###### Parameter setting:
In `run.py`, you can set:

`--input`: input dataset.

`--epochs`: number of training epochs.

`--batch_size`: batch size of training set.

`--learning_rate`: learning rate.

`--noise_level`: noise level in float, i.e., 1.0 refers to generate 100% noisy facts of positive facts.

`--active_sample_per_epoch`: active labeling budget in float, i.e., 0.0025 refers to labeling 0.25% elements per epoch.

`--aug_amount`: number of pseudo-labeled positive facts for every positive fact in training of confidence evaluator. For more detail please refer to our paper.

# Python lib versions
Python: 3.10.12

torch: 2.1.0

# Reference
If you use our code or datasets, please cite:
```
@inproceedings{yu2024robust,
  title={Robust Link Prediction over Noisy Hyper-Relational Knowledge Graphs via Active Learning},
  author={Yu, Weijian and Yang, Jie and Yang, Dingqi},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={},
  year={2024}
}
```
