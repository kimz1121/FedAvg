from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import wandb


class Logger:
    def __init__(self, args):
        self.args = args
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb

    def log(self, logs: Dict[str, Any]) -> None:
        if self.wandb:
            self.wandb.log(logs)


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])
    # weights_avg를 저장할 위치를 사전할당하는 목적인 듯
    # 이 구현은 copy.deepcopy 말고, weights_avg = {key: torch.zeros_like(value) for key, value in weights[0].items()} 로도 가능할 듯.

    # 왜 dict 형태로 관리되지?
    # key는 각 레이어의 이름, value는 해당 레이어의 파라미터(weight, bias) 텐서인 듯.
    for key in weights_avg.keys():
        for i in range(1, len(weights)):# len(weights)는 클라이언트 수와 동일
            weights_avg[key] += weights[i][key] # 같은 레이어(같은 key 값)끼리 모두 더함
        weights_avg[key] = torch.div(weights_avg[key], len(weights))  # 더한 값을 클라이언트 수로 나누어 평균을 구함
        # 이를 전체 레이어(key 값)에 대해 반복

        # 모델 전체의 파라미터의 평균을 구하게 됨.
    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")

    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    return parser.parse_args()
