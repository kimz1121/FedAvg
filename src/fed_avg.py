from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MNISTDataset, FederatedSampler
from models import CNN, MLP
from utils import arg_parser, average_weights, Logger


class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = Logger(args)

        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )

        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=1, n_classes=10).to(self.device)
            self.target_acc = 0.99
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int

    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        train_set = MNISTDataset(root=root, train=True)
        test_set = MNISTDataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
        )

        train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        return train_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        model = copy.deepcopy(root_model)
        model.train()# 클라이언트 모델을 학습 모드로 설정, root_model은 이미 train 모드로 설정된 상태이므로 불필요 할 수 있어보이나, 안전하게 Train 모드로 설정하는 듯.
        # 아니면 nn.module의 초기화 과정중에서 다시 eval 모드로 바뀌는 경우가 있나?
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # optimizer 초기화 (gradient 0으로 초기화)

                logits = model(data)
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= idx+1 # 마지막 index를 클라이언트 수로 보고 평균을 취함.
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}".ljust(100),
                end="\r",
            )

        return model, epoch_loss / self.args.n_client_epochs

    def train(self) -> None:
        """Train a server model."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            # frac? 전체 클라이언트 중이 일부를 샘플링?
            # frac은 (0, 1] 사이의 실수 값으로 설정.
            # max를 사용하는 이유는 frac이 너무 작을 때 최소 1명은 선택되도록 하기 위함인 듯.

            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)
            # self.args.n_clients 전체 클라이언트 중 m개를 choice 샘플링
            

            # Train clients
            # Train mode를 활성화 해주었기 때문에 _train_client()함수 내부의 deepcopy된 모델도 train mode로 동작함.
            self.root_model.train()

            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)
                # train_loader는 client 수 만큼 할당.?
                # train_loader는 는 하나인데, 내부 sampler가 client에 맞게 샘플링을 조절하는 듯.
                # train_loader가 각 client마다 데이터를 partioning 하는 역할을 하는 건가?
                # 각 클라이언트들 마다 곂치지 않고 고유한 데이터를 갖도록 강제가 되나? 서로 데이터가 곂치는 일이 생기지는 않나?

                # Train client
                # 클라이언트 하나를 학습하는 과정을 표현한 코드
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            # root_model의 weights는 clients_models의 평균으로 업데이트됨.
            # 내부적으로 soft update가 아닌 완전 교체임.
            
            self.root_model.load_state_dict(updated_weights)
            # load_state_dict함수를 활용해 root_model의 weights를 updated_weights로 교체

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)# avg_loss는 logging 용도로만 활용되는 것으로 보임.

            # 이 아래는 로깅 및 테스트 코드
            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                self.logger.log(logs)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break
    
    # root_model을 테스트하는 함수 
    # 테스트 기능으 root_model에만 적용하고, client_model에는 적용하지 않는 듯
    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx+1
        total_acc = total_correct / total_samples
        print()# 줄바꿈 이전 출력 버퍼를 비우기 위한 용도
        print(f"Total Correct: {total_correct}, Total Samples: {total_samples}\n")

        return total_loss, total_acc


if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train()
