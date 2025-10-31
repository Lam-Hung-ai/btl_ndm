import argparse

from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--folder", "-f", help="Nhập thư mục lưu bộ dữ liệu CIFAR10", default="./cifar_10")

arg = parser.parse_args()
CIFAR10(root=arg.folder, train=True, download=True)