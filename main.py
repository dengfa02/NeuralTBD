from NeuralTBD import NeurlTBD
import argparse
import yaml
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='./configs/multiuav.yaml')
    parser.add_argument('--dataset', default='MultiUAV')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]  # multiuav
    config["dataset"] = args.dataset
    config = EasyDict(config)
    model = NeurlTBD(config, is_train=not config["eval_mode"])

    if config["eval_mode"]:
        model.eval()
    else:
        model.train()


if __name__ == '__main__':
    main()
