import argparse
import torch

parser = argparse.ArgumentParser()

def navi_list(s):
    try:
        navi_strings = s.split(',')
        navi_list = []
        for navi in navi_strings:
            navi_list.append(navi)
        return sorted(navi_list)
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y,z")


parser.add_argument("--dataset", type=str, default="aifb", help="dataset ID/name")
parser.add_argument("--embedding_engine", default="pyRDF2Vec", help="Name of the used embedding engine")
parser.add_argument("--embedding_file", default="standard", help="ID of the initial embedding file")
parser.add_argument("--no_cuda", action="store_true", default=False, help="Enables CUDA training.")
parser.add_argument("--loss_name", default="Inv-MSE", help="Name of the Loss Function")
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument("--drop", type=float, default=0.3, help="Dropout of RGCN")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--early_stop", type=int, default=10, help="Early stop count.")
parser.add_argument("--validation", action="store_false", default=True, help="Run validation data.")
parser.add_argument("--using_bias", action="store_false", default=True, help="Use Bias in the Manager Layer.")
parser.add_argument("--manager_name", type=str, default='wenger', help="Name of the NaviManager.")
parser.add_argument("--split_type", type=str, default='meta', help="Type of data split procedure")
parser.add_argument('--navi_list', type=navi_list, help="Navi List", default='iniesta,diego')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
