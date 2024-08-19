from nngraph.graph import Graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_pkl_path", default='./models/model1.pkl', type=str, help='pikle input path')
parser.add_argument('--output_path', default='./nngraph/outputs', type=str, help='output images path')
parser.add_argument("--config_file_path", default='./nngraph/layer_config.json', type=str, help="config file path")
parser.add_argument("--excluded_params", default=["weight", "in_proj_weight", "in_proj_bias", "bias"], type=list, help="Parameters to exclude when building graph")
parser.add_argument("--visualize_torch_funcs", default=False, type=bool, help="Shows visual representation of torch functions")
parser.add_argument("--visualize_layers", default=True, type=bool, help="Shows visual representation of layers")
parser.add_argument("--visualize_operational", default=True, type=bool, help="Shows visual representation of operations")

args = parser.parse_args()
args.do_lower_case = True

g=Graph(args.input_pkl_path, args.output_path, args.config_file_path, args.excluded_params)
g.visualize(args.visualize_operational, args.visualize_layers, args.visualize_torch_funcs)
