from nngraph.graph import Graph
import argparse
import pickle

def str_to_int_list(input):
    return [int(x) for x in input.strip('[]').split(',')]

parser = argparse.ArgumentParser()
parser.add_argument("--input_model_path", default='./models/model.onnx', type=str, help='onnx model input path')
parser.add_argument('--output_path', default='./nngraph/outputs', type=str, help='output images path')
parser.add_argument("--visualize_layers", default=True, type=bool, help="Shows visual representation of layers")
parser.add_argument("--visualize_operational", default=True, type=bool, help="Shows visual representation of operations")
parser.add_argument("--visualize_primitives", default=True, type=bool, help="Shows visual representation of primitives")
parser.add_argument("--input_shape", default="[28,28]", type=str_to_int_list, help="Input shape for torch model")

args = parser.parse_args()
args.do_lower_case = True

g=Graph(args.input_model_path, args.output_path, args.input_shape)
g.visualize(args.visualize_operational, args.visualize_layers, args.visualize_primitives)

with open("test.pkl", 'wb') as f:
    pickle.dump(g, f)   

with open('test.pkl', 'rb') as f:
    load_test = pickle.load(f)
    load_test.visualize(0,0,1)
