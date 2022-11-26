import tkinter as tk
import argparse
import Layer
import Network



n = Network.Network()
def key_right(e):
    n.next_step()
def key_left(e):
    n.prev_step()



if __name__ == "__main__":
    # net = Network.Network()
    # l1 = Layer.Layer()
    # net.add_layer(Layer)
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='SFC - Demonstration. Program launch GUI to show step by step how neural network learns using BackPropagation and activation function softmax on all layers',
        epilog='')
    parser.add_argument('-i', default=9, type=int, help="input size neural network input")
    parser.add_argument('-o', default=4, type=int, help="output size neural network output")
    parser.add_argument('--custom-input', nargs=1, help="input for demonstration - use file", default="./input.txt")
    parser.add_argument('--custom-output', nargs=1, help="output for demonstration - use file", default="./output.txt")
    parser.add_argument('-l', '--layers', nargs='+', required=False, default=[], type=int, help="list of number of neurons except last layer, also number of provided numbers will generate layers")
    args = parser.parse_args()
    inputs = []
    outputs = []
    with open(args.custom_input, 'r') as fi:
        for i in fi.readlines():
            inputs.append(list(map(int, i.replace("\n",'').split())))
    with open(args.custom_output, 'r') as fo:
        for i in fo.readlines():
            outputs.append(list(map(int, i.replace("\n",'').split())))
    print(inputs)
    next_input = int(args.i)
    if len(args.layers) == 0:
        n.add_layer(Layer.Layer(next_input, int(args.o), "L0"))
        print(next_input, args.o)
    else:
        for i, pn in enumerate(args.layers):
            pn = int(pn)
            n.add_layer(Layer.Layer(next_input, pn, f"L{i}"))
            next_input = pn
        n.add_layer(Layer.Layer(next_input, int(args.o), f"L{len(args.layers)}"))
    # l1 = Layer.Layer(9, 4)
    # l2 = Layer.Layer(8, 4, name="L2")
    # l3 = Layer.Layer(3, 4, name="L3")
    # n.add_layer(l1)
    # n.add_layer(l2)
    # n.add_layer(l3)
    window = tk.Tk()
    window.bind('<Right>', key_right)
    window.bind('<Left>', key_left)
    n.draw(window, 1200, 800)
    n.draw_lines_full_connect()
    n.generate_steps(inputs, outputs)
    window.mainloop()
