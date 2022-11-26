import numpy as np
import tkinter as tk
import Layer
from DemonstrationSteps import DemonstrationSteps


class Network:
    def __init__(self):
        self.layers = []
        self.canvas = None
        self.neuron_size = 40
        self.neuron_margin = 10
        self.highlighted_lines = []
        self.neuron_drawed_values = []
        self.input_locations = []
        self.output_location = []
        self.drawed_input = []
        self.drawed_output = []
        self.steps = DemonstrationSteps()
        self.expression = None
        self.all_lines = []


    def add_layer(self, layer: Layer.Layer):
        self.layers.append(layer)
        neurons = []
        for _ in layer.neurons:
            neurons.append(None)
        self.neuron_drawed_values.append(neurons)

    def train_one_input(self, one_input, output):
        current_input = one_input
        for index, layer in enumerate(self.layers):
            layer.calculate_neurons(current_input)
            current_input = layer.calculate_activation()
        self.layers[len(self.layers) - 1].calculate_delta(None, output)
        self.layers[len(self.layers) - 1].calculate_weights(self.layers[len(self.layers) - 2].activation)

        for index in range(len(self.layers) - 2, -1, -1):
            self.layers[index].calculate_delta(self.layers[index + 1].delta, None, self.layers[index + 1].weights)
            self.layers[index].calculate_weights(one_input if index == 0 else self.layers[index - 1].activation)

    def evaluate(self, n_input):
        current_input = n_input
        for index, layer in enumerate(self.layers):
            layer.calculate_neurons(current_input)
            current_input = layer.calculate_activation()
        return current_input

    def draw(self, window, width, height):
        number_of_cols = len(self.layers) + 2
        width_of_col = width//number_of_cols
        center_of_col = width_of_col // 2 - self.neuron_size//2
        center_vertical = height // 2

        print(width_of_col)
        window.geometry(f"{width}x{height}")
        self.canvas = tk.Canvas(window, width=width, height=height)
        for li, l in enumerate(self.layers):
            xl = center_of_col + (width_of_col * (li + 1))
            neurons_count = len(l.neurons)
            y_first = center_vertical - (self.neuron_margin * (neurons_count//2)) - (self.neuron_size * (neurons_count//2 + 0.5)) if neurons_count % 2 != 0 else center_vertical - ( self.neuron_margin * (neurons_count//2-0.5)) - (self.neuron_size * (neurons_count//2))
            for ni, n in enumerate(l.neurons):
                yn = int(y_first + ((self.neuron_size + self.neuron_margin) * ni))
                l.neuron_location.append((xl, yn, xl + self.neuron_size, yn + self.neuron_size))
                self.canvas.create_oval(xl, yn, xl + self.neuron_size, yn + self.neuron_size)
                print(f"{li} - {ni}", xl, yn)

        input_count = self.layers[0].l_input.shape[0]
        output_count = self.layers[len(self.layers) - 1].neurons.shape[0]

        input_x = center_of_col
        y_first = center_vertical - (self.neuron_margin * (neurons_count//2)) - (
                    self.neuron_size * (input_count // 2 + 0.5)) if input_count % 2 != 0 else center_vertical - (
                    self.neuron_margin * (input_count // 2 - 0.5)) - (self.neuron_size * (input_count // 2))
        for i in range(input_count):
            yin = y_first + (i*(self.neuron_size + self.neuron_margin))
            self.canvas.create_rectangle(input_x, yin, input_x+self.neuron_size, yin+self.neuron_size)
            self.input_locations.append((input_x, yin, input_x+self.neuron_size, yin+self.neuron_size))

        input_x = center_of_col + width_of_col * (len(self.layers)+1)
        y_first = center_vertical - (self.neuron_margin * (neurons_count//2)) - (
                self.neuron_size * (output_count // 2 + 0.5)) if output_count % 2 != 0 else center_vertical - (
                self.neuron_margin * (output_count // 2 - 0.5)) - (self.neuron_size * (output_count // 2))
        for i in range(output_count):
            yin = y_first + (i*(self.neuron_size + self.neuron_margin))
            self.canvas.create_rectangle(input_x, yin, input_x+self.neuron_size, yin+self.neuron_size)
            self.output_location.append((input_x, yin, input_x+self.neuron_size, yin+self.neuron_size))
        self.expression = self.canvas.create_text(width // 2, 30, text="", justify=tk.CENTER, fill="#000")
        self.canvas.pack()

    def change_expression(self, text):
        self.canvas.itemconfig(self.expression, text=text)
    def draw_line_input_to_neuron(self, input_index, neuron_index, color="#c5c5c5", to_clear = True):
        line = self.canvas.create_line(self.input_locations[input_index][2], self.input_locations[input_index][3] - (self.neuron_size // 2), self.layers[0].neuron_location[neuron_index][0],
                                self.layers[0].neuron_location[neuron_index][1] + (self.neuron_size // 2), fill=color)
        if to_clear:
            self.all_lines.append(line)

    def draw_line_between(self, start_l_index, neuron_l_index, neuron_l_next_index, color="#c5c5c5", to_clear=True):
        if 0 > start_l_index >= len(self.layers)-1:
            print(f"Bad call for draw_line_between, from layer {start_l_index}!")
            return
        l_from = self.layers[start_l_index]
        l_to = self.layers[start_l_index+1]
        if 0 > neuron_l_index >= len(l_from.neuron_location):
            print(f"Bad call for draw_line_between, from neuron {neuron_l_index}!")
            return
        neuron_from = l_from.neuron_location[neuron_l_index]
        if 0 > neuron_l_next_index >= len(l_to.neuron_location):
            print(f"Bad call for draw_line_between, to neuron {neuron_l_next_index}!")
            return
        neuron_to = l_to.neuron_location[neuron_l_next_index]
        line =self.canvas.create_line(neuron_from[2], neuron_from[3]-(self.neuron_size//2), neuron_to[0], neuron_to[1]+(self.neuron_size//2), fill=color)
        if to_clear:
            self.all_lines.append(line)

    def draw_lines_full_connect(self):
        for i in range(len(self.layers)-1):
            for n1 in range(len(self.layers[i].neuron_location)):
                for n2 in range(len(self.layers[i+1].neuron_location)):
                    self.draw_line_between(i, n1, n2, to_clear=False)
        for inp in range(len(self.input_locations)):
            print(inp)
            for n in range(len(self.layers[0].neuron_location)):
                self.draw_line_input_to_neuron(inp, n, to_clear=False)

    def unhighlight_values(self):
        for hl in self.neuron_drawed_values:
            for onehl in hl:
                self.canvas.itemconfig(onehl, fill="#5c5c5c")
    def unhighlight_output(self):
        for hl in self.drawed_output:
            self.canvas.itemconfig(hl, fill="#000")
    def unhightlight_lines(self):
        for hl in self.highlighted_lines:
            if hl[0] != "INPUT":
                self.draw_line_between(*hl)
            else:
                self.draw_line_input_to_neuron(hl[1], hl[2])
        self.highlighted_lines = []

    def hightlight_line_input(self, input_index, neuron_index, unhighlight = True):
        if unhighlight:
            self.unhightlight_lines()
        self.draw_line_input_to_neuron(input_index, neuron_index, "#ff0000")
        self.highlighted_lines.append(("INPUT", input_index, neuron_index))

    def highlight_line(self, start_l_index, neuron_l_index, neuron_l_next_index, unhighlight = True):
        if unhighlight:
            self.unhightlight_lines()

        self.draw_line_between(start_l_index, neuron_l_index, neuron_l_next_index, "#ff0000")
        self.highlighted_lines.append((start_l_index, neuron_l_index, neuron_l_next_index))

    def highlight_neuron_value(self, l_index, n_index, value, unhighlight=True):
        if unhighlight:
            self.unhighlight_values()
        if self.neuron_drawed_values[l_index][n_index] is not None:
            self.canvas.itemconfig(self.neuron_drawed_values[l_index][n_index], fill="#f00", text=str(round(value, 2)))
        else:
            self.draw_neuron_value(l_index, n_index, value, "#f00")

    def draw_neuron_value(self, l_index, n_index, value, color="#000"):
        if 0 > l_index >= len(self.layers) and 0 > n_index >= len(self.layers[l_index].neuron_location):
            print(f"draw_neuron_value - Layer index {l_index} or neuron index {n_index} not found!")
            return

        neuron_loc = self.layers[l_index].neuron_location[n_index]
        halfNeuronSize = self.neuron_size//2
        label = (neuron_loc[0] + halfNeuronSize, neuron_loc[1] + halfNeuronSize)
        text = self.canvas.create_text(label[0], label[1], justify=tk.CENTER, text=str(round(value, 2)), fill=color)
        self.neuron_drawed_values[l_index][n_index] = text

    def draw_inputs_value(self, values):
        halfNeuronSize = self.neuron_size // 2
        for i, v in enumerate(values):
            input_loc = self.input_locations[i]
            label = (input_loc[0] + halfNeuronSize, input_loc[1] + halfNeuronSize)
            text = self.canvas.create_text(label[0], label[1], justify=tk.CENTER, text=str(round(v, 2)))
            self.drawed_input.append(text)
    def draw_outputs_value(self, values):
            halfNeuronSize = self.neuron_size // 2
            for i, v in enumerate(values):
                input_loc = self.output_location[i]
                label = (input_loc[0] + halfNeuronSize, input_loc[1] + halfNeuronSize)
                text = self.canvas.create_text(label[0], label[1], justify=tk.CENTER, text=str(round(v, 2)))
                self.drawed_output.append(text)

    def clear_input(self):
        for di in self.drawed_input:
            self.canvas.delete(di)
    def clear_output(self):
        for di in self.drawed_output:
            self.canvas.delete(di)

    def generate_steps(self, inputs, outputs):
        prev_input = inputs[0]
        prev_output = outputs[0]
        for input_index, inp in enumerate(inputs):
            current_input = inp

            action, actionData = DemonstrationSteps.action_load_input(inp, outputs[input_index], prev_input, prev_output)
            prev_input = inp
            prev_output = outputs[input_index]
            self.steps.add_step(self.layers, action, actionData)

            for index, layer in enumerate(self.layers):
                layer.calculate_neurons(current_input)
                for ni, n in enumerate(layer.neurons):
                    action, actionData = DemonstrationSteps.action_calculate_neuron(index, ni)
                    self.steps.add_step(self.layers, action, actionData)
                current_input = layer.calculate_activation()
                for ni, n in enumerate(layer.neurons):
                    action, actionData = DemonstrationSteps.action_activate_neuron(index, ni)
                    self.steps.add_step(self.layers, action, actionData)

            self.layers[len(self.layers) - 1].calculate_delta(None, outputs[input_index])
            for ni, n in enumerate(self.layers[len(self.layers) - 1].neurons):
                action, actionData = DemonstrationSteps.action_calculate_delta(len(self.layers) - 1, ni)
                self.steps.add_step(self.layers, action, actionData)

            self.layers[len(self.layers) - 1].calculate_weights(self.layers[len(self.layers) - 1].l_input)
            for ni, n in enumerate(self.layers[len(self.layers) - 1].neurons):
                for xi in range(len(self.layers[len(self.layers) - 1].l_input)):
                    action, actionData = DemonstrationSteps.action_calculate_weights(len(self.layers) - 1, ni, xi)
                    self.steps.add_step(self.layers, action, actionData)

            for index in range(len(self.layers) - 2, -1, -1):
                self.layers[index].calculate_delta(self.layers[index + 1].delta, None, self.layers[index + 1].weights)
                for ni, n in enumerate(self.layers[index].neurons):
                    action, actionData = DemonstrationSteps.action_calculate_delta(index, ni)
                    self.steps.add_step(self.layers, action, actionData)
                self.layers[index].calculate_weights(inp if index == 0 else self.layers[index - 1].activation)
                for ni, n in enumerate(self.layers[index].neurons):
                    for xi in range(len(self.layers[index].l_input)):
                        action, actionData = DemonstrationSteps.action_calculate_weights(index, ni, xi)
                        self.steps.add_step(self.layers, action, actionData)
        action, actionData = DemonstrationSteps.action_end_epoch(prev_input, prev_output)
        self.steps.add_step(self.layers, action, actionData)

    def clear_lines(self):
        for l in self.all_lines:
            self.canvas.delete(l)
        self.all_lines = []

    def show_step(self, step):
        if step["action"] == "LOAD_INPUT":
            self.demonstrate_load_input(step["layers"], step["action_data"])
        elif step["action"] == "CALCULATE_NEURON":
            self.demonstrate_calculate_neuron(step["layers"], step["action_data"])
        elif step["action"] == "ACTIVATE_NEURON":
            self.demonstrate_activate_neuron(step["layers"], step["action_data"])
        elif step["action"] == "CALCULATE_DELTA":
            self.demonstrate_calculate_delta(step["layers"], step["action_data"])
        elif step["action"] == "CALCULATE_WEIGHTS":
            self.demonstrate_calculate_weights(step["layers"], step["action_data"])
        elif step["action"] == "END_EPOCH":
            self.demonstrate_end_epoch(step["layers"])
        else:
            print("Undefined step!")

    def demonstrate_load_input(self, layers, action_data):
        self.clear_input()
        self.clear_output()
        self.unhightlight_lines()
        self.unhighlight_values()
        self.clear_lines()
        self.draw_lines_full_connect()
        self.layers = layers
        self.draw_inputs_value(action_data["input"])
        self.change_expression("Load training data and initialize all weights on random (0,1)")
        self.draw_outputs_value(action_data["output"])

    def demonstrate_end_epoch(self, layers):
        self.layers = layers
        self.clear_lines()
        self.unhightlight_lines()
        self.unhighlight_values()
        self.change_expression("END: All inputs have been seen.")
        self.clear_input()
        self.clear_output()

    def demonstrate_calculate_neuron(self, layers, action_data):
        self.layers = layers
        l_i = action_data["l_i"]
        n_i = action_data["n_i"]
        calculation = ""
        self.unhightlight_lines()
        self.clear_lines()
        if l_i == 0:
            last = len(self.input_locations) - 1
            for inp in range(len(self.input_locations)):
                self.hightlight_line_input(inp, n_i, inp == 0)
                calculation += f"{round(self.layers[l_i].l_input[inp],3)} * {round(self.layers[l_i].weights[n_i,inp], 3)} {'+ ' if inp != last else ''}"
        else:
            last = len(self.layers[l_i-1].neurons) - 1
            for inp in range(len(self.layers[l_i-1].neurons)):
                self.highlight_line(l_i-1, inp, n_i, False)
                calculation += f"{round(self.layers[l_i].l_input[inp],3)} * {round(self.layers[l_i].weights[n_i, inp], 3)} {'+ ' if inp != last else ''}"
        self.change_expression(
            f"Calculate neuron {n_i} on layer {l_i} based on input and weights\n{calculation}")
        self.highlight_neuron_value(l_i, n_i, self.layers[l_i].neurons[n_i])
    def demonstrate_activate_neuron(self, layers, action_data):
        self.layers = layers
        l_i = action_data["l_i"]
        n_i = action_data["n_i"]
        self.unhightlight_lines()
        self.clear_lines()
        calculation = f""
        last = len(self.layers[l_i].neurons) -1
        for i in range(len(self.layers[l_i].neurons)):
            calculation += f"(e**{round(self.layers[l_i].neurons[i],3)}) {'+ ' if i != last else ''}"
        calculation = f"(e**{round(self.layers[l_i].neurons[n_i], 3)})/{calculation}"
        self.change_expression(
            f"Use activation function softmax on neuron {n_i} on layer {l_i}\n{calculation}")
        self.highlight_neuron_value(l_i, n_i, self.layers[l_i].activation[n_i])
    def demonstrate_calculate_delta(self, layers, action_data):
        self.unhighlight_output()
        self.layers = layers
        l_i = action_data["l_i"]
        n_i = action_data["n_i"]
        self.unhightlight_lines()
        self.clear_lines()
        self.highlight_neuron_value(l_i, n_i, self.layers[l_i].activation[n_i])
        if l_i == (len(self.layers) -1):
            self.canvas.itemconfig(self.drawed_output[n_i], fill="#f00")
            calculation = f"({self.layers[l_i].expected[n_i]} - {round(self.layers[l_i].activation[n_i], 3)}) * {round(self.layers[l_i].activation[n_i], 3)} * (1 - {round(self.layers[l_i].activation[n_i], 3)}) = {round(self.layers[l_i].delta[n_i], 3)}"
        else:
            self.highlight_neuron_value(l_i, n_i, self.layers[l_i].activation[n_i])
            calculation = f""
            last = len(self.layers[l_i+1].neurons) -1
            for n_ip in range(last+1):
                calculation += f"{round(self.layers[l_i+1].delta[n_ip], 3)} * {round(self.layers[l_i+1].weights.T[n_i, n_ip], 3)} {'+ ' if n_ip != last else ''}"
                self.highlight_line(l_i, n_i, n_ip, False)
            calculation = f"({calculation}) * {round(self.layers[l_i].activation[n_i], 3)} * (1 - {round(self.layers[l_i].activation[n_i], 3)}) = {round(self.layers[l_i].delta[n_i], 3)}"

        self.change_expression(
            f"Calculate delta on neuron {n_i} on layer {l_i}\n{calculation}")
    def demonstrate_calculate_weights(self, layers, action_data):
        self.layers = layers
        l_i = action_data["l_i"]
        n_i = action_data["n_i"]
        x_i = action_data["x_i"]
        self.unhightlight_lines()
        self.unhighlight_values()
        calculation = f"{round(self.layers[l_i].lr,3)} * {round(self.layers[l_i].delta[n_i],3)} * {round(self.layers[l_i].l_input[x_i],3)} = {round(self.layers[l_i].lr * self.layers[l_i].delta[n_i] * self.layers[l_i].l_input[x_i],8)}"

        if l_i == 0:
            # INPUT
            self.hightlight_line_input(x_i, n_i, False)
            calculation = f"Calculate weight from input {x_i} to neuron {n_i} between input and {l_i} layer with delta {self.layers[l_i].delta[n_i]}\n{calculation}"
        else:
            self.highlight_line(l_i - 1, x_i, n_i, False)
            calculation = f"Calculate weight from input {x_i} to neuron {n_i} between layers {l_i - 1} and {l_i} with delta {self.layers[l_i].delta[n_i]}\n{calculation}"

        self.change_expression(calculation)
    def next_step(self):
        current_step = self.steps.next_step()
        self.show_step(current_step)

    def prev_step(self):
        if self.steps.steps[self.steps.active_step_index]['action'] == "LOAD_INPUT" or self.steps.steps[self.steps.active_step_index]['action'] == "END_EPOCH":
            self.demonstrate_load_input(self.layers, {"input": self.steps.steps[self.steps.active_step_index]['action_data']['prev_i'], "output": self.steps.steps[self.steps.active_step_index]['action_data']['prev_o']})
        prev_step = self.steps.prev_step()
        self.show_step(prev_step)