import copy


class DemonstrationSteps:
    def __init__(self):
        self.steps = []
        self.active_step_index = -1

    def add_step(self, layers, action, action_data):
        new_layers = []
        for l in layers:
            new_layers.append(copy.deepcopy(l))
        step = {
            "layers": new_layers,
            "action": action,
            "action_data": action_data
        }
        self.steps.append(step)

    def next_step(self):
        if self.active_step_index + 1 < len(self.steps):
            self.active_step_index += 1
        return self.steps[self.active_step_index]

    def prev_step(self):
        if self.active_step_index - 1 >= 0:
            self.active_step_index -= 1
        return self.steps[self.active_step_index]

    @staticmethod
    def action_calculate_neuron(layer_index, neuron_index):
        return "CALCULATE_NEURON", {"l_i": layer_index, "n_i": neuron_index}

    @staticmethod
    def action_activate_neuron(layer_index, neuron_index):
        return "ACTIVATE_NEURON", {"l_i": layer_index, "n_i": neuron_index}

    @staticmethod
    def action_load_input(inputs, output, prev_in, prev_o):
        return "LOAD_INPUT", {"input": copy.deepcopy(inputs), "output": copy.deepcopy(output), "prev_i": copy.deepcopy(prev_in), "prev_o": copy.deepcopy(prev_o)}

    @staticmethod
    def action_calculate_delta(layer_index, neuron_index):
        return "CALCULATE_DELTA", {"l_i": layer_index, "n_i": neuron_index}

    @staticmethod
    def action_calculate_weights(layer_index, neuron_index, xi):
        return "CALCULATE_WEIGHTS", {"l_i": layer_index, "n_i": neuron_index, "x_i": xi}

    @staticmethod
    def action_end_epoch(l_in, l_out):
        return "END_EPOCH", {"prev_i": copy.deepcopy(l_in), "prev_o": copy.deepcopy(l_out)}
