from calflops import calculate_flops

class ModelAnalysis:
    def __init__(self, model, input_shape=(1, 3, 32, 32)):
        self.model = model.eval()
        self.input_shape = input_shape

    def get_flops_macs_params(self):
        flops, macs, params = calculate_flops(
            model=self.model,
            input_shape=self.input_shape,
            output_as_string=False,
            output_precision=4,
            print_detailed=False,
        )
        return params, macs, flops

    def get_analysis(self):
        params, macs, flops = self.get_flops_macs_params()
        return params, macs, flops
