from calflops import calculate_flops

class ModelAnalysis:
    def __init__(self, model, input_shape=(1, 3, 224, 224)):
        self.model = model
        self.input_shape = input_shape

    def get_flops_macs_params(self):
        flops, macs, params = calculate_flops(
            model=self.model,
            input_shape=self.input_shape,
            output_as_string=True,
            output_precision=4
        )
        return params, macs, flops

    def get_analysis(self):
        params, macs, flops = self.get_flops_macs_params()
        return params, macs, flops
