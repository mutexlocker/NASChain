class ValidationConfig:
    def __init__(self):
        self.min_parameters = 0
        self.max_parameters = 0
        self.min_flops = 0
        self.max_flops = 0
        self.min_accuracy = 0
        self.max_accuracy = 0
        self.max_download_file_size = 5*1024*1024
        self.train_epochs = 50