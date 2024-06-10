<div align="center">


---
<img src="imgs/naschain_logo.png" alt="NASchain logo" width="700" height="300">


[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
## Introduction

Neural Architecture Search (NAS) is a critical field in machine learning that focuses on automating the design of artificial neural network architectures. As deep nerual network models become increasingly complex and computationally expensive, the significance of NAS grows. The primary goal of NAS is to identify the optimal model that not only maximizes accuracy for a given use-case but also minimizes the number of parameters and the computational cost, measured in Floating Point Operations (FLOPs). However, performing such searches can be very resource-intensive, often requiring days or weeks of computation on hundreds of GPUs to find an optimal model.

NASChain aims to address these challenges by leveraging the power of the Bittensor network and an innovative incentive mechanism. This approach distributes NAS tasks among participants (referred to as miners), thereby decentralizing the computational effort and potentially reducing the time and resources required for finding efficient and effective neural architectures.

---
## How it works

1. **Miners Running NAS Algorithm:** Miners execute the Neural Architecture Search (NAS) algorithm on the dataset described by the sample mining code. The objective of the NAS is to minimize the number of parameters while maximizing accuracy on the test set.
   
2. **Model Submission:** Miners upload their best models to Hugging Face with the miner code and submit the metadata for the commit to the blockchain.

3. **Validation Process:** Validators sync with the blockchain, download all models from the miners, and evaluate them on the test set. Architectures that lie on the Pareto Optimal line will have their weights reset and undergo further training by validators on the standard train/valid set to ensure no test set leakage occurred during the miners' model training.

4. **Rewards for Miners:** Miners who produce models that lie on the Pareto Optimal line will be rewarded.



<div align="center">
<img src="imgs/subnet31_v2.svg" alt="NASChain V2" width="960" height="770">
</div>


## Hardware Requirements

### Miners:
- **GPU**: Miners can select any open-source NAS algorithm to find the best architectures. The NAS project, depending on the design, can support single or multiple GPUs, giving the miner the ability to speed up the NAS runtime.

### Validators:
- **GPU**: TODO

---
## Installation

We recommend using virtual environments such as Conda to manage and isolate your project dependencies.

- Ensure you have Python >= 3.10 installed on your system.
- Both Miner and Validator code is only tested on Linux OS.
- It is advised to use a virtual environment to avoid conflicts with other projects or system-wide packages.

### Runing Miner and Validator

1. Clone the repository:
   ```bash
   git clone https://github.com/nimaaghli/NASChain
2. Navigate to the project directory:
    ```bash
    cd NASChain
3. if setting up virtual enviuuement(Skip this step if running python on system level):
    - if using conda:
        ```bash
        conda create --name myenv python=3.10
        conda activate myenv
    - if using venv
        ```bash
        python -m venv env
        source env/bin/activate
4. Install the required packages:
    ```bash
    pip install -r requirements.txt

5. **Running the Miner:**
   1. Create a Hugging Face account.
   2. Create a write token and export it as an environment variable:
      ```bash
      export HF_ACCESS_TOKEN="YOUR_HG_WRITE_TOKEN"
      ```
   3. Run a Miner to Train a Dummy Model (You Can Manually Modify the Architecture):
      ```bash
      python neurons/miner.py --netuid 31 --wallet.name <wallet_cold_name> --wallet.hotkey <wallet_hot_name> --logging.debug --hf_repo_id <your_hf_repo_id>
      ```
   4. Run a Miner with a Pretrained PyTorch Model (Model Exported by NAS in a Different Directory):
      ```bash
      python neurons/miner.py --netuid 31 --wallet.name <wallet_cold_name> --wallet.hotkey <wallet_hot_name> --logging.debug --hf_repo_id <your_hf_repo_id> --model.dir path/to/mode/model.pt
      ```

6. **Running the Validator:**
   ```bash
   TODO
