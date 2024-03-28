<div align="center">


---
<img src="imgs/naschain.png" alt="Optional Image Description" width="700" height="300">


[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
  - [Setting Up](#setting-up)
  - [Running the Miner](#running-the-miner)
  - [Running the Validator](#running-the-validator)
- [Self-Improvement Mechanism](#self-improvement-mechanism)
- [Validation and Incentive Mechanism](#validation-and-incentive-mechanism)
- [Roadmap](#roadmap)
---

## Introduction

Neural Architecture Search (NAS) is a critical field in machine learning that focuses on automating the design of artificial neural network architectures. As computational models become increasingly complex and computationally expensive, the significance of NAS grows. The primary goal is to identify the optimal model that not only maximizes accuracy for a given use case but also minimizes the number of parameters and the computational cost, measured in Floating Point Operations (FLOPs). However, performing such searches can be immensely resource-intensive, often requiring days or weeks of computation on hundreds of GPUs to find an optimal model.

NASChain aims to address these challenges by leveraging the power of the Bittensor network and an innovative incentive mechanism. This approach distributes NAS tasks among participants (referred to as miners), thereby decentralizing the computational effort and potentially reducing the time and resources required for finding efficient and effective neural architectures.

---
## How it works

1. **Genetic Algorithm-Based NAS:** NASChain uses genetic algorithm for optimizing neural networks, where each network is a binary-encoded "genome". This allows for the systematic exploration of architectural possibilities.

2. **Optimization Process:** Through mutations and evaluations, NASChain refines these genomes to improve performance, aiming for the optimal blend of accuracy and efficiency across generations.

3. **Distributed Training:** Leveraging the Bittensor network, NASChain decentralizes the intensive computational process, enabling parallel genome training by a network of miners.

4. **Blockchain Integration:** This ensures security and transparency, with miners rewarded for contributing computational resources towards training and evaluating network models.

5. **Outcome:** The process yields optimal neural architectures that balance high accuracy with low computational demands, achieved more efficiently through distributed efforts.

> **The algorithm in NASChain utilizes the NSGA approach for optimization. For more insights, refer to the following resources: [paper](https://arxiv.org/abs/1810.03522) | [code](https://github.com/ianwhale/nsga-net).**
<div align="center">
<img src="imgs/naschain_graph.png" alt="Optional Image Description" width="960" height="770">
</div>

---

## Hardware Requirements

### Miners:
- **GPU**: Nvidia GPU with at least 16GB of memory. Note that 8GB graphics cards might work in some use cases, but their compatibility and performance are not guaranteed.

### Validators:
- **CPU**: Machines with only CPU are sufficient for validators as they do not undergo intensive computational loads.

---
## Installation

We recommend using virtual environments such as Conda to manage and isolate your project dependencies.

- Ensure you have Python >= 3.10 installed on your system.
- Both Miner and Validator code is only tested on Linux OS.
- It is advised to use a virtual environment to avoid conflicts with other projects or system-wide packages.

### Setting Up

1. Clone the repository:
   ```bash
   git clone [https://github.com/nimaaghli/NASChain]
2. Navigate to the project directory:
    ```bash
    cd NASChain
3. if setting up virtual enviuuement:
    - if using conda:
        ```bash
        conda create --name myenv python=3.10
        conda activate myenv
    - if suing venv
        ```bash
        python -m venv env
        source env/bin/activate
4. Install the required packages:
    ```bash
    pip install -r requirements.txt

5. Running the miner :
    ```bash
    python neurons/miner.py --netuid <TBD>  --wallet.name <wallet_name> --wallet.hotkey <wallet_name> --logging.debug --axon.port <your_sxon_port> --dht.port <your_dht_port> --dht.announce_ip <your_public_ip> --dht.announce_ip <your_public_ip>   --genomaster.ip <TBD> --genomaster.port <TBD>
    
> **Make sure your ports for DHT and Axon are accessible from outside by setting up port forwarding.**
 
5. Running the Validator :
    ```bash
    python neurons/validator.py --netuid <TBD>  --wallet.name <wallet_name> --wallet.hotkey <wallet_name> --logging.debug --axon.port <your_sxon_port> --dht.port <your_dht_port> --dht.announce_ip <your_public_ip> --dht.announce_ip <your_public_ip>  --genomaster.ip <TBD> --genomaster.port <TBD>
---
## Self-improvement mechanism
The subnet's self-improvement mechanism, orchestrated by the Genomaster, initially assigns training jobs fairly across the network's neurons based on the current subnetwork metagraph. However, the process evolves dynamically based on performance:

1. **Early Completion Reassignment:** If a neuron completes its assigned jobs more quickly than its peers, it is deemed more efficient. Consequently, it is granted additional jobs that remain unfinished, particularly those initially assigned to slower-performing miners. This ensures that active, high-performance neurons are utilized to their fullest capacity without idle time.

2. **Reassignment due to Delay:** Conversely, if a miner is significantly lagging behind the average job completion time of the network, indicating underperformance or less capable hardware, the Genomaster intervenes. The underperforming miner's pending job is reassigned to a neuron that has already completed its workload and is ready for more. This intervention is carefully balanced to ensure fairness while optimizing overall network efficiency.

> **By implementing these strategies, the competition within the subnet ensures continuous improvement in the quality and speed of computations. This adaptive mechanism aims to perpetually enhance the computational quality available in the network, ensuring that resources are not just allocated efficiently, but are also in constant refinement to leverage the fastest and most capable GPUs. This dynamic optimization helps maintain the subnet's competitive edge, ensuring it continuously evolves and improves in line with technological advancements and network demands.**

> **Currently, the Genomaster assigns only one job per miner. This will change in future releases, where miners can earn more rewards by acquiring as many jobs as their multiple GPUs can handle, resulting in better rewards and enhanced self-improvement of the subnetwork. Ultimately, miners can connect an entire GPU cluster to their neuron, creating what we refer to as a SUBSUBnetwork.**

---

## Validation and Incentive mechanism

Every miner returns the response to the Genomaster in an array of size three, such as [accuracy, parameters, FLOPs]. To ensure the results are legitimate from the miners, the Genomaster will assign each job to three different miners randomly, assuming that no miner can be in the job batch more than once (the subnet will not function if there are fewer than three miners in the network). Once responses are returned from all miners of the job batch, they will be delivered to validators upon request. To mark the three results as legitimate, they should be in agreement. Below, we describe the validation method in more detail.

### Batch Definition and Structure

**Batches**: Defined as `B = {b_1, b_2, ..., b_n}`, each batch `b_i` corresponds to evaluations from different users for the same job.

**Responses**: Each batch `b_i` contains responses structured as `[(acc_1, params_1, flops_1), (acc_2, params_2, flops_2), (acc_3, params_3, flops_3)]`:
- `acc_j`: Accuracy reported by the j-th user.
- `params_j`: Model parameters reported by the j-th user.
- `flops_j`: Floating-point operations (FLOPs) reported by the j-th user.

### Agreement Checks

- **Accuracy Agreement**: Users' responses within a batch agree on accuracy if the absolute difference between their reported accuracies is within a specific tolerance level.
- **Parameters and FLOPs Agreement**: Agreement on model parameters and FLOPs requires exact matches between users' responses.
- **Overall Agreement**: Full agreement is considered when both the accuracy (within tolerance) and the exact match on model parameters and FLOPs are satisfied between any pair of responses within the batch.

### Total Number of Jobs Finished

- The total contribution of a user is also evaluated based on the total number of jobs they have completed, fostering not only accuracy and consensus but also productivity.

### Scoring System

#### Level 1: Agreement-Based Scoring

- Users receive points for each job where their response is part of a consensus within the batch. The system evaluates and assigns points based on pairwise agreements.

#### Level 2: Productivity-Based Scoring

- Users are additionally scored based on the total number of jobs they have completed. This encourages not only quality in terms of agreement but also quantity, enhancing overall productivity.

> **The results submitted by miners are expected to show agreement unless there has been tampering with the mining code, especially regarding training parameters such as weight initialization seed, number of training epochs, or batch size. The three-batch agreement system is designed to ensure that all miners use exactly the configuration dictated by the GenoMaster, to ensure that the results returned are reliable and correct.**

> **In terms of productivity and constant improvement, the system will reward faster miners, as they will be able to finish more jobs during the training phase of each generation. This encourages not only adherence to specified configurations for consistency and accuracy but also efficiency and speed in completing tasks.**

---

## Roadmap

Our development journey is planned as follows, to enhance the functionality and reach of our platform:

1. **Testnet Launch**: Initiate debugging and error fixing with the help of miner collaborators to ensure a stable environment.

2. **Successful Search on Benchmark Datasets**:
   - Conduct successful searches on common datasets such as CIFAR-10 and CIFAR-100.
   - Share results and findings with the community to foster collaboration and improvement.

3. **Expand Dataset Range**:
   - Perform searches on larger datasets like ImageNet to test scalability and efficiency.

4. **Multi-GPU/Job Support**:
   - Enable miners to leverage multiple GPUs, allowing for parallel processing and faster computations.

5. **Disseminate Findings**:
   - Publish findings and results in conferences to contribute to the scientific community and gain feedback.

6. **Live Dashboard Website**:
   - Develop a live dashboard website to display GenoMaster stats, jobs, and interactive visualizations of the genetic algorithm at work.

7. **Frontend Website for Users**:
   - Create a frontend website where customers can register, create a search task for their use case, upload their dataset, and obtain their optimal architecture from the subnetwork.
---
