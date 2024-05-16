# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import torch
import asyncio
import threading
import argparse
import traceback
import sys
import bittensor as bt
from requests.exceptions import ConnectionError, InvalidSchema, RequestException
from src.base.neuron import BaseNeuron
from src.utils.config import add_miner_args
from src.utils.config import add_genomaster_args
sys.path.insert(0, 'nsga-net/')
from search import train_search
import pynvml
import requests


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"
    version = "0.0.1"
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)
        add_genomaster_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.gpu_monitoring_thread: threading.Thread = None
        self.cudas = [0, 1]
        self.lock = asyncio.Lock()
        try:
            # Initialize NVML
            pynvml.nvmlInit()
            self.nvmhandle = pynvml.nvmlDeviceGetHandleByIndex(0) 
        except Exception as e:
            bt.logging.error(f'❌ Error Init nvm {e}')
        self.average_power = 0
    
    def request_job(self):
        bt.logging.info(f"⏩ Requsting Job from Genomaster: {self.config.genomaster.ip}:{self.config.genomaster.port}")
        try:
            # Make a POST request to the server to request a job
            response = requests.post(f'{self.config.genomaster.ip}:{self.config.genomaster.port}/request_job', json={'user_name': self.uid})
            if response.status_code == 200:
                return response.json()  # Return the job details
            else:
                return None  # No job available or user already has a job
        except ConnectionError as e:
            bt.logging.error(f'❌ Failed to connect to Genomaster server: {e}')
        except InvalidSchema as e:
            bt.logging.error(f'❌ Invalid URL schema: {e}')
        except RequestException as e:
            bt.logging.error(f'❌ Error during request to Genomaster server: {e}')

    def finish_job(self, user_name, genome_string, genome_results, attempts):
        # Make a POST request to the server to mark a job as finished
        for i in range(attempts):
            try:
                response = requests.post(f'{self.config.genomaster.ip}:{self.config.genomaster.port}/finish_job', json={
                    'user_name': user_name,
                    'genome_string': genome_string,
                    'response_values': genome_results
                })
                if response.status_code == 200:
                    bt.logging.info(f"✅ Job {genome_string} results submitted by {user_name}. Responses: {genome_results}")
                    return
                else:
                    error_message = response.json().get('message', 'No message provided')
                    bt.logging.error(f"❌ Job results submmited to sarver was not accepted. status code {response.status_code}, Reason = {error_message}")
            except Exception as e:
                    bt.logging.error(f'❌ Failed to connect to Genomaster server: {e}, retrying in 15 seconds..')
                    time.sleep(2)


    def train_genome(self,config, cuda_id):
        performance = train_search.main(genome=config['Genome'],
                                                search_space = config['config']['search_space'],
                                                init_channels = config['config']['init_channels'],
                                                layers=config['config']['layers'], cutout=config['config']['cutout'],
                                                epochs=config['config']['epochs'],
                                                save='arch_{}'.format(1),
                                                expr_root='', gpu=cuda_id)
        return list(performance.values())
    
    def run_job(self, job_info, cuda_id):
        bt.logging.info(f"⏪ Job received for {self.uid}: {job_info}")
        train_res = self.train_genome(job_info, cuda_id)
        train_res.append(self.average_power)
        train_res.append(pynvml.nvmlDeviceGetName(self.nvmhandle))
        train_res.append(pynvml.nvmlDeviceGetMemoryInfo(self.nvmhandle).total)
        train_res.append(self.version)
        self.finish_job(self.uid, job_info['Genome_String'], train_res, 5)
        time.sleep(1)
        self.cudas.append(cuda_id)
    
    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"⛏️ Miner starting at block: {self.block}")
        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while(True):
                    bt.logging.info(f"🔌 Power Usage {self.average_power}")
                    job_info= self.request_job()
                    if job_info and len(self.cudas):
                        # bt.logging.info(f"⏪ Job received for {self.uid}: {job_info}")
                        # train_res = self.train_genome(job_info)
                        # train_res.append(self.average_power)
                        # train_res.append(pynvml.nvmlDeviceGetName(self.nvmhandle))
                        # train_res.append(pynvml.nvmlDeviceGetMemoryInfo(self.nvmhandle).total)
                        # train_res.append(self.version)
                        # self.finish_job(self.uid, job_info['Genome_String'], train_res, 5)
                        cuda_id = self.cudas.pop()

                        job_runner = threading.Thread(target=self.run_job, args={job_info, cuda_id}, daemon=True)
                        job_runner.start()


                        # print(f"Sleeping for {wait_time} seconds before finishing the job...")
                        time.sleep(0)  # Sleep for the specified wait time before finishing the job
                    else:
                        bt.logging.info(f"ℹ️ No jobs are available! Error occurred or all jobs are finished, or the miner joined in the middle of the training epoch.")

                    time.sleep(0)

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def monitor_gpu_power(self):
        """
        Monitors GPU power consumption continuously until `should_exit` is set to True.
        """
        alpha = 0.001  # Smoothing factor
        first_measurement = True
        try:
            while True: 
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.nvmhandle) / 1000.0  # Convert milliwatts to watts
                if first_measurement:
                    self.average_power = power_usage  # Start with the first measurement
                    first_measurement = False
                else:
                    self.average_power = alpha * power_usage + (1 - alpha) * self.average_power  # Update the EMA
                time.sleep(1) 
        except Exception as e:
            bt.logging.error(f"An error occurred: {e}")
        finally:
            pynvml.nvmlShutdown()  # Ensure NVML shutdown if an error occurs or loop is manually stopped


    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            # Thread for GPU power monitoring
            self.gpu_monitoring_thread = threading.Thread(target=self.monitor_gpu_power, daemon=True)
            self.gpu_monitoring_thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.gpu_monitoring_thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
