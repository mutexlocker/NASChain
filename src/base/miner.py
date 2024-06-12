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
import os
import sys
import bittensor as bt
from requests.exceptions import ConnectionError, InvalidSchema, RequestException
from src.base.neuron import BaseNeuron
from src.utils.config import add_miner_args
sys.path.insert(0, 'nsga-net')
from utilities import utils
import requests
import datetime as dt
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.remote_model_store import RemoteModelStore
from model.dummy_trainer import DummyTrainer
from model.model_analysis import ModelAnalysis
from model.vali_config import ValidationConfig

class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

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
        # self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        # bt.logging.info(f"Attaching forward function to miner axon.")
        # self.axon.attach(
        #     forward_fn=self.forward,
        #     blacklist_fn=self.blacklist,
        #     priority_fn=self.priority,
        # )
        # bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.save_dir = 'saved_model'

    
    async def run(self):
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
        # bt.logging.info(
        #     f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        # )
        # self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # # Start  starts the miner's axon, making it active on the network.
        # self.axon.start()

        bt.logging.info(f"⛏️ Miner starting at block: {self.block}")
        # This loop maintains the miner's operations until intentionally stopped.
        try:
            vali_config = ValidationConfig()
            metadata_store = ChainModelMetadataStore(self.subtensor, self.wallet, self.config.netuid)
            remote_model_store = HuggingFaceModelStore()
            upload_dir = ""
            # namespace, name = utils.validate_hf_repo_id(self.config.hf_repo_id)
            # bt.logging.info(f"Hugface namespace and name : {namespace},{name}")
            model_id = ModelId(namespace=self.config.hf_repo_id, name='naschain')
            HuggingFaceModelStore.assert_access_token_exists()
            # Replace below code with you NAS algo to generate optmial model for you or give a path to model from args
            if self.config.model.dir is None:
                bt.logging.info("Training Model!")
                #replace dummy Trainer with NAS or manully optmize the Dummy Trainer 
                trainer = DummyTrainer(epochs=vali_config.train_epochs)
                trainer.train()
                model = trainer.get_model()    
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                save_path = os.path.join(self.save_dir, 'model.pt')
                torch.save(model, save_path)
                analysis = ModelAnalysis(model)
                params, macs, flops = analysis.get_analysis()
                bt.logging.info(f"🖥️ Params, Macs, Flops: {params} , {macs}, {flops}")
                upload_dir = save_path
                
            else:
                bt.logging.info("loading model offline!")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = torch.load(self.config.model.dir,map_location="cpu")
                analysis = ModelAnalysis(model)
                params, macs, flops = analysis.get_analysis()
                bt.logging.info(f"🖥️ Params, Macs, Flops: {params} , {macs}, {flops}")
                upload_dir = self.config.model.dir

            model_id = await remote_model_store.upload_model(Model(id=model_id, pt_model=upload_dir))
            bt.logging.success(f"Uploaded model to hugging face. {model_id} , {upload_dir}")

        
            await metadata_store.store_model_metadata(
                self.wallet.hotkey.ss58_address, model_id)

            bt.logging.info(
                "Wrote model metadata to the chain. Checking we can read it back..."
            )

            model_metadata =  await metadata_store.retrieve_model_metadata(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"⛏️ model_metadata: {model_metadata}")
            # if not model_metadata or model_metadata.id != model_id:
            #     bt.logging.error(
            #         f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
            #     )
            #     raise ValueError(
            #         f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
            #     )

            bt.logging.success("Committed model to the chain.")

            
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
                
        # If someone intentionally stops the miner, it'll safely terminate operations.
        # except KeyboardInterrupt:
        #     # self.axon.stop()
        #     bt.logging.success("Miner killed by keyboard interrupt.")
        #     exit()

        # # In case of unforeseen errors, the miner will log the error and continue operations.
        # except Exception as e:
        #     bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run_async_main)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")
    
    def run_async_main(self):
        asyncio.run(self.run())

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
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
