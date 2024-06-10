# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Developer: Nima Aghli   
# Copyright © 2023 Nima Aghli

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

import bittensor as bt

from src.protocol import Dummy
from src.validator.reward import get_rewards
from src.utils.uids import get_random_uids
import requests
import pandas as pd

import pandas as pd
import requests
import bittensor as bt  # Assuming this is the correct way to import bittensor in your context
import asyncio
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.disk import utils



class ValidationConfig:
    def __init__(self, min_parameters=None, max_parameters=None, min_flops=None, max_flops=None, 
                 min_accuracy=None, max_accuracy=None, max_download_file_size=None):
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.min_flops = min_flops
        self.max_flops = max_flops
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.max_download_file_size = max_download_file_size



def append_row(df, row_data):
    """
    Appends or updates a row in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to append to.
    row_data (dict): A dictionary containing the row data.

    Returns:
    pd.DataFrame: The DataFrame with the appended or updated row.
    """
    # Check if the uid exists in the DataFrame
    existing_row_index = df.index[df['uid'] == row_data['uid']].tolist()

    if existing_row_index:
        # If commit value is different, update the existing row
        index = existing_row_index[0]
        df.loc[index] = row_data
    else:
        # If uid does not exist, append the new row
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


async def get_metadata(metadata_store, hotkey):
    """Get metadata about a model by hotkey"""
    return await metadata_store.retrieve_model_metadata(hotkey)

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """

    vali_config = ValidationConfig(
        min_parameters=1000, 
        max_parameters=1000000, 
        min_flops=1e9, 
        max_flops=1e12, 
        min_accuracy=0.8, 
        max_accuracy=0.99, 
        max_download_file_size=10*1024*1024
    )

    metadata_store = ChainModelMetadataStore(self.subtensor, self.wallet, self.config.netuid)
    hg_model_store = HuggingFaceModelStore()
    for uid in range(self.metagraph.n.item()):
        bt.logging.error(f"--------------------")
        hotkey = self.metagraph.hotkeys[uid]
        bt.logging.info(f"uid {uid} {hotkey}")
        model_metadata =  await metadata_store.retrieve_model_metadata(hotkey)
        try:
            model_with_hash = await hg_model_store.download_model(model_metadata.id, local_path='cache', model_size_limit= vali_config.max_download_file_size)
            bt.logging.info(f"hash_in_metadata: {model_metadata.id.hash}, {model_with_hash.id.hash}, {model_with_hash.pt_model},{model_with_hash.id.commit}")
            new_row = {
                'uid': uid,
                'local_model_dir': model_with_hash.pt_model,
                'commit': model_with_hash.id.commit,
                'params': None,
                'accuracy': None,
                'evaluate': False,
                'pareto': False
            }
            self.eval_frame = append_row(self.eval_frame, new_row)


            print(self.eval_frame)
        except Exception as e:
            
            bt.logging.error(f"Unexpected error: {e}")
    





    
    # Send a GET request to the server
    # response = requests.get(url)


    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # # The dendrite client queries the network.
    # responses = await self.dendrite(
    #     # Send the query to selected miner axons in the network.
    #     axons=[self.metagraph.axons[uid] for uid in miner_uids],
    #     # Construct a dummy query. This simply contains a single integer.
    #     synapse=Dummy(dummy_input=self.step),
    #     # All responses have the deserialize function called on them before returning.
    #     # You are encouraged to define your own deserialization function.
    #     deserialize=True,
    # )

    # # Log the results for monitoring purposes.
    # bt.logging.info(f"Received responses: {responses}")

    # # TODO(developer): Define how the validator scores responses.
    # # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses)

    # bt.logging.info(f"Scored responses: {rewards}")
    # # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # self.update_scores(rewards, miner_uids)
