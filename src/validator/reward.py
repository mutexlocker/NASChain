# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Nima Aghli
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

import torch
from typing import List
import pandas as pd
from collections import defaultdict
import bittensor as bt
def reward(self, df: pd.DataFrame , tolerance : int):
    try:
        # Initialize the dictionary to store user scores and job counts
        user_scores = defaultdict(lambda: {'points': 0, 'accepted_jobs': 0, 'rejected_jobs': 0})

        # Process each job batch
        for _, group in df.groupby('Genome_String'):
            responses = group['Response'].tolist()
            user_ids = group['Assigned_User'].tolist()

            # Initialize agreement checks
            agreements = [False] * len(user_ids)  # Default all to False

            # Check pairwise agreement within tolerance for the first element and exact match for others
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    # Check agreement between i and j
                    agree_first = abs(responses[i][0] - responses[j][0]) <= tolerance
                    agree_second = responses[i][1] == responses[j][1]
                    agree_third = responses[i][2] == responses[j][2]

                    # Update agreement status
                    if agree_first and agree_second and agree_third:
                        agreements[i] = True
                        agreements[j] = True

            # Update scores based on agreement
            for i, user_id in enumerate(user_ids):
                if agreements[i]:
                    user_scores[user_id]['points'] += 1
                    user_scores[user_id]['accepted_jobs'] += 1
                else:
                    user_scores[user_id]['rejected_jobs'] += 1

        # Normalize the points and prepare the lists
        total_points = sum(user['points'] for user in user_scores.values())
        normalized_scores_list = []
        user_ids_list = []

        # Fill the lists with normalized scores and user IDs
        for user_id, score in user_scores.items():
            normalized_score = score['points'] / total_points if total_points > 0 else 0
            normalized_scores_list.append(normalized_score)
            user_ids_list.append(user_id)

        
        
        all_users_in_metagraoh = list(range(int(self.metagraph.n)))
        all_scores_tensor = torch.zeros(int(self.metagraph.n))
        # Scatter the scores to the corresponding user IDs
        all_scores_tensor[user_ids_list] = torch.tensor(normalized_scores_list)
        return all_scores_tensor, all_users_in_metagraoh, user_scores
    except Exception as e:
        # Handle or log the error as appropriate
        bt.logging.error(f"❌ An error occurred in reward function: {e}")
        # Optionally, return a meaningful error value or re-raise the exception
        return None, None, None

def get_rewards(
    self,
    query: int,
    responses_df: pd.DataFrame,
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.

    rewards, uids, msgs = reward(self, responses_df,1)
    return torch.FloatTensor(rewards).to(self.device), uids, msgs
    # return torch.FloatTensor(
    #     [reward(query, response) for response in responses]
    # ).to(self.device)
