import functools
import multiprocessing
import os
import time
from typing import Any, List, Optional, Tuple
import bittensor as bt

from model.data import ModelId, ModelMetadata


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters. Got={repo_id}"
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Hugging Face repo id must be in the format <org or user name>/<repo_name>. Got={repo_id}"
        )

    return parts[0], parts[1]


def get_hf_url(model_metadata: ModelMetadata) -> str:
    """Returns the URL to the Hugging Face repo for the provided model metadata."""
    return f"https://huggingface.co/{model_metadata.id.namespace}/{model_metadata.id.name}/tree/{model_metadata.id.commit}"


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)


def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


def get_version(filepath: str) -> Optional[int]:
    """Loads a version from the provided filepath or None if the file does not exist.

    Args:
        filepath (str): Path to the version file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            line = f.readline()
            if line:
                return int(line)
            return None
    return None


def save_version(filepath: str, version: int):
    """Saves a version to the provided filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(str(version))


def move_file_if_exists(src: str, dst: str) -> bool:
    """Moves a file from src to dst if it exists.

    Returns:
        bool: True if the file was moved, False otherwise.
    """
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        return True
    return False


def list_top_miners(metagraph: bt.metagraph) -> List[int]:
    """Returns the list of top miners, chosen based on weights set on the largest valis.

    Args:
        metagraph (bt.metagraph): Metagraph to use. Must not be lite.
    """

    top_miners = set()

    # Find the top 10 valis by stake.
    valis_by_stake = get_top_valis(metagraph, 10)

    # For each, find the miner that has more than 50% of the weights.
    for uid in valis_by_stake:
        # Weights is a list of (uid, weight) pairs
        weights: List[Tuple[int, float]] = metagraph.neurons[uid].weights
        total_weight = sum(weight for _, weight in weights)

        # Only look for miners with at least half the weight from this vali
        threshold = total_weight / 2.0
        for uid, weight in weights:
            if weight > threshold:
                top_miners.add(uid)
                # Break now because only 1 miner can have more than half the weight.
                break

    return list(top_miners)


def get_top_valis(metagraph: bt.metagraph, n: int) -> List[int]:
    """Returns the N top validators, ordered by stake descending.

    Returns:
      List[int]: Ordered list of UIDs of the top N validators, or all validators if N is greater than the number of validators.
    """
    valis = []
    for uid, stake in enumerate(metagraph.S):
        # Use vPermit to check for validators rather than vTrust because we'd rather
        # cast a wide net in the case that vTrust is 0 due to an unhealthy state of the
        # subnet.
        if metagraph.validator_permit[uid]:
            valis.append((stake, uid))

    return [uid for _, uid in sorted(valis, reverse=True)[:n]]


def run_with_retry(func, max_retries=3, delay_seconds=1, single_try_timeout=30):
    """
    Retry a function with constant backoff.

    Parameters:
    - func: The function to be retried.
    - max_retries: Maximum number of retry attempts (default is 3).
    - delay_seconds: Initial delay between retries in seconds (default is 1).

    Returns:
    - The result of the successful function execution.
    - Raises the exception from the last attempt if all attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                # If it's the last attempt, raise the exception
                raise e
            # Wait before the next retry.
            time.sleep(delay_seconds)
    raise Exception("Unexpected state: Ran with retry but didn't hit a terminal state")
