import json
import asyncio
import requests
import jsonlines
import numpy as np
from tqdm import tqdm
from glob import glob

from os import makedirs, remove
from os.path import join, basename

from timeit import default_timer
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage

START_TIME = default_timer()


class DatasetBuilder:
    r"""Builds a dataset of historical transaction data.
    Arguments:
    --
    rpc_provider (str): RPC provider name (e.g. quicknode, alchemy, infura).
    rpc_provider_url (Optional[str],default=None): RPC endpoint. Optional if using api_key.
    api_key (Optional[str],default=None): RPC Provider API key. Optional if using rpc_provider_url.
    out_dir (Optional[str], default='./output'): Output directory to save API responses to. 
    chain (str, default=ethereum): Which chain to pull data from?
    start_block (int, default=1): Which block number to start pulling data from?
    end_block (Optional[int], default=None): Which block number to stop pulling data from?
        If None is supplied, defaults to the latest block.
    save_every (int, default=100000): Transactions will be saved in batches of this size. 
        For example, a dataset of 1M rows will be saved through 10 files if `save_every` is 100k.
    """
    def __init__(self,
                 rpc_provider,
                 rpc_provider_url=None,
                 api_key=None,
                 out_dir='./output',
                 chain='ethereum',
                 start_block=1,
                 end_block=None,
                 save_every=100000,
                 ):

        if rpc_provider_url==None and api_key==None:
            raise Exception ('You have to provide at least one of the following parameters: rpc_provider_url or api_key.')
        
        rpc_url = get_rpc_provider(rpc_provider, rpc_provider_url, chain, api_key)
        
        if end_block is None:
            # Ping RPC to get the latest block
            last_block = get_current_block(rpc_url)
            if last_block is not int:
                raise Exception('Failed to fetch latest block number. RPC Provider response: ' + str(last_block))
            end_block = last_block

        # Create directory to save output if not existing yet
        makedirs(out_dir, exist_ok=True)

        # Save every can't be bigger than the # of blocks
        save_every = min(save_every, end_block - start_block)

        # Save to class
        self.rpc_url = rpc_url
        self.out_dir = out_dir
        self.start_block = start_block
        self.end_block = end_block
        self.save_every = save_every

    def async_get(self, num_threads=10):
        r"""Parallel API calls.
        Arguments:
        --
        num_threads (int, default=10): Number of parallel threads
        """
        chunks = np.arange(self.start_block, self.end_block + 1, self.save_every)

        for i in range(len(chunks) - 1):
            start_block_i = int(chunks[i])
            end_block_i = int(chunks[i+1])
            out_file = join(self.out_dir, 
                            f'blocks-{start_block_i}-to-{end_block_i}.jsonl',
                            )

            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(
                async_make_api_requests(url=self.rpc_url,
                                        start_block=start_block_i,
                                        end_block=end_block_i,
                                        num_threads=num_threads,
                                        out_file=out_file,
                                        )
                )
            loop.run_until_complete(future)

    def upload_buckets(self,
                       bucket_name,
                       create_bucket=False,
                       delete_post_upload=False,
                       ):
        r"""Upload cached CSVs to google cloud bucket.
        Arguments:
        --
        bucket_name (str): Name of the storage bucket to upload to.
        create_bucket (bool, default=False): If true, create the bucket.
        delete_post_upload (bool, default=False): Remove CSV after uploading if True.
        Notes:
        --
        A progress bar will be printed upon execution.
        """
        storage_client = storage.Client()

        if create_bucket:
            # If designated, create the storage bucket
            success = create_storage_bucket(bucket_name)

            if not success:
                raise Exception(f'Could not create bucket {bucket_name}. Try again?')

        # Check that the bucket exists
        assert storage_bucket_exists(bucket_name)

        # Fetch the bucket
        bucket = storage_client.bucket(bucket_name)

        # Find all files we want to upload
        data_files = glob(join(self.out_dir, '*.jsonl'))

        pbar = tqdm(total=len(data_files), desc='uploading to bucket')
        for data_file in data_files:
            blob_name = basename(data_file)
            blob = bucket.blob(blob_name)

            blob.upload_from_filename(data_file)

            # Delete CSV if requested
            if delete_post_upload:
                remove(data_file)

            pbar.update()
        pbar.close()


def get_rpc_provider(rpc_provider, rpc_provider_url, chain, api_key):
    r"""Returns the chain URL from RPC provider.
    Arguments:
    --
    chain (str): Chain to pull data from.
        Choices: ethereum | polygon | optimism | arbitrum
    api_key (str): API key
        API key
    Returns:
    --
    provider_url (str): RPC url
    """
    if rpc_provider == 'quicknode':
        provider_url = rpc_provider_url

    if rpc_provider == 'alchemy':
        if chain == 'ethereum':
            provider_url = f'https://eth-mainnet.g.alchemy.com/v2/{api_key}'
        elif chain == 'polygon':
            provider_url = f'https://polygon-mainnet.g.alchemy.com/v2/{api_key}'
        elif chain == 'optimism':
            provider_url = f'https://opt-mainnet.g.alchemy.com/v2/{api_key}'
        elif chain == 'arbitrum':
            provider_url = f'https://arb-mainnet.g.alchemy.com/v2/{api_key}'
        else:
            raise Exception(f'Chain {chain} not supported.')
    
    if rpc_provider == 'infura':
        if chain == 'ethereum':
            provider_url = f'https://mainnet.infura.io/v3/{api_key}'
        elif chain == 'polygon':
            provider_url = f'https://polygon-mainnet.infura.io/v3/{api_key}'
        elif chain == 'optimism':
            provider_url = f'https://optimism-mainnet.infura.io/v3{api_key}'
        elif chain == 'arbitrum':
            provider_url = f'https://arbitrum-mainnet.infura.io/v3/{api_key}'
        else:
            raise Exception(f'Chain {chain} not supported.')

    return provider_url

def get_current_block(url):
    r"""Get the current block.
    Arguments:
    --
    url (str)
    Returns:
    --
    block_number (int): Latest block number.
    """
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "eth_blockNumber"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        return json.loads(response.text)

    response_data = json.loads(response.text)
    block_number = int(response_data['result'], 0)

    return block_number


async def async_make_api_requests(url,
                                  start_block,
                                  end_block,
                                  out_file,
                                  num_threads=10,
                                  ):
    r"""Make async API requests.
    Arguments:
    --
    url (str): API endpoint.
    start_block (int): Block number to start pulling at.
    end_block (int): Block number to stop pulling at.
    out_file (str): Where to save API results
    num_threads (int, default=10): Number of threads to use.
    """
    print("{0:<30} {1:>20}".format("Block number", "Completed at"))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        with requests.Session() as session:
            loop = asyncio.get_event_loop()
            START_TIME = default_timer()  # update global start time
            tasks = [
                loop.run_in_executor(
                    executor,
                    make_api_request,
                    *(session, block, url)
                )
                for block in range(start_block, end_block)
            ]

            # Discard any failed responses
            dataset = []
            for block_data in await asyncio.gather(*tasks):
                if block_data is not None:
                    dataset.append(block_data)

            # Write to file
            with jsonlines.open(out_file, mode='w') as writer:
                writer.write_all(dataset)


def make_api_request(session, block_number, url):
    r"""Pings the method `eth_getBlockByNumber`.
    Arguments:
    --
    block_number (int): Number of the block.
    url (str): URL for the RPC API.
    Returns:
    --
    response_data (Dict[str, any]): Block and transaction JSON.
    Notes:
    --
    Prints block #'s and timestamps as it runs.
    """
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), True],
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    with session.post(url, json=payload, headers=headers) as response:
        if response.status_code != 200:
            return None

        response = json.loads(response.text)
        response_data = response['result']

        # Print update
        elapsed_time = default_timer() - START_TIME
        completed_at = "{:5.2f}s".format(elapsed_time)
        print("{0:<30} {1:>20}".format(block_number, completed_at))

        return response_data


def storage_bucket_exists(gcloud_bucket):
    r"""Checks if a storage bucket exists.
    Returns:
    --
    exists (bool): True if the bucket exists; False otherwise.
    """
    client = storage.Client()
    exists = storage.Bucket(client, gcloud_bucket).exists()

    return exists


def create_storage_bucket(gcloud_bucket):
    r"""Creates a new storage bucket.
    Returns:
    --
    success (bool): True if the bucket was created. False otherwise.
    Notes:
    --
    Assumes the storage bucket does not exist yet. We do not explicitly 
    check for this.
    """
    client = storage.Client()
    client.create_bucket(gcloud_bucket)
    success = storage_bucket_exists(gcloud_bucket)
    return success
