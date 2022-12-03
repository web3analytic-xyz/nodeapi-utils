# Node API Utilities

This is a toolkit to make large batches of async queries to a RPC Provider API to download a historical dataset of blocks and transactions. See below for more information:

- [QuickNode](https://www.quicknode.com/core-api)
- [Alchemy](https://docs.alchemy.com/reference/api-overview)
- [Infura](https://www.infura.io/product/overview)
- [Chainstack](https://chainstack.com/solution)

Making serial API calls to `eth_getBlockByNumber` costs roughly 0.15 seconds per call. Parallelizing across $N$ threads roughly cuts that time down to $\frac{0.15}{N}$ seconds per call. For instance, downloading 42.5M blocks from Arbitrum with 1 thread would take ~74 days, and with 10 threads it would take ~6 days (average 0.01 seconds per call). For machines with a large number of threads, this can greatly reduce the time-cost.

## Disclaimer

Parallelization does not reduce the cost of using RPC Provider API. Please monitor your usage and keep track of method call costs:

- [QuickNode's Credits](https://www.quicknode.com/api-credits)
- [Alchemy's Compute Units](https://docs.alchemy.com/reference/compute-units)
- [Infura's Requests](https://www.infura.io/pricing)
- [Chainstack's Requests](https://chainstack.com/pricing)

## Setup

After cloning this repo, run:

```
pip install -e .
```

If you want to upload data to Google cloud storage, you will need to provide a path to a service account credentials file at the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

## Usage

Here is an example code snippet to download all blocks and transactions from Arbitrum. It will save a file of [jsonlines](https://pypi.org/project/jsonlines/) every 100k blocks.

```python
from nodeapi_utils import DatasetBuilder

builder = DatasetBuilder(
    rpc_provider=...,     # RPC provider name (e.g. quicknode, alchemy, infura, chainstack)
    rpc_provider_url=..., # Your RPC provider url (Optional if using api_key)
    api_key=None,         # Your API key (Optional if using rpc_provider_url. Required for quicknode & chainstack)
    out_dir='./output',   # Optional: Output directory to save API responses to
    chain='arbitrum',     # Supports ethereum, arbitrum, optimism, polygon, etc (Optional if using rpc_provider_url)
    start_block=16092775, # Block to begin pulling data from
    save_every=100000,    # Saves a file for every 100k blocks
)
# Increase number of threads for faster performance
builder.async_get(num_threads=10)

# After that completes, upload to storage buckets
# NOTE: requires authentication through gcloud CLI
builder.upload_buckets(
    'some_unique_bucket_name',
    create_bucket=True,        # Creates a bucket
    delete_post_upload=False,  # Delete raw file after upload
)
```

The code only supports `eth_getBlockNumber` RPC method, and it's supported for the following chains (depending on the provider selected): `ethereum`, `arbitrum`, `arbitrum-nova`, `optimism`, `polygon`, `avalanche`, `celo`, `fantom`, `binance-smart-chain`, `gnosis`. Reach out for inquiries on additional methods to include.
