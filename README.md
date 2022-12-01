# Block API Utilities

This is a toolkit to make large batches of async queries to the Alchemy API to download a historical dataset of blocks and transactions. See [Alchemy's website](https://docs.alchemy.com/reference/api-overview) for more information on their API.

Making serial API calls to `eth_getBlockByNumber` costs roughly 0.15 seconds per call. Parallelizing across $N$ threads roughly cuts that time down to $\frac{0.15}{N}$ seconds per call. For instance, downloading 42.5M blocks from Arbitrum with 1 thread would take ~74 days, and with 10 threads it would take ~6 days (average 0.01 seconds per call). For machines with a large number of threads, this can greatly reduce the time-cost. 

## Disclaimer

Parallelization does not reduce the cost of using Alchemy's API in [compute credits](https://docs.alchemy.com/reference/compute-units). Please monitor your usage. Each call (per block) made by this toolkit costs 16 units.

## Setup

After cloning this repo, run:
```
pip install -e .
```

If you want to upload data to Google cloud storage, you will need to provide a path to a service account credentials file at the environment variable `GOOGLE_APPLICATION_CREDENTIALS`. 

## Usage

Here is an example code snippet to download all blocks and transactions from Arbitrum. It will save a file of [jsonlines](https://pypi.org/project/jsonlines/) every 100k blocks.

```python
from alchemy_utils import DatasetBuilder

builder = DatasetBuilder(
    api_key=...,        # Your Alchemy API key
    out_dir='./out',
    chain='arbitrum',   # Supports ethereum, arbitrum, optimism, and polygon
    start_block=1,
    save_every=100000,  # Saves a file for every 100k blocks
)
# Increase # threads for faster performance
builder.async_get(num_threads=10)

# After that completes, upload to storage buckets 
# NOTE: requires authentication through gcloud CLI
builder.upload_buckets(
    'some_unique_bucket_name',
    create_bucket=True,        # Creates a bucket 
    delete_post_upload=False,  # Delete raw file after upload
)
```

Four chains are supported (`ethereum`, `arbitrum`, `optimism`, and `polygon`), though more can be easily added. The code only supports `eth_getBlockByNumber` RPC endpoint, although this can be expanded upon request.
