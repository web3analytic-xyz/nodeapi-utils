# Block API Utilities

Toolkit to make large batches of async queries to the Alchemy API to download a historical dataset of blocks and transactions. See [Alchemy's website](https://docs.alchemy.com/reference/api-overview) for more information on their API.

Making serial API calls to `eth_getBlockByNumber` costs roughly 0.15 seconds per call. Parallelizing across $N$ background workers roughly cuts that time down to $\frac{0.15}{N}$ seconds per call. For instance, downloading 42.5M blocks from Arbitrum with 10 threads averages 0.01 seconds per call (~6 days for total). For machines with a large number of threads, this can greatly reduce the time-cost. 

## Disclaimer

Parallelization does not reduce the cost of using Alchemy's API in [compute credits](https://docs.alchemy.com/reference/compute-units). Please monitor your usage. Each call (per block) made by this toolkit costs 16 units.

## Setup

After cloning this repo, run:
```
pip install -e .
```

## Usage

Here is an example code snippet to download all blocks and transactions from Arbitrum. It will save a file of [jsonlines](https://pypi.org/project/jsonlines/) every 100k blocks.

```python
from alchemy_utils import DatasetBuilder

builder = DatasetBuilder(
    api_key=os.environ['ALCHEMY_API_KEY'],
    out_dir='./out',
    chain='arbitrum',
    start_block=1,
    save_every=100000,
)
builder.async_get(num_threads=10)
```

Four chains are supported (`ethereum`, `arbitrum`, `optimism`, and `polygon`), though more can be easily added. The code only supports `eth_getBlockByNumber` RPC endpoint, although this can be expanded upon request.
