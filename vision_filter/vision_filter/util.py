import trio
import outcome


async def run_all(*async_fns):
    results = [None] * len(async_fns)

    async def run_one(i, async_fn):
        results[i] = await outcome.acapture(async_fn)

    async with trio.open_nursery() as nursery:
        for i, async_fn in enumerate(async_fns):
            nursery.start_soon(run_one, i, async_fn)

    return results
