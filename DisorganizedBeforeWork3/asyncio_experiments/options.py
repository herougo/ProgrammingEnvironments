import time
import asyncio

# https://stackoverflow.com/questions/36342899/asyncio-ensure-future-vs-baseeventloop-create-task-vs-simple-coroutine

async def doit(i):
    print("Start %d" % i)
    await asyncio.sleep(3)
    print("End %d" % i)
    return i

if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop()
    #futures = [asyncio.ensure_future(doit(i), loop=loop) for i in range(10)]
    #futures = [loop.create_task(doit(i)) for i in range(10)]
    futures = [doit(i) for i in range(10)]
    result = loop.run_until_complete(asyncio.gather(*futures))
    print(result)
    print('end time', time.time() - start)
