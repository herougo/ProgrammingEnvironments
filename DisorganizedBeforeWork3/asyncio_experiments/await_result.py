import asyncio
import time

async def async_fn():
    print('hi start')
    time.sleep(1)
    print('hi end')
    return 0

async def main():
    result = await async_fn()
    print(result)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

''' prints
0
'''