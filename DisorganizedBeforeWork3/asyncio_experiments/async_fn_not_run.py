import asyncio
import time

async def async_fn():
    print('hi start')
    time.sleep(1)
    print('hi end')

def main():
    task = async_fn()
    time.sleep(10)
    print('yo man')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task)

if __name__ == '__main__':
    main()

''' prints
yo man
hi start
hi end
'''