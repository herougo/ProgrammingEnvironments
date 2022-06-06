import asyncio
import time

async def async_fn():
    print('hi start')
    time.sleep(1)
    print('hi end')
    return 1

def main():
    task = async_fn()
    print(task + 1)
    time.sleep(1)
    print('yo man')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task)

if __name__ == '__main__':
    main()

'''
(throws error)
'''