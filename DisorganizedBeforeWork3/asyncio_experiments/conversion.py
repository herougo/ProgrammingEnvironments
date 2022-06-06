import asyncio

async def sync_to_async(fn, *args, **kwargs):
    return fn(*args, **kwargs)

'''
def async_to_sync(coroutine_object):
    # throws error saying you can't await inside a non-async function
    return await coroutine_object
'''

def main():
    print('hi there')

if __name__ == '__main__':
    task = sync_to_async(main)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(task)