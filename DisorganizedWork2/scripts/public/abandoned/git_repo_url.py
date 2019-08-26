import os

raise NotImplementedError()

def cmd_output(cmd):
    import subprocess
    result = subprocess.run(list(cmd.split()), stdout=subprocess.PIPE)
    # result.stdout is a byte string (ie b'abced')
    return result.stdout.decode("utf-8")

url = cmd_output('git config --get remote.origin.url')

to_remove = ['git@', 'https://', 'http://']
for text in to_remove:
	url = url.replace(text, '')

url = url.replace(':', '/')
print(url)