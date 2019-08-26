import snapshottest
import os
import sys

'''
def remove_snapshots_and_save(test_path, snapshot_ids):
	# !!! Be sure to preserve ordering
	snapshot_module = snapshottest.module.SnapshotModule.get_module_for_testpath(test_path)
	keys = list(snapshot_module.snapshots.keys())
	for key in keys:
		if any([key.startswith(snapshot_id) for snapshot_id in snapshot_ids]):
			del snapshot_module.snapshots[key]
	snapshot_module.save()
'''

def extract_from_line(line):
	if line.startswith('FAIL'):
		line = line.split('FAIL:')
	else:
		line = line.split('ERROR:')
	line = line[1].strip()
	not_in_brackets, other = line.split('(')
	not_in_brackets = not_in_brackets.strip()
	in_brackets = other.strip()[:-1]
	split = in_brackets.split('.')
	test_name = not_in_brackets
	class_name = split[-1]
	test_path = os.sep.join(['tests'] + split[:-1]) + '.py'
	dotted_test = '.'.join(split + [test_name])

	snapshot_key = '{}::{}'.format(class_name, test_name)

	return test_path, dotted_test, snapshot_key

for line in sys.stdin:
	if line.strip():
		try:
			test_path, dotted_test, snapshot_key = extract_from_line(line)
			print(dotted_test.strip())
		except Exception as ex:
			#import pdb; pdb.set_trace()
			import sys; import traceback; exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
			print('Failed line extraction from: {} because of {}'.format(line, ex))
			continue
		#if not os.path.exists(test_path):
		#	print('MISSING TEST PATH: {}'.format(test_path))
		#	continue
		# remove_snapshots_and_save(test_path, [snapshot_key])
		#print('Finished line: {}'.format(line))





