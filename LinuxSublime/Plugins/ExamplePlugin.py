import sublime
import sublime_plugin


def getFileFolder(filepath):
	if "\\" in filepath: #  Windows
		return "\\".join(filepath.split('\\')[:-1]) + "\\"
	elif "/" in filepath: # Linux
		return "/".join(filepath.split('/')[:-1]) + "/"
	else:
		return ""

class ExampleCommand(sublime_plugin.TextCommand):
	def run(self, edit):
		# line = self.view.fileName() + "\n"
		filepath = self.view.file_name()
		filefolder = getFileFolder(filepath)
		line = filepath + "\n" + filefolder + "\n"
 		# line = "Hello world"
		# self.view.run_command("save")
		self.view.insert(edit, 0, line)
		

