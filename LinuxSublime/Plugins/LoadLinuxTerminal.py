import sublime
import sublime_plugin
import os

def getFileFolder(filepath):
	if "\\" in filepath: #  Windows
		return "\\".join(filepath.split('\\')[:-1]) + "\\"
	elif "/" in filepath: # Linux
		return "/".join(filepath.split('/')[:-1]) + "/"
	else:
		return ""

class LoadTerminalCommand(sublime_plugin.TextCommand):
	def run(self, edit):
		filefolder = getFileFolder(self.view.file_name())
		error_code = os.system("gnome-terminal --tab-with-profile=Dev --working-directory=" + filefolder)
		if error_code != 0:
			sublime.message_dialog("An error occurred with loading the terminal")
