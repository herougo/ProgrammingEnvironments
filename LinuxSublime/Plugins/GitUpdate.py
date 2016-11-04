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


class GitUpdateCommand(sublime_plugin.TextCommand):
	def run(self, edit):
		filefolder = getFileFolder(self.view.file_name())
		error_code = os.system("cd " + filefolder + "; git commit -am update")
		if error_code != 0:
			sublime.error_message("An error occurred with 'git commit -am update'")
		else:
			sublime.status_message("git update successful")
