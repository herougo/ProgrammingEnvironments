import sublime, sublimeplugin
import os


self.view.file_name()

self.view.run_command("revert") # reload current file
self.view.run_command("save")   # save current file

sublime.message_dialog("hello world")



error_code = os.system("cd " + filefolder + "; git commit -am update")
if error_code != 0:
	# display messsage in a popup
	sublime.error_message("An error occurred with 'git commit -am update'")
else:
	# print message at bottom like when you save
	sublime.status_message("git update successful")



# replace current line with "haha replaced!"
region = self.view.line(self.view.sel()[0])
current_line_text = self.view.substr(region)
self.view.replace(edit, region, "haha replaced!")