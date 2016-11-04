import sublime
import sublime_plugin


class StringReplaceCommand(sublime_plugin.TextCommand):
	def run(self, edit):
		region = self.view.line(self.view.sel()[0])
		current_line_text = self.view.substr(region)
		self.view.replace(edit, region, "haha replaced!")
