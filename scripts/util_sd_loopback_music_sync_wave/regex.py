


arg_regex = r'([\-]?[0-9]*\.?[0-9]+)'
opt_regex = r'(?:\s*,\s*([\-]?[0-9]*\.?[0-9]+))?'


def create_regex(prefix, name, num_of_args, num_of_opts=0):
	args = [arg_regex for x in range(num_of_args)]
	opts = [opt_regex for x in range(num_of_opts)]
	return prefix + name + r'\(\s*' +\
		r'\s*,\s*'.join(args) +\
		r''.join(opts) +\
		r'\s*\)'

arg_text_regex = r'\"([^\"]+)\"'
opt_text_regex = r'(?:\s*,\s*\"([^\"]+)\")?'

def create_regex_text(prefix, name, num_of_args, num_of_opts=0):
	args = [arg_text_regex for x in range(num_of_args)]
	opts = [opt_text_regex for x in range(num_of_opts)]
	return prefix + name + r'\(\s*' +\
		r'\s*,\s*'.join(args) +\
		r''.join(opts) +\
		r'\s*\)'

