import argparse
import sys
import numpy as np
import matplotlib as mlp
import os.path as path
import os
from scipy.interpolate import interp1d

def parse_args():
	parser = argparse.ArgumentParser(
		description='Analysize the caffe log file'
	)
	parser.add_argument(
		'--log',
		dest='log_file',
		help='The log file / folder to be analysized',
		default=None,
		type=str
	)
	parser.add_argument(
		'--item',
		dest='item',
		help='The actual work for analysis,' + \
		' for default is accuracy_cls to plot accuracy item,' + \
		' you can customize $item_name for any' + \
		' item you want to plot in the json log file',
		default="accuracy_cls",
		type=str
	)
	parser.add_argument(
		'--prefix',
		dest='prefix',
		help='The prefix of output file',
		default="",
		type=str
	)
	parser.add_argument(
		'--sample_rate',
		dest='sample_rate', 
		help='sample rate for train / test loss at each iteration, only int',
		default=10,
		type=int
	)
	parser.add_argument(
		'-display',
		dest='display', 
		help='show the figures',
		action='store_true'
	)

	if len(sys.argv) in [1,2]:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()

def main():
	args = parse_args()
	if not args.display:
		mlp.use('Agg')
	getStat(args)

def getStat(args):
	log_file = "\\\\".join(args.log_file.split('\\'))
	assert path.isdir(log_file) or path.isfile(log_file), \
		log_file + " is not a file or path!"
	if path.isdir(log_file): # it's a path
		file_list = [] 
		for root,dirs,files in os.walk(log_file):
			file_list.extend([path.join(root,file) for file in files])
		print "Plot %s item in dir: %s"%(args.item,log_file)
		ind_list = []
		items_list = []
		file_list_good = []
		for file in file_list:
			print "Find log file: " + file
			with open(file,'r') as logfile:
				ind, items = _getStat(logfile,args.item)
				if len(ind) == 0:
					print "Log file: %s doesn't have target item: %s, will skip it" \
						%(file,args.item)
				else:
					ind = [int(i) for i in ind]
					ind_list.append(ind[::args.sample_rate])
					items_list.append(items[::args.sample_rate])
					file_list_good.append(file)
		assert len(file_list_good) > 0, \
				"At least one log file should have net Iteration and target item!"
		_plotWithinFile([path.basename(file) for file in file_list_good],
			    	ind_list, items_list,
			    	args.prefix + args.item,
			    	args.display)		
	else: # it's a file
		print "Plot %s item in file: %s"%(args.item,log_file)
		with open(log_file,'r') as logfile:
			ind, items = _getStat(logfile,args.item)
			assert len(ind) > 0, "Log file must have net Iteration and target item!"
			ind = [int(i) for i in ind]
			ind = ind[::args.sample_rate]
			items = items[::args.sample_rate]
			_plotWithinFile(path.basename(log_file),
						ind, items,
						args.prefix + args.item,
						args.display)

def _plotWithinFile(filelist,x,y,type,display):
	#plot stats in several files on same figure
	import pylab as plt
	print "Drawing figure: " + type
	f_list = filelist
	if (not isinstance(f_list,list)) and isinstance(f_list,str):
		f_list = [f_list]
	if (not isinstance(x[0],list)) and isinstance(x,list):
		x = [x]
	if (not isinstance(y[0],list)) and isinstance(y,list):
		y = [y]
	assert len(f_list) == len(x) and len(f_list) == len(y), \
		"Log file numbers and x/y plot numbers are not consistent!"

	for k,v in enumerate(f_list):
	 	plt.plot(x[k],y[k],linewidth=1.,label=v)
	plt.legend()
	plt.title(type+" figure")
	plt.xlabel('Iteration')
	plt.ylabel(type)
	print "Saved at " + type + ".png"
	plt.savefig(type + ".png",dpi=150)
	if display:
		plt.show()
	else:
		plt.close()

def _getStat(logfile,key):
	# get iterations in the logfile 
	# and put the Iteration and item stats to the lists
	index = []
	items = []
	for line in logfile.readlines():
		if '"%s"'%(key) in line:
			x = line.split()
			for k,v in enumerate(x):
				if '"%s"'%(key) in v:
					items.append(float(x[k+1][:-1]))
				if '"iter"' in v:
					index.append(x[k+1][:-1])

	if len(index)-len(items) > 0:
		index = index[:-(len(index)-len(items))]

	assert len(index) == len(items), \
			"Train iteration and item values are not consistent!"
	return index,items

if __name__ == '__main__':
	main()
