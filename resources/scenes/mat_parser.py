#!/usr/bin/env python

import sys
import io
import pywavefront
import json

def parse_obj_file(obj_file):
	try:
		# Load the material file
		materials = pywavefront.Wavefront(obj_file)
	except FileNotFoundError:
		print("Either " + obj_file + " does not exist or there isn't a .mtl file accompanying it.")
		sys.exit(1)

	# Print material names
	print("Found materials:")
	index = 0
	for material in materials.materials:
		print(str(index) + ': ' + material)
		index += 1

	return materials

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python mat_parser.py <mat_file>")
		sys.exit(1)

	# Disable logging
	pywavefront.log_handler.setStream(io.StringIO())

	obj_file = sys.argv[1]
	mats = parse_obj_file(obj_file)

	mat_file = sys.argv[1].split('.')[0]
	if mat_file == "":
		# If the input file is in "./..." format (windows TAB completion)
		mat_file = sys.argv[1].split('.')[1][1:]
	mat_file += ".json"

	mat = dict()

	while True:
		temp = dict()
		choice = input("Enter the index of material to be changed into media: ")
		print("You chose: " + list(mats.materials.keys())[int(choice)])

		sigma_s = input("Enter comma separated RGB scattering coefficients (r, g, b): ")
		sigma_s = [float(x) for x in sigma_s.split(',')]

		temp['sigma_s'] = sigma_s

		sigma_a = input("Enter comma separated RGB absorption coefficients (r, g, b): ")
		sigma_a = [float(x) for x in sigma_a.split(',')]

		temp['sigma_a'] = sigma_a

		g = input("If known, enter comma separated RGB anisotropy values (r, g, b), else leave the input blank: ")
		if g:
			g = [float(x) for x in g.split(',')]
		else:
			g = [0.0, 0.0, 0.0]

		temp['g'] = g

		mat[str(choice)] = temp

		choice = input("Do you want to change another material into media? (y/n): ")
		if choice == 'n':
			break

	f = open(mat_file, "w")
	json.dump(mat, f, indent=4)
	f.close()

		