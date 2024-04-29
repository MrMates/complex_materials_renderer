#!/usr/bin/env python

import sys
import io
import pywavefront
import json
import collections

DICTIONARY_FILE_NAME = "mat_dictionary.json"

def parse_obj_file(obj_file):
	try:
		# Load the material file
		materials = pywavefront.Wavefront(obj_file)
	except FileNotFoundError:
		print("Either " + obj_file + " does not exist or there isn't a .mtl file accompanying it.")
		sys.exit(1)

	return materials

def get_from_dictionary():
	temp = dict()

	f = open(DICTIONARY_FILE_NAME, "r")
	dictionary = json.load(f, object_pairs_hook=collections.OrderedDict)
	f.close()

	print("Available materials:")
	index = 0
	for key in dictionary:
		print(str(index) + ': ' + key)
		index += 1

	choice = input("Enter the index of the material you want to use: ")
	choice = list(dictionary.keys())[int(choice)]

	temp['sigma_s'] = dictionary[choice]['sigma_s']
	temp['sigma_a'] = dictionary[choice]['sigma_a']
	temp['g'] = dictionary[choice]['g']
	temp['ior'] = dictionary[choice]['ior']

	return temp

def get_scene_from_input():
	temp = dict()

	camera = input("Enter comma separated camera position (x, y, z): ")
	camera = [float(x) for x in camera.split(',')]

	temp['camera'] = camera

	cameraLookAt = input("Enter comma separated camera look at point (x, y, z): ")
	cameraLookAt = [float(x) for x in cameraLookAt.split(',')]

	temp['cameraLookAt'] = cameraLookAt

	fov = input("Enter camera FOV: ")
	temp['fov'] = float(fov)

	lightPos = input("Enter comma separated light position (x, y, z): ")
	lightPos = [float(x) for x in lightPos.split(',')]
	temp['lightPos'] = lightPos

	lightColor = input("Enter comma separated light color (r, g, b): ")
	lightColor = [float(x) for x in lightColor.split(',')]
	temp['lightColor'] = lightColor

	lightIntensity = input("Enter light intensity: ")
	temp['lightIntensity'] = float(lightIntensity)

	scale = input("Enter scale (1: 1 scene unit = 1 mm; 10: 1 su = 1 cm; 1000: 1 su = 1 m; etc.): ")
	temp['scale'] = float(scale)

	return temp

def get_from_input():
	temp = dict()

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

	ior = input("Enter the index of refraction: ")

	temp['ior'] = float(ior)

	return temp

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

	choice = input("Do you want to use the default scene settings (camera position, light position, etc.)? (y/n): ")
	if choice == 'y':
		temp = dict()
		temp["camera"] = [0.2, 4.2, 6.5]
		temp["cameraLookAt"] = [0.0, 4.1, 0.2]
		temp["fov"] = 36.0
		temp["lightPos"] = [-1.001, 5.0, 6.0]
		temp["lightColor"] = [0.8, 0.8, 0.6]
		temp["lightIntensity"] = 100.0
		temp["scale"] = 10.0
		mat["scene"] = temp
	else:
		mat["scene"] = get_scene_from_input()

	while True:

		# Print material names
		print("Found materials:")
		index = 0
		for material in mats.materials:
			print(str(index) + ': ' + material)
			index += 1

		choice = input("Enter the index of material to be changed into media: ")
		print("You chose: " + list(mats.materials.keys())[int(choice)])

		dictionary = input("Do you want to use a predefined material? (y/n): ")
		if dictionary == 'y':
			mat[str(choice)] = get_from_dictionary()
		else:
			mat[str(choice)] = get_from_input()

		choice = input("Do you want to change another material into media? (y/n): ")
		if choice == 'n':
			break

	f = open(mat_file, "w")
	json.dump(mat, f, indent=4)
	f.close()

		