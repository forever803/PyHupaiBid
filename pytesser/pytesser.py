"""OCR in Python using the Tesseract engine from Google
http://code.google.com/p/pytesser/
by Michael J.T. O'Kelly
V 0.0.1, 3/10/07"""

from PIL import Image, ImageEnhance
import subprocess
import cv2, imutils

if __name__=='__main__':
	import util
	import errors
	tesseract_exe_name = 'tesseract'
else:
	from . import util
	from . import errors
	tesseract_exe_name = './pytesser/tesseract.exe' # Name of executable to be called at command line

scratch_image_name = "temp.bmp" # This file must be .bmp or other Tesseract-compatible format
scratch_text_name_root = "temp" # Leave out the .txt extension
cleanup_scratch_flag = True  # Temporary files cleaned up after OCR operation

def call_tesseract(input_filename, output_filename):
	"""Calls external tesseract.exe on input file (restrictions on types),
	outputting output_filename+'txt'"""
	args = [tesseract_exe_name, input_filename, output_filename]
	proc = subprocess.Popen(args)
	retcode = proc.wait()
	if retcode!=0:
		errors.check_for_errors()

def image_to_string(im, cleanup = cleanup_scratch_flag, temp_scratch_name="temp"):
	"""Converts im to file, applies tesseract, and fetches resulting text.
	If cleanup=True, delete scratch files after operation."""
	text = None
	temp_scratch_image_name = temp_scratch_name + ".bmp"
	temp_scratch_text_name_root = temp_scratch_name
	try:
		util.image_to_scratch(im, temp_scratch_image_name)
		call_tesseract(temp_scratch_image_name, temp_scratch_text_name_root)
		text = util.retrieve_text(temp_scratch_text_name_root)
	finally:
		if cleanup:
			util.perform_cleanup(temp_scratch_image_name, temp_scratch_text_name_root)
	return text

def image_file_to_string(filename, cleanup = cleanup_scratch_flag, graceful_errors=True):
	"""Applies tesseract to filename; or, if image is incompatible and graceful_errors=True,
	converts to compatible format and then applies tesseract.  Fetches resulting text.
	If cleanup=True, delete scratch files after operation."""
	try:
		try:
			call_tesseract(filename, scratch_text_name_root)
			text = util.retrieve_text(scratch_text_name_root)
		except errors.Tesser_General_Exception:
			if graceful_errors:
				im = Image.open(filename)
				text = image_to_string(im, cleanup)
			else:
				raise
	finally:
		if cleanup:
			util.perform_cleanup(scratch_image_name, scratch_text_name_root)
	return text
	

def test():
	# 读取输入图片
	image = cv2.imread("example.jpg")
		
	# 将输入图片裁剪到固定大小
	image = imutils.resize(image, height=200)
	# 将输入转换为灰度图片
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imwrite('edge.png', gray)
	# rawImage = Image.open("edge.png")
	rawImage = Image.fromarray(gray)

	print(image_to_string(rawImage))

if __name__=='__main__':
	im = Image.open('phototest.tif')
	text = image_to_string(im)
	print(text)
	image = Image.open("example.jpg")
	print(image_to_string(image))
	enhancer = ImageEnhance.Contrast(image)
	image_enhancer = enhancer.enhance(4)
	print(image_to_string(image_enhancer))
	test()
	#
	
	# try:
	# 	text = image_file_to_string('fnord.tif', graceful_errors=False)
	# except errors.Tesser_General_Exception as value:
	# 	print("fnord.tif is incompatible filetype.  Try graceful_errors=True")
	# 	print(value)
	# text = image_file_to_string('fnord.tif', graceful_errors=True)
	# print("fnord.tif contents:{}".format(text))
	# text = image_file_to_string('fonts_test.png', graceful_errors=True)
	# print(text)


