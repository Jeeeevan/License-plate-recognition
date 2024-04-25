import easyocr
reader=easyocr.Reader(['en'])
print(reader.readtext('just_plate.jpg'))
