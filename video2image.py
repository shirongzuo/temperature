import cv2
vidcap = cv2.VideoCapture('TOP134_rA2_ESM4450_TempReading.MOV')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("nihao/frame%d.jpg" % count, image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])     # save frame as JPEG file

  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1