from tensorflow.python.keras.models import load_model
import json
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import skimage.io as io
import os
import getopt, sys

opts, args = getopt.getopt(sys.argv[1:], "ho:", ["video_name"])
root=args[0]
if not os.path.exists('/content/temp'):
    os.makedirs('/content/temp')
l1=len(root)
j=-1
for i in range(l1-1,-1,-1):
  if j==-1 and root[i]!='/':
    j=i
  if j!=-1 and root[i]=='/':
    break
name=root[i+1:j+1]
name=name[:-4]
root=root[:j+1]
os.mkdir '/content/temp/videoTempFor_{name}/'
root2='/content/temp/videoTempFor_{}/'.format(name)
video = mp.VideoFileClip(root)
width,height=video.size
l=video.duration
i=0
j=2
while i+j<=l:
  v=video.subclip(i,i+j)
  k=int(i/j)
  if k<10:
    v.write_videofile(root2+"{}-00{}.mp4".format(name,k),fps=10,audio=False)
  elif k<100:
    v.write_videofile(root2+"{}-0{}.mp4".format(name,k),fps=10,audio=False)
  else:
    v.write_videofile(root2+"{}-{}.mp4".format(name,k),fps=10,audio=False)
  i+=j
names=os.listdir(root2)
names.sort()
inp=np.ndarray((len(names),20,50), dtype='float32')
for i,v in enumerate(names):
  !mkdir '/content/temp/frameFor_{v}/'
  !cd openpose &&./build/examples/openpose/openpose.bin --video $root2$v --write_json '/content/temp/frameFor_{v}/' --display 0 --render_pose 0
  root3='/content/temp/frameFor_{}/'.format(v)
  ls=os.listdir(root3)
  ls.sort()
  for j,k in enumerate(ls):
    with open(root3+k,'r') as load_f:
      load_dict = json.load(load_f)
    if load_dict['people']:
      ls=load_dict['people'][0]['pose_keypoints_2d']
      ls2=[]
      for ind,valu in enumerate(ls):
        if ind%3==0:
          ls2.append(valu/width)
        elif ind%3==1:
          ls2.append(valu/height)
      inp[i,j,:]=ls2
    else:
      inp[i,j,:]=[0]*50
model=load_model('new_model20.h5')
result=model.predict(inp)
x=[]
data=[]
for i in range(len(result)):
  x.append(2*(i+1))
  data.append([2*(i+1),str(result[i][0])])
jsontext = {'stretch body':data}
jsondata = json.dumps(jsontext)
f = open('timeLable.json'.format(name), 'w')
f.write(jsondata)
f.close()
if len(names)==1:
  plt.plot(x,result,'bo',label='prediction')
else:
  plt.plot(x,result,'b',label='prediction')
plt.ylim((0,1))
plt.xticks(x, x)
plt.xlabel('seconds')
plt.ylabel('prob of stretching body')
plt.title('prediction for every 2s')
plt.legend()
plt.savefig('timeLabel.jpg')
plt.show()

