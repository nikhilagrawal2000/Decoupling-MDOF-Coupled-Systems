
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import PySimpleGUI as sg      

layout = [[sg.Text('Input Number of Masses'),sg.InputText(size = (4, 1))],
          [sg.Text('All values of K and M are same or different')],
                 [sg.Radio('Same', "RADIO1", default=True, size=(10,1)), sg.Radio('Different', "RADIO1")],
          [sg.Text('Initial Position and Velocity of Masses are same or different')],
          [sg.Radio('Same', "RADIO2", default=True, size=(10,1)), sg.Radio('Different', "RADIO2")],
                 [sg.Submit()]]      

window = sg.Window('Window Title', layout)    

event, values = window.Read()  
print(values)
window.Close()

#print(values)
n = int(values[0])
if values[1] == True :
    layout2 = [[sg.Text('Input K'),sg.InputText(size = (4, 1)) , sg.Text('Input M'),sg.InputText(size = (4, 1))],[sg.Submit()]]
else :
    layout2  = []
    for i in range(n):
        layout2.append([sg.Text('Input K{}'.format(i+1)),sg.InputText(size = (4, 1)) , sg.Text('Input M{}'.format(i+1)),sg.InputText(size = (4, 1))])
    layout2.append([sg.Text('Input K{}'.format(int(n)+1)),sg.InputText(size = (4, 1))])
    layout2.append([sg.Submit()])
    
window = sg.Window('Window Title', layout2)
event, values2 = window.Read()
print(values2)
window.Close()

if values[3] == True :
    layout3 = [[sg.Text('Input X'),sg.InputText(size = (4, 1)) , sg.Text('Input V'),sg.InputText(size = (4, 1))],[sg.Submit()]]
else :
    layout3  = []
    for i in range(n):
        layout3.append([sg.Text('Input X{}'.format(i+1)),sg.InputText(size = (4, 1)) , sg.Text('Input V{}'.format(i+1)),sg.InputText(size = (4, 1))])
    layout3.append([sg.Submit()])
                
window = sg.Window('Window Title', layout3)
event, values3 = window.Read()
print(values3)
window.Close()


# In[4]:


# n = 2 #@param {type: "number", min: 0}
k = np.zeros([n, n])
m = np.zeros([n, n])
x = np.zeros([n, 1])


# In[5]:


if values[1] == True:
    data_val = 'Same' #@param ["Same", "Different"]{type: "string", allow-input: false}
elif values[1] == False:
    data_val = 'Different'

if data_val == 'Same':
#   k_val = 1 #@param {type: "number", min: 0}
#   m_val = 1 #@param {type: "number", min: 0}
  k_val = int(values2[0])
  m_val = int(values2[1])  
  for i in range(n):
    m[i][i] = m_val
    k[i][i] = 2*k_val
  for i in range(n-1):
    k[i+1][i] = -1*k_val
    k[i][i+1] = -1*k_val

elif data_val == 'Different':
    k_val = np.zeros([n+1, 1])
    m_val = np.zeros([n, 1])
    # for i in range(n):
    # k_val[i] = 0 #@param {type: "number", min: 0}
    # m_val[i] = 0 #@param {type: "number", min: 0}
    for i in range(n):
        k_val[i] = int(values2[2*i])
    #     k_val[i] = input("k_val[{}]".format(i+1))
    for i in range(n):
        m_val[i] = int(values2[2*i + 1])
    #     m_val[i] = input("m_val[{}]".format(i+1)) 
    for j in range(n):
        m[j][j] = m_val[j]
    for i in range(n):
        k[i][i] = k_val[i] + k_val[i+1]
    for i in range(n-1):
        k[i+1][i] = -1*k_val[i+1]
        k[i][i+1] = -1*k_val[i+1]


# In[6]:


print(k)
print(m)


# In[7]:


m_inv = np.linalg.inv(m)


# In[8]:


k_mult_m_inv = np.dot(m_inv, k)


# In[9]:


from scipy import linalg as LA

e_vals, e_vecs = LA.eig(k_mult_m_inv)


# In[10]:


print(e_vals)
print(e_vecs)


# In[11]:


w = np.sqrt(e_vals)


# In[12]:


print(w)
print(w.shape)


# In[13]:


e_vecs = np.divide(e_vecs, np.amax(e_vecs))
print(e_vecs)


# In[14]:


from scipy.optimize import fsolve, root
from mpmath import findroot
import random


# In[15]:


x_val = np.zeros([n, 1])
x_acc = np.zeros([n, 1])

if values[3] == True:
    data_val_x = 'Same' #@param ["Same", "Different"]{type: "string", allow-input: false}
else :
    data_val_x = 'Different'

if data_val_x == 'Same':
  for i in range(n):
    x_val[i] = int(values3[0])
    x_acc[i] = int(values3[1])
#     x_val[i] = random.uniform(0, 10)
#     x_acc[i] = random.uniform(0, 10)
elif data_val_x == 'Different':
  for i in range(0, n):
    x_val[i] = int(values3[2*i])
    x_acc[i] = int(values3[2*i + 1])
#     x_val[i] = input("x_val[{}]".format(i+1))
#     x_acc[i] = input("x_acc[{}]".format(i+1))

def func(z):
  z = np.array(z).reshape((2*n, 1))
  # z.reshape((2*n, 1))
  c = np.array(z[0:n]).reshape((n, 1)) 
  phi = np.array(z[n:2*n]).reshape((n, 1))
  c_phi = np.multiply(c, np.cos(phi))
  f = np.zeros([2*n, 1])
  f[0:n] = np.array(x_val[0:n]).reshape((n, 1))
  arr = np.array(x_acc[0:n]).reshape((n, 1))
  for i in range(n, 2*n):
    f[i] = arr[i - n]
# changed transpose
  c_phi_eig = np.dot(np.transpose(c_phi), np.transpose(e_vecs))
#   print(c_phi_eig)
  f[0:n] = np.transpose(c_phi_eig)
  c_phi_sin = np.multiply(c, np.sin(phi))
#   print(c_phi_sin.shape)
  c_phi_sin_w = np.multiply(w.reshape((n, 1)), c_phi_sin)
#   print(c_phi_sin_w.shape)
  c_phi_eig_sin = np.dot(np.transpose(c_phi_sin_w), np.transpose(e_vecs))

  arr_1 = np.transpose(c_phi_eig_sin)
#   print("arr_1 : ", arr_1.shape)
  for i in range(n, 2*n):
    f[i] = arr_1[i - n]
  
  return f.ravel()



z = fsolve(func, np.ones([2*n, 1]).ravel())
# z = findroot(func, np.ones([2*n, 1]).ravel())
# z = root(func, np.zeros([2*n, 1]).ravel(), method = 'lm')


# In[16]:


print(z)


# In[17]:


print(func(z))


# In[18]:


t_size = 1000
t = np.linspace(0, 10, t_size)
print(t.shape)
e_vec_t = np.transpose(e_vecs)
x_final = np.zeros((n, t_size, n))
x_n = np.zeros((n ,t_size, n))
for l in range(n):
  e_vecs_col = e_vec_t[:, l]
  x_n_eigen = np.zeros((n, t.shape[0]))
  x_n_temp = np.zeros((n, t_size))
  for i in range(n):
    x_n_temp[i, :] = np.multiply(z[i], np.cos(np.add(np.multiply(w[i],t), z[i+n])))
    x_n_eigen[i] = np.multiply(e_vecs_col[i], x_n_temp[i])
  x_final[:, :, l] = x_n_eigen
  x_n[:, :, l] = x_n_temp


# In[19]:


print(x_n_eigen)
print(x_n_eigen.shape)


# In[20]:


print(x_final.shape)
print(x_final)


# In[21]:


x_n_sum = np.zeros((n, t_size))
for i in range(n):
  x_n_sum[i, :] = x_final[:, :, i].sum(axis = 0)
print(x_n_sum.shape)
print(x_n_sum)


# In[22]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[23]:


# fig, ax = plt.subplots(n, figsize = (20, 20))
# fig.suptitle('Motion of all masses')
# for num in range(n):
#   x_n_temp = np.zeros((n, t_size))
#   x_n_temp = x_n[:, :, num]
#   ax[num].set_xlim([0, 10])
#   ax[num].set_ylim([np.amin(x_n_temp), np.amax(x_n_temp)])
#   ax[num].plot(x_n_sum[num], label = 'x_final')
#   ax[num].set_title('Motion of Mass {}'.format(num+1))
#   # for i in range(n):
#   #   ax[num].plot(x_n_temp[i], label = 'x')
# # plt.legend()
# plt.show()


# In[24]:


plt.figure(figsize=(20,20))
# plt.plot(x_n_sum, label = 'x_final')
for i in range(n):
  plt.plot(x_n[i], label = 'x_eigen{}'.format(i+1))
  plt.plot(x_n_sum[i], label = 'x{}'.format(i+1))
plt.legend()
plt.show()


# In[30]:


fig, ax = plt.subplots()
ax.set_xlim([0, 10])
ax.set_ylim([np.amin(x_n), np.amax(x_n)])
sinegraph, = ax.plot([], [])
dot, = ax.plot([],[], 'o', color='red', animated = True)

def sine(i):
    sinegraph.set_data(t[:i], x_n_sum[:i])
    dot.set_data(t[i], x_n_sum[i])
    return sinegraph
anim = animation.FuncAnimation(fig, sine, frames=1000, interval = 40, repeat = True)
plt.show()

mywriter = animation.FFMpegFileWriter(fps=25,codec="libx264")
anim.save("test.mp4", writer=mywriter)


# In[28]:


import matplotlib
# matplotlib.use('Qt5Agg') #use Qt5 as backend, comment this line for default backend

fig, ax = plt.subplots()

ax.set_xlim([0, 10])
ax.set_ylim([np.amin(x_n), np.amax(x_n)])
lines = [plt.plot([], [], animated = True)[0] for _ in range(n)] #lines to animate
dots = [ax.plot([],[], 'o', color='red', animated = True)[0] for _ in range(n)]

patches = lines + dots
def init():
    #init lines
    for line in lines:
      line.set_data([], [])

    for dot in dots:
      dot.set_data([], [])

    return patches #return everything that must be updated

def animate(i):
    #animate lines
    for j,line in enumerate(lines):
        line.set_data(t[:i], x_n[j][:i])

    for k,dot in enumerate(dots):
        dot.set_data(t[i], x_n[k][i])

    return patches #return everything that must be updated

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, blit=True)

plt.show()

mywriter = animation.FFMpegFileWriter(fps=25,codec="libx264")
anim.save("test_1.mp4", writer=mywriter)


# In[ ]:


# from google.colab.patches import cv2_imshow
# import cv2
# cap = cv2.VideoCapture('test.mp4') 
   
# # Check if camera opened successfully 
# if (cap.isOpened()== False):  
#   print("Error opening video  file") 
   
# # Read until video is completed 
# while(cap.isOpened()): 
      
#   # Capture frame-by-frame 
#   ret, frame = cap.read() 
#   if ret == True: 
   
#     # Display the resulting frame 
#     cv2_imshow(frame) 
   
#     # Press Q on keyboard to  exit 
#     if cv2.waitKey(25) & 0xFF == ord('q'): 
#       break
   
#   # Break the loop 
#   else:  
#     break
   
# # When everything done, release  
# # the video capture object 
# cap.release() 
   
# # Closes all the frames 
# cv2.destroyAllWindows() 


# In[ ]:


from IPython.display import HTML
from base64 import b64encode
mp4 = open('test_1.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# In[ ]:



mp4 = open('test.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# In[33]:


from vpython import *
scene = canvas()
from PIL import Image


# In[34]:


springs = []
balls = []
ini_pos = []
for i in range(n):
    if i==0:
        x=2
    else:
        x=balls[i-1].pos.y
        
    spring = helix( pos=vector(0,x,0), axis=vector(0,-1,0), size=vector(1.8,0.3,0.3),
                    stiffness=10.0 )
    
    ball = sphere( pos=spring.pos+spring.axis, color=color.red, radius=0.2,
                   mass=1.0, velocity=vector(0,0.1,0) )
        
    springs.append(spring)
    balls.append(ball)
    ini_pos.append(ball.pos.y)
 
x=balls[n-1].pos.y 
spring = helix( pos=vector(0,x-1.8,0), axis=vector(0,1,0), size=vector(1.8,0.3,0.3),
                    stiffness=10.0 )
springs.append(spring)    

floor1 = box(pos=vector(0,2,0), size = vector(1,0.05,1), color=color.green)
floor2 = box(pos=vector(0,springs[n].pos.y,0), size = vector(1,0.05,1), color=color.green)

t = 0
while (t < 1000):
    rate(30)
    for i in range(n):
        if i>0 :
            springs[i].pos=balls[i-1].pos
        #balls[i].velocity = vector(0,-1,0) #balls[i].velocity + balls[i].force/balls[i].mass*dt
        balls[i].pos.y = ini_pos[i] + x_n_sum[i, t]/np.amin(x_n_sum)
#         springs[i].axis = balls[i].pos-springs[i].pos
        springs[i].size.x = springs[i].pos.y-balls[i].pos.y
        
#         springs[i+1].pos=balls[i].pos
        #balls[i].force = -springs[i].stiffness*(balls[i].pos-springs[i].pos-springs[i].axis)
    t = t + 1
    springs[n].size.x=balls[n-1].pos.y-springs[n].pos.y
    
    print('time - {}'.format(t))
    
# while (True):
#     rate(10)
#     for i in range(2):
#         if i>0 :
#             springs[i].pos=balls[i-1].pos
#         balls[i].velocity = vector(0,-i-1,0) #balls[i].velocity + balls[i].force/balls[i].mass*dt
#         balls[i].pos = balls[i].pos + balls[i].velocity*dt
#         springs[i].size.x = springs[i].pos.y-balls[i].pos.y
#         #springs[i+1].pos=balls[i].pos
#         #balls[i].force = -springs[i].stiffness*(balls[i].pos-springs[i].pos-springs[i].axis)
#     springs[2].size.x=balls[1].pos.y-springs[2].pos.y    
    
    


# In[36]:


scene.capture('x.png')


# In[37]:


#github.com/nikhilagrawal2000/Decoupling-MDOF-Coupled-Systems


# In[ ]:


#Please feel free to contribute, Happy Coding!!

