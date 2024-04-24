import numpy as np
import math
import cv2
import sys
import os
import datetime
import tqdm


#bottom-imgとtop-imgが逆ですが、実行に影響はありません。

#x = 0.5, y = width, z = height
def back_map():

    x_tmp = [[0.5]*out_img_w]
    tmp = [[0]*out_img_w]

    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    y, z = np.meshgrid(w, h)

    xx, tt = np.meshgrid(x_tmp, tmp)

    rho = np.sqrt(xx*xx + y*y + z*z)

    x = xx/rho
    y = y/rho 
    z = z/rho

    return x,y,z

#x = width, y = 0.5, z = height
def left_map():

    y_tmp = [[0.5]*out_img_w]
    tmp = [[0]*out_img_w]

    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    x, z = np.meshgrid(w, h)

    yy, tt = np.meshgrid(y_tmp, tmp)

    rho = np.sqrt(x*x + yy*yy + z*z)

    x = x/rho
    y = yy/rho 
    z = z/rho

    return x,y,z

#x = width, y = -0.5, z = height
def right_map():

    y_tmp = [[-0.5]*out_img_w]
    tmp = [[0]*out_img_w]

    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    x, z = np.meshgrid(w, h)

    yy, tt = np.meshgrid(y_tmp, tmp)

    rho = np.sqrt(x*x + yy*yy + z*z)

    x = x/rho
    y = yy/rho 
    z = z/rho

    return x,y,z

#x = -0.5, y = witdh, z = height
def front_map():

    x_tmp = [[-0.5]*out_img_w]
    tmp = [[0]*out_img_w]

    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    y, z = np.meshgrid(w, h)

    xx, tt = np.meshgrid(x_tmp, tmp)

    rho = np.sqrt(xx*xx + y*y + z*z)

    x = xx/rho
    y = y/rho 
    z = z/rho

    return x,y,z

#x = width, y = height, z = -0.5
def bottom_map():

    z_tmp = [[-0.5]*out_img_w]
    tmp = [[0]*out_img_w]

    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    x, y = np.meshgrid(w, h)

    zz, tt = np.meshgrid(z_tmp, tmp)

    rho = np.sqrt(x*x + y*y + zz*zz)

    x = x/rho
    y = y/rho 
    z = zz/rho

    return x,y,z

#x = width, y = height, z = 0.5
def top_map():

    z_tmp = [[0.5]*out_img_w]
    tmp = [[0]*out_img_w]


    #senser_ = 0.5
    #w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    #h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    #w = w + senser_ / out_img_w
    #h = h + senser_ / out_img_w

    w, h = gen_grid()

    x, y = np.meshgrid(w, h)

    zz, tt = np.meshgrid(z_tmp, tmp)

    rho = np.sqrt(x*x + y*y + zz*zz)

    x = x/rho
    y = y/rho 
    z = zz/rho

    return x,y,z


def phi_theta(x, y, z):

    """ 
    # 緯度・経度へ変換
    theta = np.arctan2(y,x) 
    phi = np.arccos(z)
    """

    phi = np.arcsin(z)
    theta = np.arcsin(np.clip(y / np.cos(phi), -1, 1))
     
    theta = np.where((x<0) & (y<0), -np.pi-theta, theta)
    theta = np.where((x<0) & (y>0),  np.pi-theta, theta)
 
    return phi, theta

def gen_grid():

    senser_ = 0.55
    w = np.linspace(-senser_, senser_, out_img_w, endpoint=False)
    h = np.linspace(-senser_, senser_, out_img_w, endpoint=False)

    w = w + senser_ / out_img_w
    h = h + senser_ / out_img_w

    return w,h


def remap_output(phi,theta, out_img_path, img, out_dir):

    img_h, img_w = img.shape[:2]
    phi = (phi * img_h / np.pi + img_h / 2).astype(np.float32) - 0.5
    theta = (theta * img_w / (2 * np.pi) + img_w / 2).astype(np.float32) - 0.5
    out_img = cv2.remap(img, theta.astype("float32"), phi.astype("float32"), cv2.INTER_CUBIC)
    return out_img
    #cv2.imwrite(out_img_path, out_img)
    #out_img = cv2.flip(out_img, 1)
    #cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

def main(img, img_id, out_dir, ext):
    x1, y1, z1, = back_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_back_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

    x1, y1, z1, = right_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_right_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

    x1, y1, z1, = left_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_left_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    out_img = cv2.flip(out_img, 1)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

    x1, y1, z1, = front_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_front_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    out_img = cv2.flip(out_img, 1)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

    #bottomとtopが逆ですが、実行に影響はありません。
    x1, y1, z1, = bottom_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_top_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    out_img = cv2.flip(out_img, 1)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)

    x1, y1, z1, = top_map()
    phi, theta = phi_theta(x1,y1,z1)
    out_img_path = "" + img_id + "_bottom_img" + ext
    out_img = remap_output(phi, theta, out_img_path, img, out_dir)
    out_img = cv2.flip(out_img, 1)
    cv2.imwrite(os.path.join(out_dir, out_img_path), out_img)


if __name__ == "__main__":

    out_img_w = 800
    arg = sys.argv
    senser = 0.55
    
    x = senser 
    y = 0.5 
 
    a = np.array([x, y])
    b = np.array([0, 0])
    vec = a - b
 
    r = np.arctan2(vec[0], vec[1])
    degree = r * (180 / np.pi)
    degree = degree * 2

    if len(arg) <= 1:
        print("\033[31m" +"error: need video path $python v2i-eq2cube.py ~" + "\033[0m")
        sys.exit()
    
    input_video_path = arg[1]
    cap = cv2.VideoCapture(input_video_path)
    ext = ".jpg"

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    dtime = now.strftime("%Y_%m_%d_%H_%M")
    
    step_f = 5 #切り出すフレーム数

    save_data_dir = "out_" + dtime + "_frameN_" + str(step_f) + "_AngleDegree_" + str(int(degree))
    #save_data_dir = "output"
    os.mkdir(save_data_dir)

    if not cap.isOpened():
        print("\033[31m" +"error: video cant load" + "\033[0m")
        sys.exit()
    
    digit  = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    
    """
    pbar = tqdm.tqdm(total = max_frame)
    #with tqdm()
    while True:

        is_img, frame = cap.read()

        if is_img:
            frame_id = str(n).zfill(digit)
            main(frame, frame_id, save_data_dir, ext)
        else:
            break
        n += 3
        pbar.update(n)
    pbar.close()
    """
    
    """
    while True:

        is_img, frame = cap.read()

        if is_img:
            frame_id = str(n).zfill(digit)
            main(frame, frame_id, save_data_dir, ext)
        else:
            break
        n += 1
    """
    
    for n in tqdm.tqdm(range(0, max_frame, step_f)):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        is_img, frame = cap.read()
        
        if is_img:
            frame_id = str(n).zfill(digit)
            main(frame, frame_id, save_data_dir, ext)
        else:
            break
        
  
    print("\033[34m" + "completion!!" + "\033[0m")
