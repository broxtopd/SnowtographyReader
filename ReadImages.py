import sys, os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import warnings
import scipy.io as sio
import copy
import PIL.Image, PIL.ExifTags
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

warnings.filterwarnings("ignore")
DEVNULL = open(os.devnull, 'wb')

S = ['5 ft', '4 ft', '3 ft', '2 ft', '1 ft', '0 ft']    # Major tick marks on the stakes
S_ht = [5, 4, 3, 2, 1, 0]                               # Heights corresponding to the tick marks
StartHour = 10                                          # Do not consider earlier images
EndHour = 16                                            # Do not consider later images

Stakes = []                                             # Initialized variable
marking_height = ''                                     # Initialized variable

def line_select_callback(eclick, erelease):

    global scale_factor, fig, ax
    
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata/scale_factor, eclick.ydata/scale_factor
    x2, y2 = erelease.xdata/scale_factor, erelease.ydata/scale_factor
    dict = {}
    dict['x1'] = int(x1)
    dict['x2'] = int(x2)
    dict['y1'] = int(y1)
    dict['y2'] = int(y2)
    dict['ClickedMarkings'] = []
    dict['ClickedMarkingHeights'] = []
    Stakes.append(dict)

    rect = patches.Rectangle((x1*scale_factor, y1*scale_factor), (x2-x1)*scale_factor, (y2-y1)*scale_factor, linewidth=1, edgecolor='r', facecolor='none')
    plt.text((x2+x1)/2*scale_factor,(y2+y1)/2*scale_factor, len(Stakes), fontsize=16, color='r')
    ax.add_patch(rect)
    fig.canvas.draw()


def toggle_selector(event):
    toggle_selector.RS.set_active(True)
    
    
def get_marking_height(event):
    global Stakes, n, d, marking_height, S, S_ht
    
    if event.key.isnumeric():
        marking_height = marking_height + event.key
    # elif event.key == 'enter':
        Stakes[n]['ClickedMarkingHeights'].append(int(marking_height))
        print('User entered a marking height of ' + marking_height + ' ft')
        marking_height = ''
        
        if len(Stakes[n]['ClickedMarkings']) == 1:
            plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Click a marking near the bottom of the stake that corresponds to a whole number of feet.')
            fig.canvas.draw()
        elif len(Stakes[n]['ClickedMarkings']) == 2:
            plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Are the markings in the right spot? Click image to redo, or close window to accept.')

            Stakes[n]['Markings'] = []
            for n2 in range(len(S)):
                x1, y1 = Stakes[n]['ClickedMarkings'][0]
                x2, y2 = Stakes[n]['ClickedMarkings'][1]
                h1 = Stakes[n]['ClickedMarkingHeights'][0]
                h2 = Stakes[n]['ClickedMarkingHeights'][1]
                pos = (h1-S_ht[n2])/(h1-h2)
                Stakes[n]['Markings'].append((int(x1+pos*(x2-x1)), int(y1+pos*(y2-y1))))
            
            TextOffset = (Stakes[n]['Markings'][4][1]-Stakes[n]['Markings'][0][1])*0.2
            c = 0
            for s in S:
                plt.plot(Stakes[n]['Markings'][c][0], Stakes[n]['Markings'][c][1], marker=".", markersize=10, color='r')
                plt.text(Stakes[n]['Markings'][c][0]-TextOffset, Stakes[n]['Markings'][c][1], s, fontsize=16, color='r')
                c = c+1
                  
            fig.canvas.draw()
            
            
def get_markings(event):

    global Stakes, n, d, file, m, done, fig, ax
    
    if m < 2:
        plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Enter the whole number of feet above the ground of the clicked marking.')
        plt.plot(event.xdata, event.ydata, marker=".", markersize=10, color='r')
        fig.canvas.draw()
        Stakes[n]['ClickedMarkings'].append((int(event.xdata),int(event.ydata)))
        m = m+1 
    else:
        m = 0
        done = False
        Stakes[n]['ClickedMarkings'] = []
        Stakes[n]['ClickedMarkingHeights'] = []
        plt.close()
    
    
def get_depth(event):
    global Stakes, depths, n, m, d, file, img_rgb_sub, fig, S
    
    if m == 0:
        diff_x = event.xdata - Stakes[n]['ClickedMarkings'][0][0]
        diff_y = event.ydata - Stakes[n]['ClickedMarkings'][0][1]
        for n2 in range(len(Stakes)):
            # if n2 >= n:
            Stakes[n2]['ClickedMarkings'][0][0] = Stakes[n2]['ClickedMarkings'][0][0] + diff_x
            Stakes[n2]['ClickedMarkings'][1][0] = Stakes[n2]['ClickedMarkings'][1][0] + diff_x
            Stakes[n2]['ClickedMarkings'][0][1] = Stakes[n2]['ClickedMarkings'][0][1] + diff_y
            Stakes[n2]['ClickedMarkings'][1][1] = Stakes[n2]['ClickedMarkings'][1][1] + diff_y
            for c in range(len(S)):
                Stakes[n2]['Markings'][c][0] = Stakes[n2]['Markings'][c][0] + diff_x
                Stakes[n2]['Markings'][c][1] = Stakes[n2]['Markings'][c][1] + diff_y
        
        plt.clf()
        plt.imshow(img_rgb_sub)
        
        TextOffset = (Stakes[n]['Markings'][4][1]-Stakes[n]['Markings'][0][1])*0.2
        c = 0
        for s in S:
            plt.plot(Stakes[n]['Markings'][c][0], Stakes[n]['Markings'][c][1], marker=".", markersize=10, color='r')
            plt.text(Stakes[n]['Markings'][c][0]-TextOffset, Stakes[n]['Markings'][c][1], s, fontsize=16, color='r')
            c = c+1
                        
        plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Click the snow depth (0 or below for 0 depth).  Close window to leave as missing but try again if there is another image on ' + datetime.strftime(d,'%Y/%m/%d') + ', press "w" to leave as missing for the date, press "m" to move the markings.')
        fig.canvas.draw()
        m = m+1
        
    else:
        depth = S_ht[0]*12-float((event.ydata-Stakes[n]['Markings'][0][1])/(Stakes[n]['Markings'][len(S)-1][1]-Stakes[n]['Markings'][0][1])*S_ht[0]*12)
        if depth < 0:
            depth = 0
        depths[n] = depth
        print('- Clicked snow depth: ' + '{:.1f}'.format(depths[n]) + ' in')
        plt.close()
    


def move_stake(event):
    global m, d, n, file, depths, Stake
    
    if event.key == 'm':
        m = 0
        plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Click at the top of the ' + str(Stakes[n]['ClickedMarkingHeights'][0]) + ' ft marking on the stake.')
        fig.canvas.draw()
    elif event.key == 'w':
        depths[n] = '-2'
        print('- Snow depth set to missing')
        plt.close()
        
        
def toggle_reselect(event):
    global reselect
    
    plt.close()
    reselect = True
        
def ReselectAll():
    global Stakes, Stake, n, m, dim, done, file, d, ax, fig
    
    Stakes = []

    print('Getting the locations of all of the stakes')

    done = False
    
    while not done:
        fig, ax = plt.subplots()
        toggle_selector.RS = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=True, button=[1, 3], minspanx=5, minspany=5, spancoords='pixels', interactive=False)
        plt.connect('key_press_event', toggle_selector)
        plt.imshow(cv2.resize(img_rgb, dim, interpolation = cv2.INTER_AREA))
        plt.title(file + ' - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Draw boxes around all of the visible stakes (must draw at least 1). When finished, close the window.')
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        if len(Stakes) > 0:
            done = True
    
    n = 0
    for Stake in Stakes:
        print('Getting the position data for stake #' + str(n+1))
        done = False
        while not done:
            done = True
            m = 0
            fig = plt.figure()
            cid = fig.canvas.mpl_connect('button_press_event', get_markings)
            cid = fig.canvas.mpl_connect('key_press_event', get_marking_height)
            img_rgb_sub = img_rgb[Stake['y1']:Stake['y2'], Stake['x1']:Stake['x2']]
            plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Click a marking near the top of the stake that corresponds to a whole number of feet.')
            plt.imshow(img_rgb_sub)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
        
        n = n+1
                    
    # Save Region info to file
    sio.savemat(image_dir + '_info.mat',{'Stakes': Stakes}) 
    
    
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
print('')
print('Use the dialogue box to open the directory of camera images...')
image_dir = askdirectory(title='Open the image directory')
if image_dir == '':
    print('No image directory was provided.  Must have a working image directory...')
    sys.exit()
print('Image directory set to ' + image_dir)
    
datelist = []
filelist = []

if os.path.exists(image_dir + '_data.csv'):
    f = open(image_dir + '_data.csv', 'r')
    lines = f.readlines()
    c = 0
    for line in lines:
        if c > 0:
            datelist.append(line.split(',')[0])
            filelist.append(line.split(',')[2])
        c = c+1
                   
    depths_str = line.split(',')[3:]
    depths = np.ones(len(depths_str))*np.nan
    for d in range(len(depths_str)):
        depths[d] = float(depths_str[d])
                    
firsttime = True                   
for file in os.listdir(image_dir):

    # Load CSV file, skip dates that have already been processed
    if file[-4:] == '.JPG' or file[-4:] == '.jpg':
    
        img = PIL.Image.open(image_dir + '/' + file)
        exif_data = img._getexif()
        d = datetime.strptime(exif_data[36868],'%Y:%m:%d %H:%M:%S')
        df = datetime.strftime(d,'%m/%d/%Y').replace('/0', '/')
        if df[0] == '0':
            df = df[1:]
        tf = datetime.strftime(d,'%H:%M')
        
        if file not in filelist:
            if (df not in datelist or (df == datelist[-1] and np.sum(depths == -1) > 0)) and d.hour >= StartHour and d.hour <= EndHour : 
            
                print('Loading ' + image_dir + '/' + file)
                img = cv2.imread(image_dir + '/' + file)  
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                scale = img_rgb.shape
                scale_factor = 1500/scale[1]
                width = int(scale[1] * scale_factor)
                height = int(scale[0] * scale_factor)
                dim = (width, height)
                
                # If region info exists, load it
                if os.path.exists(image_dir + '_info.mat'):
                    a = sio.loadmat(image_dir + '_info.mat',simplify_cells=True)
                    Stakes = a['Stakes']
                    try:
                        a = Stakes[0]
                    except:
                        Stake = Stakes
                        Stakes = []
                        Stakes.append(Stake)
                        
                    if firsttime:
                        print('Displaying the existing boxes...')
                        reselect = False
                        img_rgb_annotated = copy.deepcopy(img_rgb)
                        c = 0
                        for Stake in Stakes:
                            c = c+1
                            img_rgb_annotated = cv2.rectangle(img_rgb_annotated, (Stake['x1'], Stake['y1']), (Stake['x2'], Stake['y2']), (255,0,0), 10)
                            fontScale = 5
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            color = (255,0,0)
                            thickness = 10
                            img_rgb_annotated = cv2.putText(img_rgb_annotated, str(c), (int((Stake['x1']+Stake['x2'])/2),int((Stake['y1']+Stake['y2'])/2)), font, fontScale, color, thickness, cv2.LINE_AA)
                        
                        img_rgb_annotated = cv2.convertScaleAbs(img_rgb_annotated, alpha=1.5, beta=1)
                        
                        fig = plt.figure()
                        plt.imshow(cv2.resize(img_rgb_annotated, dim, interpolation = cv2.INTER_AREA))
                        plt.title(file + ' - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Are the boxes in the right spot? Close window if no change, or click in the window if you want to change them.')
                        manager = plt.get_current_fig_manager()
                        manager.window.showMaximized()
                        plt.connect('button_press_event', toggle_reselect)
                        plt.show()
                        
                        if reselect:
                            ReselectAll()
                        firsttime = False
                else:
                    ReselectAll()
                    
                    firsttime = False
                
                if not os.path.exists(image_dir + '_data.csv'):
                    depths = np.ones(len(Stakes))*-1
                ex_stake = []
                for i in range(len(depths)):
                    if datelist == []:
                        ex_stake.append(True)
                    else:
                        if df == datelist[-1]:
                            if depths[i] == -1:
                                ex_stake.append(True)
                            else:
                                ex_stake.append(False)
                        else:
                            ex_stake.append(True)
                    
                
                # Loop through visible Stakes
                n = 0
                depths = np.ones(len(Stakes))*-1
                for Stake in Stakes:
                    print('Getting Snow Depth for Stake #' + str(n+1))
                   
                    if ex_stake[n]:
                        m = 1
                        fig = plt.figure()
                        manager = plt.get_current_fig_manager()
                        manager.window.showMaximized()
                        cid = fig.canvas.mpl_connect('button_press_event', get_depth)
                        cid = fig.canvas.mpl_connect('key_press_event', move_stake)
                        
                        img_rgb_sub = img_rgb[Stake['y1']:Stake['y2'], Stake['x1']:Stake['x2']]
                        
                        img_rgb_sub = cv2.convertScaleAbs(img_rgb_sub, alpha=1.5, beta=1)
                        
                        plt.title(file + ' (Stake #' + str(n+1) + ') - ' + datetime.strftime(d,'%Y/%m/%d %H:%M') + '\n Click the snow depth (0 or below for 0 depth).  Close window to leave as missing but try again if there is another image on ' + datetime.strftime(d,'%Y/%m/%d') + ', press "w" to leave as missing for the date, press "m" to move the markings.')
                        plt.imshow(img_rgb_sub)

                        TextOffset = (Stake['Markings'][4][1]-Stake['Markings'][0][1])*0.2
                        c = 0
                        for s in S:
                            plt.plot(Stake['Markings'][c][0], Stake['Markings'][c][1], marker=".", markersize=10, color='r')
                            plt.text(Stake['Markings'][c][0]-TextOffset, Stake['Markings'][c][1], s, fontsize=16, color='r')
                            c = c+1

                        plt.show()
                        
                    else:
                        depths[n] = -2
                    
                    n = n+1
                    sio.savemat(image_dir + '_info.mat',{'Stakes': Stakes})
                    
            
                # If the file doesn't exist, write the headers before we write data
                if not os.path.exists(image_dir + '_data.csv'):
                    f = open(image_dir + '_data.csv', 'w')
                    stakes_str = ''
                    c = 0
                    for depth in depths:
                        stakes_str = stakes_str + ',Depth #' + str(c+1) + ' (in)'
                        c = c+1
                    f.write('Date,Time,Image' + stakes_str + '\n')
                    f.close()
                    
                # Update CSV file
                f = open(image_dir + '_data.csv', 'a')
                depths_str = ''
                for depth in depths:
                    depths_str = depths_str + ',{:.1f}'.format(depth)
                f.write(df + ',' + tf + ',' + file + depths_str + '\n')
                f.close()
                
                datelist = [datelist, df]
                filelist = [filelist, file]
            