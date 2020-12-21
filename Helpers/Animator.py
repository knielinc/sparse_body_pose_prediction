from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mpl_anim
from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import cv2

def QImageToCvMat_2(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr

def QImageToCvMat(qtimg):
    # such as the odd 16-bit packed formats QT supports
    arrayptr = qtimg.bits()
    # QT may pad the image, so we need to use bytesPerLine, not width for
    # the conversion to a numpy array
    bytesPerPixel = qtimg.depth() // 8
    pixelsPerLine = qtimg.bytesPerLine() // bytesPerPixel
    img_size = pixelsPerLine * qtimg.height() * bytesPerPixel
    arrayptr.setsize(img_size)
    img = np.array(arrayptr)
    # Reshape and trim down to correct dimensions
    if bytesPerPixel > 1:
        img = img.reshape((qtimg.height(), pixelsPerLine, bytesPerPixel))
        img = img[:, :qtimg.width(), :]
    else:
        img = img.reshape((qtimg.height(), pixelsPerLine))
        img = img[:, :qtimg.width()]
    # Strip qt's false alpha channel if needed
    # and reorder color axes as required
    if bytesPerPixel == 4 and not qtimg.hasAlphaChannel():
        img = img[:, :, 2::-1]
    elif bytesPerPixel == 4:
        img[:, :, 0:3] = img[:, :, 2::-1]
    return img

class MocapAnimator:
    def __init__(self, global_positions, joint_names, bone_dependencies, frame_time, verbose=False, write_to_file=True):
        self.global_positions = global_positions
        self.frame_count = global_positions.shape[0]
        self.joint_count = global_positions.shape[1]
        self.joint_names = joint_names
        self.frame_time = frame_time
        self.verbose = verbose
        self.bone_dependencies = bone_dependencies
        self.write_to_file = write_to_file
        self.__initPyQT()
        #self.__setup_animation()

    def __setup_animation(self):
        self.fig = plt.figure()
        plt.gcf().set_size_inches(15, 18)
        self.ax = self.fig.add_subplot(111, projection='3d')

        xs = self.global_positions[0, :, 0]
        ys = self.global_positions[0, :, 2]
        zs = self.global_positions[0, :, 1]
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Z Label')
        self.ax.set_zlabel('Y Label')
        self.ax.scatter(xs, ys, zs, c='r', marker='o')

        xs_ = self.global_positions[:, :, 0]
        ys_ = self.global_positions[:, :, 2]
        zs_ = self.global_positions[:, :, 1]

        max_range = np.array([xs_.max() - xs_.min(), ys_.max() - ys_.min(), zs_.max() - zs_.min()]).max() / 2.0

        mid_x = (xs_.max() + xs_.min()) * 0.5
        mid_y = (ys_.max() + ys_.min()) * 0.5
        mid_z = (zs_.max() + zs_.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # start_time = time.ctime()
        #
        # for idx in range(self.frame_count):
        #     self.__animation_frame(idx)
        #
        # end_time = time.ctime()
        #
        # print(str(end_time- start_time))

        self.animation = mpl_anim.FuncAnimation(self.fig, func=self.__animation_frame,
                                                frames=np.arange(0, self.frame_count, 1),
                                                interval=self.frame_time * 1000)

    def play_animation(self):
        plt.show()

    def save_animation(self, file_name):
        FFMpegWriter = mpl_anim.writers['ffmpeg']
        writer = FFMpegWriter(fps=1.0 / self.frame_time, metadata=dict(artist='Me'), bitrate=1800)

        self.animation.save(file_name, writer=writer)

    def __animation_frame(self, i):
        xs = self.global_positions[i, :, 0]
        ys = self.global_positions[i, :, 2]
        zs = self.global_positions[i, :, 1]
        self.ax.cla()
        #
        # for curr_joint_idx in range(self.joint_count):
        #     self.ax.text(xs[curr_joint_idx], ys[curr_joint_idx], zs[curr_joint_idx], self.joint_names[curr_joint_idx], None, color='green', fontsize=8)
        self.ax.scatter(xs, ys, zs, c='r', marker='o')

        max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0

        mid_x = (xs.max() + xs.min()) * 0.5
        mid_y = (ys.max() + ys.min()) * 0.5
        mid_z = (zs.max() + zs.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        if self.verbose and i % 100 == 0:
            print("processed frame" + str(i))

    def __initPyQT(self):
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('Mocap Viewer')
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()

        # create the background grids
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        gz.scale(10, 10, 10)
        self.w.addItem(gz)

        self.frame_idx = 0

        pts = (self.global_positions[self.frame_idx])[:, [0, 2, 1]]

        self.bone_lines = []
        for bone_dependency in self.bone_dependencies:
            if bone_dependency[1] == -1:
                continue

            curr_bone_pos = pts[bone_dependency[0]]
            parent_bone_pos = pts[bone_dependency[1]]

            line = np.vstack((curr_bone_pos, parent_bone_pos))

            self.bone_lines.append(gl.GLLinePlotItem(pos=line, color=pg.glColor(0.5), width=10, antialias=True))
        for bone_line in self.bone_lines:
            self.w.addItem(bone_line)


        self.points = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(1.0), size=10)
        self.w.addItem(self.points)
        if self.write_to_file:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.out = cv2.VideoWriter('test_cv2.avi', fourcc, int(1.0 / self.frame_time), (1920, 1080))

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):
        if self.frame_idx < self.frame_count - 1:
            self.frame_idx += 1
            pts = (self.global_positions[self.frame_idx])[:, [0, 2, 1]]
            self.points.setData(pos=pts, color=pg.glColor(1.0), size=10)
            i = 0
            for bone_dependency in self.bone_dependencies:

                if bone_dependency[1] == -1:
                    continue


                curr_bone_pos = pts[bone_dependency[0]]
                parent_bone_pos = pts[bone_dependency[1]]

                line = np.vstack((curr_bone_pos, parent_bone_pos))

                self.bone_lines[i].setData(pos=line, color=pg.glColor(0.5), width=10, antialias=True)
                i += 1


            xs = self.global_positions[self.frame_idx, :, 0]
            zs = self.global_positions[self.frame_idx, :, 2]
            ys = self.global_positions[self.frame_idx, :, 1]

            max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0

            mid_x = (xs.max() + xs.min()) * 0.5
            mid_y = (ys.max() + ys.min()) * 0.5
            mid_z = (zs.max() + zs.min()) * 0.5

            curr_cam_pos = self.w.opts['center']
            curr_dist = self.w.opts['distance']
            new_dist = curr_dist + 0.05 * (max_range * 5 - curr_dist)
            new_cam_pos = curr_cam_pos + 0.05 * (Vector(mid_x, mid_z, mid_y) - curr_cam_pos)
            self.w.setCameraPosition(pos=new_cam_pos, distance=new_dist)

            if self.write_to_file:
                currQImage = self.w.grabFrameBuffer()
                cvMat = QImageToCvMat(currQImage)
                self.out.write(cvMat)
        else:
            if self.write_to_file:
                self.out.release()
            self.timer.stop()
            print("finished animation")

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.frame_time * 1000)
        self.start()
