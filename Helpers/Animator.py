from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mpl_anim
import matplotlib.patheffects as pe

from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import cv2
from Helpers import DataPreprocessor as dp
from scipy.spatial.transform import Rotation as R
import time

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
    def __init__(self, global_positions, joint_names, bone_dependencies, frame_time, verbose=False, write_to_file=True, heading_dirs=None, name="Animation"):
        self.global_positions = global_positions
        self.frame_count = global_positions.shape[0]
        self.joint_count = global_positions.shape[1]
        self.joint_names = joint_names
        self.frame_time = frame_time
        self.verbose = verbose
        self.bone_dependencies = bone_dependencies
        self.write_to_file = write_to_file
        self.heading_dirs = heading_dirs
        self.name = name
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

    def __get_arrow(self, heading_dir, origin, scale=1):
        #heading_dir = dp.get_angles_from_data(self.global_positions[self.frame_idx:self.frame_idx + 1], self.l_shoulder_idx, self.l_shoulder_idx, self.hip_idx)
        x_ = np.cos(heading_dir)
        y_ = np.sin(heading_dir)

        dir = np.array([x_, y_, 0]) * scale

        rotate_fwd = R.from_euler('z', 90, degrees=True)
        rotate_bwd = R.from_euler('z', -90, degrees=True)

        start_pos = origin
        end_pos = origin + dir
        end1 = end_pos - dir * 0.5 + rotate_fwd.apply(dir) * 0.5
        end2 = end_pos - dir * 0.5 + rotate_bwd.apply(dir) * 0.5

        line = np.vstack((start_pos, end_pos))
        line2 = np.vstack((end_pos, end1))
        line3 = np.vstack((end_pos, end2))

        return [line, line2, line3]

    def __initPyQT(self):
        self.traces = dict()
        global app
        self.app = QtGui.QApplication(sys.argv)

        self.window = gl.GLViewWidget()
        self.window.opts['distance'] = 40
        self.window.setWindowTitle('Mocap Viewer')
        self.window.setGeometry(0, 110, 512, 512)
        self.window.show()


        self.dot_radius = 3
        self.line_width = 4
        self.frame_idx = 0
        self.line_color = pg.glColor(0.5)
        self.arrow_color = pg.glColor((255, 0, 0, 150))

        pts = (self.global_positions[self.frame_idx])[:, [0, 2, 1]]

        # create the background grids
        self.gz = gl.GLGridItem()
        self.gz.translate(0, 0, self.global_positions[:, :, 1].min())
        self.gz.scale(10, 10, 10)
        self.window.addItem(self.gz)

        self.bone_lines = []
        for bone_dependency in self.bone_dependencies:
            if bone_dependency[1] == -1:
                continue

            curr_bone_pos = pts[bone_dependency[0]]
            parent_bone_pos = pts[bone_dependency[1]]

            line = np.vstack((curr_bone_pos, parent_bone_pos))

            self.bone_lines.append(gl.GLLinePlotItem(pos=line, color=self.line_color, width=self.line_width, antialias=True))

        if not self.heading_dirs is None:
            origin = np.mean(pts, axis=0)
            line, line2, line3 = self.__get_arrow(self.heading_dirs[self.frame_idx], origin, 10)

            self.bone_lines.append(gl.GLLinePlotItem(pos=line, color=self.arrow_color, width=self.line_width, antialias=True))
            self.bone_lines.append(gl.GLLinePlotItem(pos=line2, color=self.arrow_color, width=self.line_width, antialias=True))
            self.bone_lines.append(gl.GLLinePlotItem(pos=line3, color=self.arrow_color, width=self.line_width, antialias=True))

        for bone_line in self.bone_lines:
            self.window.addItem(bone_line)

        self.points = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(1.0), size=10)
        self.window.addItem(self.points)
        if self.write_to_file:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.out = cv2.VideoWriter(self.name, fourcc, int(1.0 / self.frame_time), (512, 512))

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

                self.bone_lines[i].setData(pos=line, color=self.line_color, width=self.line_width, antialias=True)
                i += 1

            xs = self.global_positions[self.frame_idx, :, 0]
            zs = self.global_positions[self.frame_idx, :, 2]
            ys = self.global_positions[self.frame_idx, :, 1]

            max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0

            mid_x = (xs.max() + xs.min()) * 0.5
            mid_y = (ys.max() + ys.min()) * 0.5
            mid_z = (zs.max() + zs.min()) * 0.5

            curr_cam_pos = self.window.opts['center']
            curr_dist = self.window.opts['distance']
            new_dist = curr_dist + 0.05 * (max_range * 7 - curr_dist)
            new_cam_pos = curr_cam_pos + 0.05 * (Vector(mid_x, mid_z, mid_y) - curr_cam_pos)
            self.window.setCameraPosition(pos=new_cam_pos, distance=new_dist)

            if not self.heading_dirs is None:
                origin = np.mean(pts, axis=0)
                line, line2, line3 = self.__get_arrow(self.heading_dirs[self.frame_idx], origin, new_dist / 5)

                self.bone_lines[i].setData(pos=line, color=self.arrow_color, width=self.line_width, antialias=True)
                self.bone_lines[i+1].setData(pos=line2, color=self.arrow_color, width=self.line_width, antialias=True)
                self.bone_lines[i+2].setData(pos=line3, color=self.arrow_color, width=self.line_width, antialias=True)

            if self.write_to_file:
                currQImage = self.window.grabFrameBuffer()
                cvMat = QImageToCvMat(currQImage)
                self.out.write(cvMat)
        else:
            if self.write_to_file:
                self.out.release()

            self.timer.stop()
            self.window.clear()
            self.window.reset()
            self.app.closeAllWindows()

            print("finished animation")

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.frame_time * 1000)
        self.start()
        self.app.quit()
        self.app.exit()
        print("test")

class MocapAnimator2:
    def __init__(self, global_positions, joint_names, bone_dependencies, frame_time, verbose=False, write_to_file=True, heading_dirs=None, name="Animation.mp4"):
        self.global_positions = global_positions
        self.frame_count = global_positions.shape[0]
        self.joint_count = global_positions.shape[1]
        self.joint_names = joint_names
        self.frame_time = frame_time
        self.verbose = verbose
        self.bone_dependencies = bone_dependencies
        self.write_to_file = write_to_file
        self.heading_dirs = heading_dirs
        self.name = name
        self.__init_matplotlib()

    def __init_matplotlib(self):

        self.fig = plt.figure()
        plt.gcf().set_size_inches(5, 5)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.grid(True)
        self.ax.set_facecolor('dimgray')
        self.ax.set_axis_off()
        pts = self.global_positions[0]
        # xs = pts[:, 0]
        # ys = pts[:, 2]
        # zs = pts[:, 1]
        # self.ax.set_xlabel('X Label')
        # self.ax.set_ylabel('Z Label')
        # self.ax.set_zlabel('Y Label')
        # self.ax.scatter(xs, ys, zs, c='r', marker='o')
        # self.ax.azim = 120.5
        # self.ax.elev = 51.25
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


        # self.points.setData(pos=pts, color=pg.glColor(1.0), size=10)
        xs = np.linspace(-10, 10, 50)
        ys = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros(X.shape)

        wframe = self.ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey', lw=0.2)

        self.points = []
        self.bone_lines = []

        for bone_dependency in self.bone_dependencies:
            if bone_dependency[1] == -1:
                continue
            plt_line = plt.plot([0,0], [0,0], [0,0], color='red', lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0]

            self.bone_lines.append(plt_line)

        if not self.heading_dirs is None:

            plt_line1 = plt.plot([0,0], [0,0], [0,0], color='blue', lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0]
            plt_line2 = plt.plot([0,0], [0,0], [0,0], color='blue', lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0]
            plt_line3 = plt.plot([0,0], [0,0], [0,0], color='blue', lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0]

            self.bone_lines.append(plt_line1)
            self.bone_lines.append(plt_line2)
            self.bone_lines.append(plt_line3)

    def __get_arrow(self, heading_dir, origin, scale=1):
        #heading_dir = dp.get_angles_from_data(self.global_positions[self.frame_idx:self.frame_idx + 1], self.l_shoulder_idx, self.l_shoulder_idx, self.hip_idx)
        x_ = np.cos(heading_dir)
        y_ = np.sin(heading_dir)

        dir = np.array([x_, 0, y_]) * scale

        rotate_fwd = R.from_euler('y', 90, degrees=True)
        rotate_bwd = R.from_euler('y', -90, degrees=True)

        start_pos = origin
        end_pos = origin + dir
        end1 = end_pos - dir * 0.5 + rotate_fwd.apply(dir) * 0.5
        end2 = end_pos - dir * 0.5 + rotate_bwd.apply(dir) * 0.5

        line = np.vstack((start_pos, end_pos))
        line2 = np.vstack((end_pos, end1))
        line3 = np.vstack((end_pos, end2))

        return [line, line2, line3]

    def animate(self, i):
        changed = []

        idx = 0
        for bone_dependency in self.bone_dependencies:
            if bone_dependency[1] == -1:
                continue

            pts = self.global_positions[i]

            curr_bone_pos = pts[bone_dependency[0]]
            parent_bone_pos = pts[bone_dependency[1]]

            line = np.vstack((curr_bone_pos, parent_bone_pos))

            data_2d = line[:,[0,2]].T
            data_3d = line[:,1]
            self.bone_lines[idx].set_data(data_2d)
            self.bone_lines[idx].set_3d_properties(data_3d)
            idx += 1

        if not self.heading_dirs is None:
            origin = np.mean(pts, axis=0)
            line, line2, line3 = self.__get_arrow(self.heading_dirs[i], origin, .5)

            data_2d = line[:,[0,2]].T
            data_3d = line[:,1]
            self.bone_lines[idx].set_data(data_2d)
            self.bone_lines[idx].set_3d_properties(data_3d)

            idx += 1

            data_2d = line2[:,[0,2]].T
            data_3d = line2[:,1]
            self.bone_lines[idx].set_data(data_2d)
            self.bone_lines[idx].set_3d_properties(data_3d)

            idx += 1

            data_2d = line3[:,[0,2]].T
            data_3d = line3[:,1]
            self.bone_lines[idx].set_data(data_2d)
            self.bone_lines[idx].set_3d_properties(data_3d)

    def animation(self):
        plt.tight_layout()

        ani = mpl_anim.FuncAnimation(self.fig, self.animate, np.arange(self.frame_count), interval=1000 * self.frame_time)

        if self.name != None:
            ani.save(self.name, fps=1.0/self.frame_time, bitrate=13934)
            ani.event_source.stop()
            del ani
            plt.close()
        try:
            plt.show()
            plt.save()
        except AttributeError as e:
            pass