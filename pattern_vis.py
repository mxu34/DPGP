#!usr/bin/env python3

from argoverse.map_representation.map_api import ArgoverseMap
from frame import Frame
import matplotlib.pyplot as plt
import pickle
import numpy as np
from argo import draw_local_map


# Frames in cluster visualization
def frame_in_pattern_vis(xmin, xmax, ymin, ymax):
    dataset = 'ARGO'
    if dataset == 'NGSIM':

        with open("data_sample/a_mixture_model_NGSIM_200", "rb") as mix_np:  # load saved mixture model
            mix_model = pickle.load(mix_np)

        with open("data_sample/frame_US_101_200", "rb") as frame_np: # load saved frames
            load_frames = pickle.load(frame_np)

        print('everything loaded')
        # visualize frames from the same pattern
        pattern_num = np.argmax(np.array(mix_model.partition))
        pattern_idx = idx = np.where(np.array(mix_model.z) == pattern_num)
        pattern_idx = np.asarray(pattern_idx)
        pattern_idx = pattern_idx[0].astype(int)

        plt.ion()
        for i in range(mix_model.partition[pattern_num]):
        # the on road is much more stable, however the off road ones are quite noisy
            plt.cla()
            frame_temp = load_frames[pattern_idx[i]]
            plt.quiver(frame_temp.x, frame_temp.y, frame_temp.vx, frame_temp.vy)
            plt.xlim([0, 60])
            plt.ylim([1300, 1600])

            plt.show()
            plt.pause(0.05)

        plt.ioff()

    elif dataset == 'ARGO':

        with open("data_sample/argo_MixtureModel_%d_%d_%d_%d" % (xmin, xmax, ymin, ymax),
                  "rb") as mix_np:  # load saved mixture model
            mix_model = pickle.load(mix_np)

        with open("data_sample/argo_%d_%d_%d_%d" % (xmin, xmax, ymin, ymax), "rb") as frame_np:  # load saved frames
            load_frames = pickle.load(frame_np)

        print('everything loaded')
        # visualize frames from the same pattern
        for i in range(mix_model.K):
            # pattern_num = np.argmax(np.array(mix_model.partition))
            pattern_num = i
            pattern_idx = idx = np.where(np.array(mix_model.z) == pattern_num)
            pattern_idx = np.asarray(pattern_idx)
            pattern_idx = pattern_idx[0].astype(int)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            avm = ArgoverseMap()
            city_name = 'PIT'
            rate = np.array(mix_model.partition)[pattern_num] / mix_model.n

            plt.ion()
            for i in range(mix_model.partition[pattern_num]):
                # the on road is much more stable, however the off road ones are quite noisy
                plt.cla()
                frame_temp = load_frames[pattern_idx[i]]
                draw_local_map(avm, city_name, xmin, xmax, ymin, ymax, ax)
                plt.quiver(frame_temp.x, frame_temp.y, frame_temp.vx, frame_temp.vy, color='#ED5107')
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
                name = 'PIT_%d_%d_%d_%d_' % (xmin, xmax, ymin, ymax) + str(pattern_num) + '_' + str(round(rate, 3))
                plt.title(name)
                plt.show()
                plt.pause(0.05)

            plt.ioff()


# velocity field visualization
def velocity_field_visualization(xmin, xmax, ymin, ymax):
    with open("data_sample/argo_MixtureModel_%d_%d_%d_%d" % (xmin, xmax, ymin, ymax),
              "rb") as mix_np:  # load saved mixture model
        mix_model = pickle.load(mix_np)

    with open("data_sample/argo_%d_%d_%d_%d" % (xmin, xmax, ymin, ymax), "rb") as frame_np:  # load saved frames
        load_frames = pickle.load(frame_np)

    print('everything loaded')
    # visualize frames from the same pattern

    # for i in range(mix_model.K):
    for i in range(1):
        pattern_num = i
        pattern_num = np.argmax(np.array(mix_model.partition))
        rate = np.array(mix_model.partition)[i]/mix_model.n
        frame_pattern_ink = mix_model.frame_ink(pattern_num, 0, True)
        # construct mesh frame
        x = np.linspace(xmin, xmax, 31)
        y = np.linspace(ymin, ymax, 31)
        [WX,WY] = np.meshgrid(x, y)
        WX = np.reshape(WX, (-1, 1))
        WY = np.reshape(WY, (-1, 1))
        frame_field = Frame(WX.ravel(), WY.ravel(), np.zeros(len(WX)), np.zeros(len(WX)))
        #get posterior
        ux_pos, uy_pos, covx_pos, covy_pos = mix_model.b[pattern_num].GP_posterior(frame_field, frame_pattern_ink, True)

        print('now start plotting')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        avm = ArgoverseMap()
        city_name = 'PIT'
        plt.quiver(WX, WY, ux_pos, uy_pos, width=0.002, color='#ED5107')
        draw_local_map(avm, city_name, xmin, xmax, ymin, ymax, ax)
        plt.xlabel('x_map_coordinate')
        plt.ylabel('y_map_coordinate')
        name = 'PIT_%d_%d_%d_%d_' % (xmin, xmax, ymin, ymax) +str(pattern_num) + '_' + str(round(rate, 3))
        print(name)
        plt.title(name)
        plt.savefig('fig/'+name + '.png')
        plt.close()
        plt.show()

    # Note that it doesn't have to show the
    return WX, WY, ux_pos, uy_pos


# dataset = 'ARGO'
# vis_vel_field = True
#
# if vis_vel_field:
#     WX, WY, ux_pos, uy_pos = velocity_field_visualization(2570, 2600, 1180, 1210)
# else:
#     frame_in_pattern_vis(dataset)
#
# print('Visualization Finished')