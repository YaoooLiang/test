from detect_eye import process_frame
from elg_model import elg
import numpy as np
import cv2
import tensorflow as tf
import time
from collections import deque
left_gaze_history = None
right_gaze_history = None


def init_queue():
    global left_gaze_history
    global right_gaze_history
    if left_gaze_history is None:
        left_gaze_history = deque(maxlen=60)
    if right_gaze_history is None:
        right_gaze_history = deque(maxlen=60)


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


def _visualize_output(frame, eye, outs):
    eye_img = eye['image']
    eye_side = eye['side']
    eye_landmarks = outs['landmarks'][0, :]
    eye_radius = outs['radius'][0][0]
    inv_landmarks_trans = eye['inv_landmarks_transform_mat']
    if eye_side == 'left':
        eye_landmarks[:, 0] = eye_img.shape[1] - eye_landmarks[:, 0]
    eye_landmarks = np.concatenate([eye_landmarks,
                                [[eye_landmarks[-1, 0] + eye_radius,
                                  eye_landmarks[-1, 1]]]])
    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                   'constant', constant_values=1.0))
    eye_landmarks = (eye_landmarks * inv_landmarks_trans.T)[:, :2]
    eye_landmarks = np.asarray(eye_landmarks)
    # eyelid_landmarks = eye_landmarks[0:8, :]
    # iris_landmarks = eye_landmarks[8:16, :]
    iris_centre = eye_landmarks[16, :]
    eyeball_centre = eye_landmarks[17, :]
    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                    eye_landmarks[17, :])
    i_x0, i_y0 = iris_centre
    e_x0, e_y0 = eyeball_centre
    theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
    phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                            -1.0, 1.0))
    current_gaze = np.array([theta, phi])
    left_gaze_history.append(phi)
    right_gaze_history.append(theta)
    if eye_side == 'left':
        pitch_str = '%f pitch' % (np.mean(right_gaze_history, 0))
        yaw_str = '%f yaw' % (np.mean(left_gaze_history, 0))
        fh, fw, _ = frame.shape
        cv2.putText(frame, pitch_str, org=(fw - 110, fh - 20),
                                       fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                                       color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(frame, pitch_str, org=(fw - 111, fh - 21),
                                       fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.79,
                                       color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    # gaze_history.append(current_gaze)
    # gaze_history_max_len = 10
    # if len(gaze_history) > gaze_history_max_len:
    #     gaze_history = gaze_history[-gaze_history_max_len:]
    draw_gaze(frame, iris_centre, current_gaze,
                        length=120.0, thickness=1)

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    cnt = 0
    input_frame = tf.placeholder(dtype=tf.float32, shape=[1, 36, 60, 1], name="input_frame")
    out = elg(input_frame)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    vars_hourglass = tf.global_variables(scope="hourglass")
    vars_radius = tf.global_variables(scope="radius")
    saver = tf.train.Saver()
    saver1 = tf.train.Saver(vars_hourglass)
    saver2 = tf.train.Saver(vars_radius)
    saver1.restore(sess, "ELG_i60x36_f60x36_n32_m2/checkpoints/hourglass/model-4672654")
    saver2.restore(sess, "ELG_i60x36_f60x36_n32_m2/checkpoints/radius/model-4672654")
    # saver.save(sess, 'model/elg.ckpt')
    last_frame_time = time.time()
    fps_history = []
    init_queue()
    while(cap.isOpened()):
        flag, im_bgr = cap.read()
        im_bgr = cv2.flip(im_bgr, 1)
        k = cv2.waitKey(1)
        eyes, frame = process_frame(im_bgr, cnt)
        for eye in eyes:
            eye_image = eye['image']
            eye_tensor = np.expand_dims(eye_image, 0)
            time1 = time.time()
            outs = sess.run(out, feed_dict={input_frame: eye_tensor})
            time2 = time.time()
            print("run time:", time2 - time1)
            im_bgr = _visualize_output(frame, eye, outs)
            time3 = time.time()
        if (k == ord('q')):
            break
        fps = int(np.round(1.0 / (time.time() - last_frame_time)))
        fps_history.append(fps)
        if len(fps_history) > 60:
            fps_history = fps_history[-60:]
        fps_str = '%d FPS' % np.mean(fps_history)
        fh, fw, _ = im_bgr.shape
        # cv2.putText(im_bgr, fps_str, org=(fw - 110, fh - 20),
        #                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
        #                            color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(im_bgr, fps_str, org=(fw - 111, fh - 21),
        #                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.79,
        #                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        last_frame_time = time.time()
        cv2.imshow("camera", im_bgr)
        cnt += 1
    cap.release()
    cv2.destroyAllWindows()
