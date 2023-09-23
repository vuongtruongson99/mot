import cv2
import math
import numpy as np
from collections import deque
from scipy.spatial import distance
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from . import matching
from .gmc import GMC
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from . import kalman_filter
from .detection import Detection


class Track(BaseTrack):
    '''
    Class này biển diễn từng track riêng biệt trong toàn bộ video. Lưu các thông tin 
    về các bounding box detect được mà ứng với từng track.

    Parameters
    ----------
    mean : ndarray (8 dimensional vector)
        Mean vector của initial state (Dùng trong KF, lưu mean của (x, y, w, h, x', y', w', h'))
    covariance : ndarray (8 dimensional vector)
        Covariance vector của initial state (Dùng trong KF, covariance của (x, y, w, h, x', y', w', h'))
    track_id : int
        Track id riêng biệt của mỗi track
    n_init : int
        Số lượng bbox detect được
    bbox : ndarray
        Tọa độ bounding box (x, y, w, h) với (x, y) là center bbox, width w, height h


    '''
    shared_kalman = KalmanFilter()
    def __init__(self, detection, n_init=3, max_age=30):
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        

        self.smooth_feature = None
        self.features = deque([], maxlen=max_age)
        self.smooth_color_hist = None
        self.color_hists = deque([], maxlen=max_age)

        self._n_init = n_init
        self._max_age = max_age # chỉ dùng để đánh dấu là missed, có thể bỏ
        
        self.is_activated = False
        self.storage = []       # lưu các Detection trong Track
        self.delete = False     # xử lý trong postprocess
        
        self.alpha = 0.9        # smooth feature

        # self.update_features(feat)
        # self.update_color_hists(color_hist)
        # Thay 2 hàm này bằng hàm update_storage()
        self.update_storage(detection)

        self.det_tlwh = np.asarray(detection.tlwh, dtype=np.float)

    def update_storage(self, detection):
        feat = detection.feature / np.linalg.norm(detection.feature)
        # self.curr_feat = feat
        if self.smooth_feature is None:
            self.smooth_feature = feat
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feature /= np.linalg.norm(self.smooth_feature)

        color_hist = detection.color_hist / np.linalg.norm(detection.color_hist)
        # self.curr_color_hist = color_hist
        if self.smooth_color_hist is None:
            self.smooth_color_hist = color_hist
        else:
            self.smooth_color_hist = self.alpha * self.smooth_color_hist + (1 - self.alpha) * color_hist
        self.color_hists.append(color_hist)
        self.smooth_color_hist /= np.linalg.norm(self.smooth_color_hist)

        self.storage.append(detection)
    
    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([t.mean.copy() for t in tracks])
            multi_covariance = np.asarray([t.covariance for t in tracks])
            for i, t in enumerate(tracks):
                if t.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov
                tracks[i].time_since_update += 1
                tracks[i].age += 1

    @staticmethod
    def multi_gmc(tracks, H=np.eye(2, 3)):
        if len(tracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in tracks])
            multi_covariance = np.asarray([st.covariance for st in tracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                tracks[i].mean = mean
                tracks[i].covariance = cov
    
    def activate(self, kalman_filter):
        ''' Bắt đầu 1 track mới '''
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self.storage[-1].tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
    
    def re_activate(self, detection, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(detection.tlwh), detection.confidence)
        
        ############ Có thể loại bỏ để xem hiệu suất thế nào #############
        # if new_track.curr_feat is not None:
        #     self.update_features(new_track.curr_feat)
        # if new_track.curr_color_hist is not None:
        #     self.update_color_hists(new_track.curr_color_hist)
        # if new_track.tlwh is not None:
        #     self.update_tlwhs(new_track.tlwh)
        # self.frame_idx.append(frame_id)
        self.update_storage(detection)
        ##################################################################

        self.hits += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.det_tlwh = detection.tlwh

        if new_id:
            self.track_id = self.next_id()

    def update(self, detection):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(detection.tlwh), detection.confidence)

        # feature = detection.feature / np.linalg.norm(detection.feature)
        # if self.smooth_feature is None:
        #     self.smooth_feature = feature
        # else:
        #     self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * feature
        # self.smooth_feature /= np.linalg.norm(self.smooth_feature)
        # self.features.append(feature)

        # if self.smooth_color_hist is None:
        #     self.smooth_color_hist = detection.color_hist
        # else:
        #     self.smooth_color_hist = self.alpha * self.smooth_color_hist + (1 - self.alpha) * detection.color_hist
        # self.smooth_color_hist /= np.linalg.norm(self.smooth_color_hist)
        # self.color_hists.append(self.smooth_color_hist)
        self.update_storage(detection)
        
        self.hits += 1
        self.time_since_update = 0
        self.is_activated = True
        self.det_tlwh = detection.tlwh
    
        # if self.state == TrackState.New and self.hits >= self._n_init:
        #     self.state = TrackState.Tracked
        self.state = TrackState.Tracked     # Thử thay lại bằng 2 dòng trên xem sao?

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'Track_id:{}_(s:{}-e:{})'.format(self.track_id, self.storage[0].frame_idx, self.storage[-1].frame_idx)

class Tracker:
    '''
    Theo dõi toàn bộ track trong video
    '''

    def __init__(self, n_init=3, cam_name='c005', image_filenames=None):
        self.image_filenames = image_filenames
        self.tracks_all = []
        self.tracks = []
        self.tracked_tracks = []  # type: list[Track]
        self.lost_tracks = []  # type: list[Track]
        self.removed_tracks = []  # type: list[Track]

        # BoT params
        BaseTrack.clear_count()
        
        # Tracking module
        self.track_high_thresh = 0.5
        self.new_track_thresh = 0.4
        self.match_thresh = 0.8

        self.buffer_size = 30
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.unconfirmed_thr = 0.7

        # ReID module
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        
        self.gmc = GMC()
        ####################################
        self.n_init = n_init
        self._next_id = 1
        
        self.cam_name = cam_name

        # Tính embedding distance 
        self.budget = 1     # Nếu có EMA, còn không sẽ là 100

    def update(self, detects, img):
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        ''' Find high threshold detections '''
        if len(detects) > 0:
            high_score_det_ids = [i for i, d in enumerate(detects) if d.confidence >= self.track_high_thresh]
            detections = [detects[high_id] for high_id in high_score_det_ids]
        else:
            detections = []

        ''' Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_tracks = []  # type: list[Track]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        track_pool = joint_stracks(tracked_tracks, self.lost_tracks)
        # print("strack_pool", strack_pool, len(strack_pool))

        # dự đoán vị trí bbox tiếp theo bằng KF
        Track.multi_predict(track_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, detects)
        Track.multi_gmc(track_pool, warp)
        Track.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes: IoU information + appearance information
        ious_dists = matching.iou_distance(track_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # fuse IoU score and detection score for association
        ious_dists = matching.fuse_score(ious_dists, detections)

        ##################### Chỉnh lại các distance cần thiết #####################
        emb_dists = matching.embedding_distance(track_pool, detections) / 2.0
        ############################################################################
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, unmatched_track, unmatched_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for idx_tracked, idx_det in matches:
            track = track_pool[idx_tracked]
            det = detections[idx_det]

            if track.state == TrackState.Tracked:
                track.update(det)
                activated_tracks.append(track)
            else:
                track.re_activate(det, new_id=False)
                refind_tracks.append(track)

        ''' Step 3: Second association, with low score detection boxes (Only IoU)'''
        if len(detects) > 0:
            low_score_det_ids = [i for i, d in enumerate(detects) if d.confidence < self.track_high_thresh]
            detections_second = [detects[low_id] for low_id in low_score_det_ids]
        else:
            detections_second = []

        # Lấy ra các miss track để cố vớt lại, sử dụng IoU (vì thường miss là do bị che khuất, không thể sử dụng feature vector)
        iou_thresh = 0.5    # Top 1 để là 0.4
        r_tracked_tracks = [track_pool[i] for i in unmatched_track if track_pool[i].state == TrackState.Tracked]
        iou_dists = matching.iou_distance(r_tracked_tracks, detections_second)
        matches_second, unmatched_track_second, unmatched_detection_second = matching.linear_assignment(
            iou_dists, thresh=iou_thresh
        )

        for idx_tracked, idx_det in matches_second:
            track = r_tracked_tracks[idx_tracked]
            det = detections_second[idx_det]
            if track.state == TrackState.Tracked:
                track.update(det)
                activated_tracks.append(track)
            else:
                track.re_activate(det, new_id=False)
                refind_tracks.append(track)

        for it in unmatched_track_second:
            track = r_tracked_tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)
        
        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in unmatched_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches_third, unmatched_unconfirmed, unmatched_detection_third = matching.linear_assignment(
            dists, thresh=0.7
        )

        for idx_tracked, idx_det in matches_third:
            unconfirmed[idx_tracked].update(detections[idx_det])
            activated_tracks.append(unconfirmed[idx_tracked])
        
        # Không thể cứu được các track
        for it in unmatched_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)
        
        """ Step 5: Init new stracks"""
        for inew in unmatched_detection_third:
            track = Track(detections[inew], self.n_init, self.max_time_lost)

            if detections[inew].confidence < self.new_track_thresh:
                continue
            
            track.activate(self.kalman_filter)
            self.tracks.append(track)
            activated_tracks.append(track)
        
        """ Step 6: Update state"""
        for track in self.lost_tracks:
            if track.time_since_update > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        """ Merge """
        self.tracked_tracks    = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]  # lấy những tracklet với TrackState = Tracked
        self.tracked_tracks    = joint_stracks(self.tracked_tracks, activated_tracks)               # Gộp những tracklet mới trong activated_tracks
        self.tracked_tracks    = joint_stracks(self.tracked_tracks, refind_tracks)                  # Gộp những tracklet được track lại trong refind_tracks
        self.lost_tracks       = sub_tracks(self.lost_tracks, self.tracked_tracks)                  # Loại bỏ những track đang được track ra khỏi self.lost_track
        self.lost_tracks.extend(lost_tracks)                                                        # Thêm những lost_tracks mới vào self.lost_track
        self.lost_tracks       = sub_tracks(self.lost_tracks, self.removed_tracks)                  # Xóa những track bị đánh dấu là removed
        self.removed_tracks.extend(removed_tracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(self.tracked_tracks, self.lost_tracks)
        
        output_stracks = [track for track in self.tracked_tracks]

        # self.tracks_all.extend([track for track in self.tracked_tracks])
        self.tracks_all = joint_stracks(self.tracks_all, self.tracked_tracks)
        # for t in self.tracks_all:
        #     print(t)
        # print("------------------------------------------")
        return output_stracks

    def postprocess(self):
        self.tracks_all = joint_stracks(self.tracked_tracks, self.lost_tracks)
        # self.tracks_all = joint_stracks(self.tracks_all, self.removed_tracks)

        for track in self.tracks_all:
            track.storage.sort(key=lambda d: d.frame_idx)

        self._update_tracklet_info()
        self.tracks_all = [track for track in self.tracks_all if track.state != TrackState.Removed]
        # self._delete_tracklets()                                                                # Lọc ra những cái track nào có số detections bé hơn const, và xét độ lớn euclid và bắt từng camera 1, còn lại delete
        # self.tracks_all = [t for t in self.tracks_all if track.state != TrackState.Removed]     # TrackState != Remove (number 4)
        # self._merge_similar()                                                                   # Merge track có detection giống nhau và start, end ở giữa và xoá track vô lí
        # self.tracks_all = [t for t in self.tracks_all if track.state != TrackState.Removed]     # TrackState != Remove (number 4)
        self._update_tracklet_info()
        self._linear_interpolation()
        # self._gaussian_smooth()
        self._update_tracklet_info()
        # self._merge_overlap()
        # self.tracks_all = [track for track in self.tracks_all if track.state != TrackState.Removed]
            

    def _update_tracklet_info(self):
        for track in self.tracks_all:
            track.start_frame = track.storage[0].frame_idx
            track.end_frame = track.storage[-1].frame_idx
    
    def _linear_interpolation(self, interval=20):
        for track in self.tracks_all:
            frame_pre = track.storage[0].frame_idx
            tlwh_pre = track.storage[0].tlwh
            interpolation_storage = []
            for det in track.storage:
                frame_curr = det.frame_idx
                tlwh_curr = det.tlwh
                if frame_pre + 1 < frame_curr < frame_pre + interval:
                    for i, frame in enumerate(range(frame_pre + 1, frame_curr), start=1):
                        step = (tlwh_curr - tlwh_pre) / (frame_curr - frame_pre) * i
                        tlwh_new = tlwh_pre + step
                        img = cv2.imread(self.image_filenames[frame])
                        color_hist = []
                        H, W, _ = img.shape
                        x1 = int(tlwh_new[0])
                        y1 = int(tlwh_new[1])
                        w = int(tlwh_new[2])
                        h = int(tlwh_new[3])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W - 1, x2)
                        y2 = min(H - 1, y2)
                        for i in range(3):
                            color_hist += cv2.calcHist([img[y1:y2, x1:x2]], [i], None, [8], [0.0, 255.0]).T.tolist()[0]
                        color_hist = np.array(color_hist)
                        norm = np.linalg.norm(color_hist)
                        color_hist /= norm
                        interpolation_storage.append(Detection(tlwh_new, 0.3, None, frame, color_hist))
                frame_pre = frame_curr
                tlwh_pre = tlwh_curr
            track.storage += interpolation_storage
            track.storage.sort(key=lambda d : d.frame_idx)

    def _gaussian_smooth(self, tau=10):
        for track in self.tracks_all:
            len_scale = np.clip(tau * np.log(tau ** 3 / len(track.storage)), tau ** -1, tau ** 2)
            gpr = GPR(RBF(len_scale, 'fixed'))
            ftlwh = []
            for det in track.storage:
                ftlwh.append([det.frame_idx, det.tlwh[0], det.tlwh[1], det.tlwh[2], det.tlwh[3]])
            ftlwh = np.array(ftlwh)
            print(ftlwh.shape)
            t = ftlwh[:, 0].reshape(-1, 1)
            x = ftlwh[:, 1].reshape(-1, 1)
            y = ftlwh[:, 2].reshape(-1, 1)
            w = ftlwh[:, 3].reshape(-1, 1)
            h = ftlwh[:, 4].reshape(-1, 1)
            gpr.fit(t, x)
            xx = gpr.predict(t)[:, 0]
            gpr.fit(t, y)
            yy = gpr.predict(t)[:, 0]
            gpr.fit(t, w)
            ww = gpr.predict(t)[:, 0]
            gpr.fit(t, h)
            hh = gpr.predict(t)[:, 0]
            for i in range(len(track.storage)):
                track.storage[i].tlwh = np.array([xx[i], yy[i], ww[i], hh[i]])
    
    def _merge_overlap(self):
        for t1 in self.tracks_all:
            t1_len = len(t1.storage)
            if t1_len < 3:
                continue
            t1_det1 = t1.storage[0]
            t1_det2 = t1.storage[1]
            t1_det3 = t1.storage[2]
            t1_start_frame = t1.storage[0].frame_idx
            t1_end_frame = t1.storage[-1].frame_idx

            for t2 in self.tracks_all:
                t2_len = len(t2.storage)
                t2_start_frame = t2.storage[0].frame_idx
                t2_end_frame = t2.storage[-1].frame_idx
                if t1 == t2 or t2_len < 3:
                    continue
                if t1_start_frame < t2_start_frame or t1_start_frame > t2_end_frame - 2:
                    continue
                if t1_end_frame > t2_end_frame:
                    t1_det4 = None
                    for det in t1.storage:
                        if det.frame_idx == t2_end_frame:
                            t1_det4 = det
                    t2_det4 = t2.storage[-1]
                    if t1_det4 is None or self._det_iou(t1_det4, t2_det4) <= 0.8:
                        continue
                else:
                    t1_det4 = t1.storage[-1]
                    t2_det4 = None
                    for det in t2.storage:
                        if det.frame_idx == t1_end_frame:
                            t2_det4 = det
                    if t2_det4 is None or self._det_iou(t1_det4, t2_det4) <= 0.8:
                        continue
                k = t1_start_frame - t2_start_frame
                t2_det1 = None
                for det in t2.storage:
                    if det.frame_idx == t1_start_frame:
                        t2_det1 = det
                if t2_det1 is not None and self._det_iou(t1_det1, t2_det1) > 0.8:
                    t1.delete = True
                    for p in range(t1_len):
                        if t1.storage[p].frame_idx > t2.storage[-1].frame_idx:
                            t2.storage += t1.storage[p:]
                            break
                    break

    
    def _det_iou(self, det1, det2):
        ltx1 = det1.tlwh[0]
        lty1 = det1.tlwh[1]
        rdx1 = det1.tlwh[0] + det1.tlwh[2]
        rdy1 = det1.tlwh[1] + det1.tlwh[3]
        ltx2 = det2.tlwh[0]
        lty2 = det2.tlwh[1]
        rdx2 = det2.tlwh[0] + det2.tlwh[2]
        rdy2 = det2.tlwh[1] + det2.tlwh[3]

        W = min(rdx1, rdx2) - max(ltx1, ltx2)
        H = min(rdy1, rdy2) - max(lty1, lty2)
        cross = W * H

        if(W <= 0 or H <= 0):
            return 0

        SA = (rdx1 - ltx1) * (rdy1 - lty1)
        SB = (rdx2 - ltx2) * (rdy2 - lty2)
        if min(SA, SB) <= 0:
            return 0
        return cross / min(SA, SB)

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_tracks(tlista, tlistb):
    tracks = {}
    for t in tlista:
        tracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())

######### Loại bỏ các track mà có độ IoU distance < 0.15 #########
def remove_duplicate_tracks(tracksa, tracksb):
    pdist = matching.iou_distance(tracksa, tracksb)
    pairs = np.where(pdist < 0.15)

    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = tracksa[p].storage[-1].frame_idx - tracksa[p].storage[0].frame_idx
        timeq = tracksb[q].storage[-1].frame_idx - tracksb[q].storage[0].frame_idx
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    
    resa = [t for i, t in enumerate(tracksa) if not i in dupa]
    resb = [t for i, t in enumerate(tracksb) if not i in dupb]

    return resa, resb