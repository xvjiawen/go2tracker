from typing import List, Tuple

import numpy as np
import supervision as sv
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.basetrack import BaseTrack, TrackState
from supervision.tracker.byte_tracker.core import STrack, detections2boxes
from kalman_filter import KalmanFilter
from copy import deepcopy
from scipy.spatial.distance import cdist

class STrackCustom(STrack):
    shared_kalman = None
    def __init__(self, tlwh, score, class_ids, minimum_consecutive_frames,
                 det_bbox=None):
        super().__init__(
            tlwh=tlwh, score=score, class_ids=class_ids,
            minimum_consecutive_frames=minimum_consecutive_frames
        )
        self.det_bbox = det_bbox

class ByteTrack:
    """
    Initialize the ByteTrack object.

    <video controls>
        <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
    </video>

    Parameters:
        track_activation_threshold (float, optional): Detection confidence threshold
            for track activation. Increasing track_activation_threshold improves accuracy
            and stability but might miss true detections. Decreasing it increases
            completeness but risks introducing noise and instability.
        lost_track_buffer (int, optional): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        minimum_matching_threshold (float, optional): Threshold for matching tracks with detections.
            Increasing minimum_matching_threshold improves accuracy but risks fragmentation.
            Decreasing it improves completeness but risks false positives and drift.
        frame_rate (int, optional): The frame rate of the video.
        minimum_consecutive_frames (int, optional): Number of consecutive frames that an object must
            be tracked before it is considered a 'valid' track.
            Increasing minimum_consecutive_frames prevents the creation of accidental tracks from
            false detection or double detection, but risks missing shorter tracks.
    """  # noqa: E501 // docs
    def __init__(
        self,
        # track_activation_threshold: float = 0.25,
        track_thresh: float = 0.3,
        init_thresh: float = 0.6,
        lost_track_buffer: int = 30,
        first_matching_threshold: float = 0.8,
        second_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
        u_track_with_all_u_dets=False,  # 未确认轨迹与所有剩余检测框再次进行匹配
        std_weight_position=1.0 / 10,
        std_weight_velocity=1.0 / 200,
        project_rate=1.0,
        use_width_match_adaptive=False, # 是否采用自适应宽度匹配阈值
        use_width_fuse=False,       # 是否采用宽度筛选策略：筛除面积和宽高比不符合要求的
        solid_width_threshold=60,   # 针对宽度匹配选择固定宽度阈值
        adaptive_width_ratio=1.3,   # 动态宽度比例，阈值为检测框宽度*adaptive_width_ratio
        width_fuse_area_ratio=2.0,  # 宽度匹配筛选掉之前匹配目标面积/当前目标面积>width_fuse_area_ratio的目标
        width_fuse_aspect_ratio=1.5,# 宽度匹配筛选掉之前匹配目标宽高比/当前目标宽高比>width_fuse_aspect_ratio的目标
        one_track_width_ratio=1.8,   # 针对单一目标的动态宽度比例，阈值为检测框宽度*one_track_width_ratio
        one_track_widthfuse_ratio=1.5 # 针对单一目标筛选面积和宽高比的阈值在之前基础上乘以该系数
    ):
        # self.track_activation_threshold = track_activation_threshold
        self.track_thresh = track_thresh    # 确定哪些检测结果可以被直接作为高置信度的跟踪结果
        self.init_thresh = init_thresh # 确定哪些检测结果可以被直接new
        self.first_matching_threshold = first_matching_threshold
        self.second_matching_threshold = second_matching_threshold
        self.u_track_with_all_u_dets = u_track_with_all_u_dets
        self.use_width_match_adaptive = use_width_match_adaptive
        self.use_width_fuse = use_width_fuse
        self.solid_width_threshold = solid_width_threshold
        self.adaptive_width_ratio = adaptive_width_ratio
        self.width_fuse_area_ratio = width_fuse_area_ratio
        self.width_fuse_aspect_ratio = width_fuse_aspect_ratio
        self.one_track_width_ratio = one_track_width_ratio
        self.one_track_widthfuse_ratio = one_track_widthfuse_ratio

        self.frame_id = 0
        # self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = lost_track_buffer
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.kalman_filter = KalmanFilter(
            std_weight_position=std_weight_position,
            std_weight_velocity=std_weight_velocity,
            project_rate=project_rate)
        STrack.shared_kalman = self.kalman_filter
        self.tracked_tracks: List[STrackCustom] = []
        self.lost_tracks: List[STrackCustom] = []
        self.removed_tracks: List[STrackCustom] = []

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections (Detections): The detections to pass through the tracker.

        """

        tensors = detections2boxes(detections=detections)
        tracks = self.update_with_tensors(tensors=tensors)

        if len(tracks) > 0:
            track_ids = []
            det_boxs = []
            track_boxs = []
            scores = []
            class_ids = []
            for track in tracks:
                track_ids.append(track.external_track_id)
                if track.state == TrackState.Lost:
                    det_boxs.append(track.tlbr)
                else:
                    det_boxs.append(track.det_bbox)
                # det_boxs.append(track.det_bbox)
                track_boxs.append(track.tlbr)
                scores.append(track.score)
                class_ids.append(track.class_ids)

            pred_instances = sv.Detections(
            xyxy=np.array(det_boxs),
            confidence=np.array(scores),
            class_id=np.array(class_ids, dtype=int),
            tracker_id=np.array(track_ids),
            data={"track_boxs": np.array(track_boxs)}
            )
        else:
            pred_instances = Detections.empty()
            pred_instances.tracker_id = np.array([], dtype=int)


        return pred_instances


    def reset(self):
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.tracked_tracks: List[STrackCustom] = []
        self.lost_tracks: List[STrackCustom] = []
        self.removed_tracks: List[STrackCustom] = []
        BaseTrack.reset_counter()
        STrackCustom.reset_external_counter()

    def width_match(self, r_tracked_stracks, r_detections, activated_starcks,
                    refind_stracks):

        # # Only consider tracks in Tracked state for center matching
        # r_tracked_stracks = [strack_pool[i] for i in u_track]
        #
        # # Only consider high-confidence detections for center matching
        # r_detections = [detections[i] for i in u_detection]


        # Calculate center points for tracks
        track_centers = np.array([
            [(track.tlbr[0] + track.tlbr[2]) / 2,
              (track.tlbr[1] + track.tlbr[3]) / 2]
             for track in r_tracked_stracks
        ])

        # Calculate center points for detections
        det_centers = np.array([
             [(det.tlbr[0] + det.tlbr[2]) / 2,
              (det.tlbr[1] + det.tlbr[3]) / 2]
             for det in r_detections
        ])

        # Calculate Euclidean distance between centers
        center_dists = cdist(track_centers, det_centers,
                              metric='euclidean')

        # Size difference check
        valid_dists = center_dists.copy()
        if self.use_width_fuse:
            valid_dists = self.width_fuse(valid_dists, r_tracked_stracks,
                                          r_detections)

        if self.use_width_match_adaptive:

            # Create adaptive thresholds based on bounding box widths
            adaptive_thresholds = np.zeros(
                (len(r_tracked_stracks), len(r_detections)))

            for i, track in enumerate(r_tracked_stracks):
                for j, det in enumerate(r_detections):
                    det_width = det.tlbr[2] - det.tlbr[0]
                    adaptive_thresholds[i, j] = self.adaptive_width_ratio * det_width

            # Apply adaptive thresholds
            for i in range(len(r_tracked_stracks)):
                for j in range(len(r_detections)):
                    if center_dists[i, j] > adaptive_thresholds[i, j]:
                        valid_dists[i, j] = 19200

            # Check if there are any valid matches (cost matrix is not all inf)
            # if not np.all(np.isinf(valid_dists)):
                # Perform linear assignment based on valid distances

            center_matches, center_u_track, center_u_detection = \
                matching.linear_assignment(
                valid_dists, thresh=1920
            )

            # Process center distance matches
            for itracked, idet in center_matches:
                track = r_tracked_stracks[itracked]
                det = r_detections[idet]
                # if valid_dists[itracked, idet] <= adaptive_thresholds[
                #     itracked, idet]:
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    track.det_bbox = det.tlbr
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    track.det_bbox = det.tlbr
                    refind_stracks.append(track)

        else:
            # Set a threshold for center distance matching
            # center_dist_thresh = 60.0  # Adjust based on your scenario

            # Create a binary mask for distances below threshold
            # valid_dists = center_dists.copy()
            valid_dists[center_dists > self.solid_width_threshold] = np.inf

            # Perform linear assignment based on center distances
            center_matches, center_u_track, center_u_detection = matching.linear_assignment(
                valid_dists, thresh=self.solid_width_threshold
            )

            # Process center distance matches
            for itracked, idet in center_matches:
                track = r_tracked_stracks[itracked]
                det = r_detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    track.det_bbox = det.tlbr
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    track.det_bbox = det.tlbr
                    refind_stracks.append(track)

        return center_u_track, center_u_detection

    def width_fuse(self, valid_dists, r_tracked_stracks, r_detections):

        for i, track in enumerate(r_tracked_stracks):
            for j, det in enumerate(r_detections):
                # Calculate track and detection sizes
                prev_width = track.det_bbox[2] - track.det_bbox[0]
                prev_height = track.det_bbox[3] - track.det_bbox[1]
                prev_area = prev_width * prev_height
                prev_ratio = prev_width / (prev_height + 1e-6)

                det_width = det.tlbr[2] - det.tlbr[0]
                det_height = det.tlbr[3] - det.tlbr[1]
                det_area = det_width * det_height
                det_ratio = det_width / (det_height + 1e-6)

                # Compute size differences
                area_change = max(prev_area, det_area) / (
                        min(prev_area, det_area) + 1e-6)
                ratio_change = max(prev_ratio, det_ratio) / (
                        min(prev_ratio, det_ratio) + 1e-6)

                # Size thresholds
                # area_threshold = 2.0  # Max 2x area difference
                # ratio_threshold = 1.5  # Max 1.5x aspect ratio difference

                # If size difference is too large, invalidate the match
                if area_change > self.width_fuse_area_ratio or \
                        ratio_change > self.width_fuse_aspect_ratio:
                    valid_dists[i, j] = 19200

        return valid_dists

    def update_with_tensors(self, tensors: np.ndarray) -> List[STrackCustom]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        class_ids = tensors[:, 5]
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores >= self.track_thresh
        # inds_low = scores > 0.1
        # inds_high = scores < self.track_thresh
        inds_second = scores < self.track_thresh
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrackCustom(STrackCustom.tlbr_to_tlwh(tlbr), s, c, self.minimum_consecutive_frames)
                for (tlbr, s, c) in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrackCustom]

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrackCustom.multi_predict(strack_pool)

        # First matching stage: pure IoU-based matching
        iou_dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            iou_dists, thresh=self.first_matching_threshold
        )

        # Process IoU matches
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                track.det_bbox = det.tlbr
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                track.det_bbox = det.tlbr
                refind_stracks.append(track)

        # Second matching stage: center distance-based matching for unmatched tracks/detections
        if len(u_track) > 0 and len(u_detection) > 0:
            # Only consider tracks in Tracked state for center matching
            r_tracked_stracks = [strack_pool[i] for i in u_track]

            # Only consider high-confidence detections for center matching
            r_detections = [detections[i] for i in u_detection]

            # if len(r_tracked_stracks) > 0 and len(r_detections) > 0:
            center_u_track, center_u_detection = self.width_match(
                r_tracked_stracks, r_detections, activated_starcks,
                refind_stracks)

            remaining_u_track = [u_track[i] for i in center_u_track]
            remaining_u_detection = [u_detection[i] for i in
                                     center_u_detection]
            u_track = remaining_u_track
            u_detection = remaining_u_detection

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrackCustom(STrackCustom.tlbr_to_tlwh(tlbr), s, c, self.minimum_consecutive_frames)
                for (tlbr, s, c) in zip(dets_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []
        # 从未匹配的轨迹中筛选出仍在跟踪状态的轨迹
        r_tracked_stracks = [strack_pool[i] for i in u_track]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=self.second_matching_threshold
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                track.det_bbox = det.tlbr
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                track.det_bbox = det.tlbr
                refind_stracks.append(track)

        # 步骤3.2: 对未匹配的轨迹和低置信度检测框使用中心点距离进行二次匹配
        if len(u_track) > 0 and len(u_detection_second) > 0:
            # 筛选未匹配的轨迹
            r_tracked_stracks_second = [
                r_tracked_stracks[i]
                for i in u_track
            ]

            # 筛选未匹配的低置信度检测框
            r_detections_second = [detections_second[i] for i in
                                   u_detection_second]

            center_u_track, center_u_detection_second = self.width_match(
                r_tracked_stracks_second, r_detections_second, activated_starcks,
                refind_stracks)

            remaining_u_track = [u_track[i] for i in center_u_track]
            remaining_u_detection_second = [u_detection_second[i] for i in
                                     center_u_detection_second]
            u_track = remaining_u_track
            u_detection_second = remaining_u_detection_second

        # 应用单Track单检测框直接匹配策略:
        # 只有一个检测框，只有一个跟踪对象且该对象为正在跟踪的状态
        if len(strack_pool) == 1 and len(bboxes) == 1 and strack_pool[0].state == TrackState.Tracked:
            track = strack_pool[0]

            all_unmatched_dets = []
            for i in u_detection:
                all_unmatched_dets.append((i, detections[i]))

            # 收集所有未匹配的低置信度检测框
            for i in u_detection_second:
                all_unmatched_dets.append(
                    (i + len(detections), detections_second[i]))

            assert len(all_unmatched_dets) <= 1
            if len(all_unmatched_dets) == 1:
                idx, det = all_unmatched_dets[0]

                # 计算中心点距离
                track_center = [(track.tlbr[0] + track.tlbr[2]) / 2,
                                (track.tlbr[1] + track.tlbr[3]) / 2]
                det_center = [(det.tlbr[0] + det.tlbr[2]) / 2,
                              (det.tlbr[1] + det.tlbr[3]) / 2]
                center_dist = np.sqrt(((track_center[0] - det_center[0]) ** 2) +
                                      ((track_center[1] - det_center[1]) ** 2))

                prev_width = track.det_bbox[2] - track.det_bbox[0]
                prev_height = track.det_bbox[3] - track.det_bbox[1]
                prev_area = prev_width * prev_height
                prev_ratio = prev_width / (prev_height + 1e-6)

                det_width = det.tlbr[2] - det.tlbr[0]
                det_height = det.tlbr[3] - det.tlbr[1]
                det_area = det_width * det_height
                det_ratio = det_width / (det_height + 1e-6)

                # Compute size differences
                area_change = max(prev_area, det_area) / (
                        min(prev_area, det_area) + 1e-6)
                ratio_change = max(prev_ratio, det_ratio) / (
                        min(prev_ratio, det_ratio) + 1e-6)

                # 使用较大的阈值进行匹配
                if (center_dist <= det_width * self.one_track_width_ratio
                        and area_change < self.width_fuse_area_ratio*self.one_track_widthfuse_ratio and
                        ratio_change < self.width_fuse_aspect_ratio*self.one_track_widthfuse_ratio):  # 更宽松的阈值
                # if center_dist <= 90:  # 更宽松的阈值
                    track.update(det, self.frame_id)
                    track.det_bbox = det.tlbr
                    activated_starcks.append(track)

                    # 从未匹配轨迹列表中移除
                    u_track = [i for i in u_track if strack_pool[i] != track]

                    # 从未匹配检测框列表中移除
                    if idx < len(detections):
                        u_detection = [i for i in u_detection if i != idx]
                    else:
                        u_detection_second = [i for i in u_detection_second if
                                              i != (idx - len(detections))]

        # 标记未匹配的轨迹为丢失状态
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                track.score = 0
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""

        if self.u_track_with_all_u_dets:
            # 计算未确认轨迹与所有剩余检测框的距离
            detections = [detections[i] for i in u_detection] + \
                        [detections_second[i] for i in u_detection_second]
        else:
            # 计算未确认轨迹与剩余检测框的距离
            detections = [detections[i] for i in u_detection]

        dists = matching.iou_distance(unconfirmed, detections)
        # dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.second_matching_threshold
        )
        # 处理匹配成功的未确认轨迹
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            unconfirmed[itracked].det_bbox = detections[idet].tlbr
            activated_starcks.append(unconfirmed[itracked])

        # 标记未匹配的未确认轨迹为移除状态
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 从未匹配的检测框中创建新轨迹（仅当置信度高于det_thresh时）
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.det_bbox = deepcopy(track.tlbr)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.removed_tracks = removed_stracks
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        # self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        total_tracks = self.tracked_tracks + self.lost_tracks
        output_stracks = [track for track in total_tracks
                          if track.is_activated and track.state != TrackState.Removed]
        # output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def joint_tracks(
    track_list_a: List[STrackCustom], track_list_b: List[STrackCustom]
) -> List[STrackCustom]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate internal_track_id values.

    Parameters:
        track_list_a: First list of tracks (with internal_track_id attribute).
        track_list_b: Second list of tracks (with internal_track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate internal_track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: List, track_list_b: List) -> List[int]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same internal_track_id with tracks in track_list_b.

    Parameters:
        track_list_a: List of tracks (with internal_track_id attribute).
        track_list_b: List of tracks (with internal_track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(tracks_a: List, tracks_b: List) -> Tuple[List, List]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
