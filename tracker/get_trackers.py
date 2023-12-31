from tracker.strongsort.utils.parser import get_config

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    
    cfg = get_config()
    cfg.merge_from_file(tracker_config)

    # if tracker_type == 'bytetrack':
    #     from tracker.bytetrack.byte_tracker import BYTETracker

    #     bytetrack = BYTETracker
    
    if tracker_type == 'strongsort':
        from tracker.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,

        )
        return strongsort
    
    elif tracker_type == 'bytetrack':
        from tracker.bytetrack.byte_tracker import BYTETracker
        bytetrack = BYTETracker()

    elif tracker_type == 'botsort':
        from tracker.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.botsort.track_high_thresh,
            new_track_thresh=cfg.botsort.new_track_thresh,
            track_buffer =cfg.botsort.track_buffer,
            match_thresh=cfg.botsort.match_thresh,
            proximity_thresh=cfg.botsort.proximity_thresh,
            appearance_thresh=cfg.botsort.appearance_thresh,
            cmc_method =cfg.botsort.cmc_method,
            frame_rate=cfg.botsort.frame_rate,
            lambda_=cfg.botsort.lambda_
        )
        return botsort
    else:
        print('No such tracker')
        exit()