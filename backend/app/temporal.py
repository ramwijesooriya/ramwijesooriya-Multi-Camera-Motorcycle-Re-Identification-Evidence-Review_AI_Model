"""Camera-relative time reasoning with explicit signed offset and uncertainty."""


def temporal_evidence(a, b, config):
    if not config.temporal_enabled:
        return {'status': 'missing', 'reason': 'Temporal checks disabled; timing cannot support acceptance.'}
    if config.camera_b_offset_seconds is None:
        return {'status': 'missing', 'reason': 'B-to-A clock/start-time offset is unknown.'}
    start_a, end_a = a['summary']['start_seconds'], a['summary']['end_seconds']
    start_b = b['summary']['start_seconds'] + config.camera_b_offset_seconds
    end_b = b['summary']['end_seconds'] + config.camera_b_offset_seconds
    travel = start_b - end_a if config.route == 'a_to_b' else start_a - end_b
    uncertainty = config.clock_uncertainty_seconds + config.boundary_uncertainty_seconds
    low, high = travel - uncertainty, travel + uncertainty
    minimum, maximum = config.min_travel_seconds, config.max_travel_seconds
    if high < minimum or low > maximum:
        status = 'contradictory'
        reason = 'Possible travel interval is outside the configured route limits.'
    elif low >= minimum and high <= maximum:
        status = 'supporting'
        reason = 'Entire possible travel interval is within configured route limits.'
    else:
        status = 'missing'
        reason = 'Travel interval only partially overlaps plausible limits; timing remains uncertain.'
    fallback = any(o.get('timestamp_source') == 'fps_fallback' for t in (a, b) for o in t['observations'])
    if fallback and status != 'missing':
        status, reason = 'missing', 'Estimated frame timestamps prevent a strong temporal conclusion.'
    return {'status': status, 'reason': reason, 'route': config.route,
            'nominal_travel_seconds': travel, 'possible_travel_seconds': [low, high],
            'allowed_travel_seconds': [minimum, maximum], 'camera_b_offset_seconds': config.camera_b_offset_seconds,
            'clock_uncertainty_seconds': config.clock_uncertainty_seconds,
            'boundary_uncertainty_seconds': config.boundary_uncertainty_seconds}
