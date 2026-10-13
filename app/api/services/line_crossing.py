from math import hypot


class LineCrossingTracker:
    """Count bbox-center crossings of a finite segment with a pixel dead band."""

    def __init__(self, line: dict, margin: float = 3.0, max_gap_ms: float = 2000):
        self.start = (line["x1"], line["y1"])
        self.end = (line["x2"], line["y2"])
        self.length = hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        if not self.length:
            raise ValueError("Counting line must have two different endpoints")
        self.margin = margin
        self.max_gap_ms = max_gap_ms
        self.tracks = {}

    def side(self, point):
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        distance = (dx * (point[1] - self.start[1]) - dy * (point[0] - self.start[0])) / self.length
        return 1 if distance > self.margin else -1 if distance < -self.margin else 0

    def intersects(self, previous, current):
        # Solve the two segment parameters; crossing the infinite extension does not count.
        rx, ry = current[0] - previous[0], current[1] - previous[1]
        sx, sy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        denominator = rx * sy - ry * sx
        if abs(denominator) < 1e-9:
            return False
        qx, qy = self.start[0] - previous[0], self.start[1] - previous[1]
        t = (qx * sy - qy * sx) / denominator
        u = (qx * ry - qy * rx) / denominator
        return 0 <= t <= 1 and 0 <= u <= 1

    def update(self, detections, timestamp_ms, offset=(0, 0)):
        self.tracks = {key: value for key, value in self.tracks.items()
                       if timestamp_ms - value[2] <= self.max_gap_ms}
        events, seen = [], set()
        for detection in detections:
            track_id = detection.track_id
            if track_id is None or track_id in seen:
                continue
            seen.add(track_id)
            box = detection.bbox
            point = ((box[0] + box[2]) / 2 + offset[0], (box[1] + box[3]) / 2 + offset[1])
            side = self.side(point)
            previous = self.tracks.get(track_id)
            if side == 0:
                # Keep the stable side while this track remains visible inside the band.
                if previous:
                    self.tracks[track_id] = (previous[0], previous[1], timestamp_ms)
                continue
            if previous and previous[1] != side and self.intersects(previous[0], point):
                events.append({**detection.model_dump(),
                               "direction": "A_to_B" if previous[1] < side else "B_to_A"})
            self.tracks[track_id] = (point, side, timestamp_ms)
        return events
