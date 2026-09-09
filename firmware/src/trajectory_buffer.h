#pragma once
#include <stdint.h>

#define MAX_WAYPOINTS 2000 // 2000 samples @ 50 Hz = 40 seconds of continuous recording

struct TrajectoryPoint {
    float theta[3];
    float gripper;
};

class TrajectoryBuffer {
public:
    TrajectoryBuffer() : _count(0), _playIndex(0) {}

    void clear() {
        _count = 0;
        _playIndex = 0;
    }

    bool addPoint(float th1, float th2, float th3, float gripper) {
        if (_count >= MAX_WAYPOINTS) return false;
        _points[_count].theta[0] = th1;
        _points[_count].theta[1] = th2;
        _points[_count].theta[2] = th3;
        _points[_count].gripper = gripper;
        _count++;
        return true;
    }

    size_t count() const { return _count; }

    bool getNextPoint(TrajectoryPoint &out) {
        if (_count == 0) return false;
        out = _points[_playIndex];
        _playIndex = (_playIndex + 1) % _count; // Loop playback
        return true;
    }

    void resetPlayback() { _playIndex = 0; }

private:
    TrajectoryPoint _points[MAX_WAYPOINTS];
    size_t _count;
    size_t _playIndex;
};
