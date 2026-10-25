#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace geode::prelude;

// AutoBot v0.1 - reactive autoplayer for classic mode, cube and ship only.
//
// Every game update the bot looks at the static hazards/solids just ahead of
// the player and decides to press or release the jump button. It does NOT read
// a macro. Limits (by design for v0.1): no orbs/pads/portals, no ball, ufo,
// wave, spider, dual mode, slopes as exact shapes, or moving objects.

namespace autobot {
    constexpr float kDefaultSpeed = 311.58f;  // units/second at 1x speed
    constexpr float kCubeApexTime = 0.195f;   // seconds from cube jump to apex
    constexpr float kClusterGap = 36.f;       // obstacles closer than this are one group
    constexpr int kCubeHoldFrames = 3;

    struct Obstacle {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
    };

    class Bot {
    public:
        static Bot& get() {
            static Bot instance;
            return instance;
        }

        void update(PlayLayer* layer, float dt) {
            if (!layer || !layer->m_player1) return;
            if (!Mod::get()->getSettingValue<bool>("enabled")) {
                if (m_pressed) setPressed(layer, false);
                m_hasLast = false;
                return;
            }
            if (dt <= 0.f) return;

            PlayerObject* player = layer->m_player1;
            ensureObstacles(layer);

            float px = player->getPositionX();
            float py = player->getPositionY();

            if (!m_hasLast || px < m_lastX - 40.f) {
                resetRun();
            } else {
                measure(px - m_lastX, py - m_lastY, dt);
            }
            m_lastX = px;
            m_lastY = py;
            m_hasLast = true;

            CCRect pr = player->getObjectRect();
            bool want = false;

            if (player->m_isShip) {
                logMode("ship");
                want = shipDecide(player, pr);
            } else if (player->m_isBird || player->m_isBall || player->m_isDart || player->m_isSpider) {
                logMode("unsupported");
                want = false;
            } else {
                logMode("cube");
                want = cubeDecide(pr);
            }
            setPressed(layer, want);
        }

    private:
        std::vector<Obstacle> m_obstacles;
        PlayLayer* m_cacheLayer = nullptr;
        std::size_t m_cacheCount = 0;

        bool m_hasLast = false;
        float m_lastX = 0.f;
        float m_lastY = 0.f;
        float m_speed = kDefaultSpeed;
        float m_vy = 0.f;
        float m_step = 1.f / 60.f;

        bool m_pressed = false;
        int m_holdFrames = 0;
        float m_handledMaxX = -1e9f;
        char const* m_lastMode = "";

        static bool debug() { return Mod::get()->getSettingValue<bool>("debug_log"); }

        void logMode(char const* mode) {
            if (mode != m_lastMode) {
                m_lastMode = mode;
                if (debug()) log::info("AutoBot: mode {}", mode);
            }
        }

        void resetRun() {
            m_pressed = false;  // the game releases input when the attempt restarts
            m_holdFrames = 0;
            m_handledMaxX = -1e9f;
            m_vy = 0.f;
        }

        static bool plausibleSpeed(float v) { return v > 150.f && v < 700.f; }

        // dt may be a frame or a physics step depending on the build, so the
        // time base is validated against the expected player speed.
        void measure(float dx, float dy, float dt) {
            float step = dt;
            float vx = dx / step;
            if (!plausibleSpeed(vx)) {
                step = 1.f / 60.f;
                vx = dx / step;
            }
            if (plausibleSpeed(vx)) {
                m_speed = m_speed * 0.8f + vx * 0.2f;
                m_step = step;
            }
            m_vy = dy / m_step;
        }

        void setPressed(PlayLayer* layer, bool want) {
            if (want == m_pressed) return;
            m_pressed = want;
            layer->handleButton(want, 1, true);
            if (debug()) log::info("AutoBot: {}", want ? "press" : "release");
        }

        void ensureObstacles(PlayLayer* layer) {
            std::size_t count = layer->m_objects ? layer->m_objects->count() : 0;
            if (m_cacheLayer == layer && m_cacheCount == count) return;
            m_cacheLayer = layer;
            m_cacheCount = count;
            m_obstacles.clear();
            if (!layer->m_objects) return;

            CCArrayExt<GameObject*> objects = layer->m_objects;
            for (auto* obj : objects) {
                if (!obj) continue;
                auto type = obj->m_objectType;
                if (type != GameObjectType::Solid && type != GameObjectType::Hazard &&
                    type != GameObjectType::Slope) {
                    continue;
                }
                CCRect const& r = obj->getObjectRect();
                Obstacle o{r.getMinX(), r.getMaxX(), r.getMinY(), r.getMaxY()};
                if (o.maxX - o.minX < 0.01f || o.maxY - o.minY < 0.01f) {
                    float x = obj->getPositionX();
                    float y = obj->getPositionY();
                    o = Obstacle{x - 15.f, x + 15.f, y - 15.f, y + 15.f};
                }
                m_obstacles.push_back(o);
            }
            std::sort(m_obstacles.begin(), m_obstacles.end(),
                      [](Obstacle const& a, Obstacle const& b) { return a.minX < b.minX; });
            if (debug()) log::info("AutoBot: indexed {} obstacles", m_obstacles.size());
        }

        std::vector<Obstacle>::const_iterator firstFrom(float x) const {
            return std::lower_bound(m_obstacles.begin(), m_obstacles.end(), x,
                                    [](Obstacle const& o, float value) { return o.minX < value; });
        }

        // ---- cube ---------------------------------------------------------
        bool cubeDecide(CCRect const& pr) {
            if (m_holdFrames > 0) {
                --m_holdFrames;
                return true;
            }

            float pcx = (pr.getMinX() + pr.getMaxX()) * 0.5f;
            float front = pr.getMaxX();

            bool haveCluster = false;
            float cMin = 0.f, cMax = 0.f;
            for (auto it = firstFrom(front - 8.f); it != m_obstacles.end(); ++it) {
                if (it->minX > front + 220.f) break;
                if (it->minX <= m_handledMaxX) continue;
                // Only obstacles in the player's current row matter.
                float overlap = std::min(it->maxY, pr.getMaxY()) - std::max(it->minY, pr.getMinY());
                if (overlap <= 4.f) continue;
                if (!haveCluster) {
                    haveCluster = true;
                    cMin = it->minX;
                    cMax = it->maxX;
                } else if (it->minX - cMax < kClusterGap) {
                    cMax = std::max(cMax, it->maxX);
                } else {
                    break;
                }
            }
            if (!haveCluster) return false;

            // Jump so the apex of the arc lands over the middle of the group.
            float center = (cMin + cMax) * 0.5f;
            float apexDistance = m_speed * kCubeApexTime;
            if (center - pcx <= apexDistance) {
                m_handledMaxX = cMax;
                m_holdFrames = kCubeHoldFrames;
                if (debug()) log::info("AutoBot: cube jump, group {:.0f}-{:.0f}", cMin, cMax);
                return true;
            }
            return false;
        }

        // ---- ship ---------------------------------------------------------
        bool shipDecide(PlayerObject* player, CCRect const& pr) {
            float inf = std::numeric_limits<float>::infinity();
            float xa = pr.getMinX();
            float xb = pr.getMaxX() + std::max(100.f, m_speed * 0.35f);
            float pcy = (pr.getMinY() + pr.getMaxY()) * 0.5f;
            float half = (pr.getMaxY() - pr.getMinY()) * 0.5f;

            float floorTop = -inf;
            float ceilBottom = inf;
            for (auto it = firstFrom(xa - 60.f); it != m_obstacles.end(); ++it) {
                if (it->minX > xb) break;
                if (it->maxX < xa) continue;
                float mid = (it->minY + it->maxY) * 0.5f;
                if (mid <= pcy) {
                    floorTop = std::max(floorTop, it->maxY);
                } else {
                    ceilBottom = std::min(ceilBottom, it->minY);
                }
            }

            // Stay where we are unless that is outside the safe band.
            float margin = half + 8.f;
            float lo = floorTop + margin;
            float hi = ceilBottom - margin;
            float target = pcy;
            if (lo > hi) {
                if (floorTop > -inf && ceilBottom < inf) target = (floorTop + ceilBottom) * 0.5f;
            } else {
                target = std::clamp(pcy, lo, hi);
            }

            float sign = player->m_isUpsideDown ? -1.f : 1.f;  // pressing pushes "up" relative to gravity
            float err = target - pcy;
            float desiredVy = std::clamp(err * 5.f, -240.f, 240.f);
            float vyRel = m_vy * sign;
            float desiredRel = desiredVy * sign;

            if (std::fabs(err) < 4.f && std::fabs(vyRel) < 20.f) return m_pressed;  // settled
            return vyRel < desiredRel;
        }
    };
}

class $modify(AutoBotPlayLayer, PlayLayer) {
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        if (PlayLayer::get() == this) autobot::Bot::get().update(this, dt);
    }
};

$execute {
    log::info("AutoBot 0.1.0 loaded.");
}
