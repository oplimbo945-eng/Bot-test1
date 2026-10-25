#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PauseLayer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace geode::prelude;

// AutoBot v0.2.0 - reactive autoplayer for classic mode, cube and ship only.
//
// Every game update the bot looks at the static hazards/solids just ahead of
// the player and decides to press or release the jump button. It does NOT read
// a macro.
//
// Phase 1: pause-menu toggle + ground-checked cube jumps.
// Phase 2: jump orbs (yellow/pink/red) + the cube jump timing now calibrates
//          itself from the real jumps it performs.
// Still missing: pads logic, portals, blue/green/black/dash orbs, ball, ufo,
// wave, spider, dual mode, exact slopes, moving objects.

namespace autobot {
    constexpr float kDefaultSpeed = 311.58f;  // units/second at 1x speed
    // Seconds from the decision to the apex of the cube jump (includes about one
    // frame of input latency). Starting guess; it is re-measured on every jump.
    constexpr float kCubeApexTime = 0.212f;
    constexpr float kClusterGap = 36.f;  // obstacles closer than this are one group
    constexpr int kCubeHoldFrames = 3;
    constexpr int kOrbHoldFrames = 2;
    constexpr float kOrbPad = 6.f;  // how much the orb hitbox is widened

    struct Obstacle {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
    };

    struct Orb {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
        bool used = false;
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
                float sign = player->m_isUpsideDown ? -1.f : 1.f;
                trackApex((pr.getMinX() + pr.getMaxX()) * 0.5f, sign);
                want = cubeDecide(pr, player->m_isOnGround);
            }
            setPressed(layer, want);
        }

    private:
        std::vector<Obstacle> m_obstacles;
        std::vector<Orb> m_orbs;
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

        // self-calibration of the cube jump
        float m_apexTime = kCubeApexTime;
        bool m_jumpActive = false;
        bool m_sawRising = false;
        float m_jumpStartX = 0.f;
        int m_jumpFrames = 0;

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
            m_jumpActive = false;
            m_sawRising = false;
            for (auto& o : m_orbs) o.used = false;
        }

        static bool plausibleSpeed(float v) { return v > 150.f && v < 700.f; }

        static bool isJumpOrb(GameObjectType t) {
            return t == GameObjectType::YellowJumpRing || t == GameObjectType::PinkJumpRing ||
                   t == GameObjectType::RedJumpRing;
        }

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
            m_orbs.clear();
            if (!layer->m_objects) return;

            CCArrayExt<GameObject*> objects = layer->m_objects;
            for (auto* obj : objects) {
                if (!obj) continue;
                auto type = obj->m_objectType;

                if (isJumpOrb(type)) {
                    CCRect const& r = obj->getObjectRect();
                    Orb orb;
                    orb.minX = r.getMinX();
                    orb.maxX = r.getMaxX();
                    orb.minY = r.getMinY();
                    orb.maxY = r.getMaxY();
                    if (orb.maxX - orb.minX < 0.01f || orb.maxY - orb.minY < 0.01f) {
                        float x = obj->getPositionX();
                        float y = obj->getPositionY();
                        orb.minX = x - 18.f;
                        orb.maxX = x + 18.f;
                        orb.minY = y - 18.f;
                        orb.maxY = y + 18.f;
                    }
                    m_orbs.push_back(orb);
                    continue;
                }

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
            std::sort(m_orbs.begin(), m_orbs.end(),
                      [](Orb const& a, Orb const& b) { return a.minX < b.minX; });
            if (debug()) {
                log::info("AutoBot: indexed {} obstacles, {} orbs", m_obstacles.size(), m_orbs.size());
            }
        }

        std::vector<Obstacle>::const_iterator firstFrom(float x) const {
            return std::lower_bound(m_obstacles.begin(), m_obstacles.end(), x,
                                    [](Obstacle const& o, float value) { return o.minX < value; });
        }

        // ---- cube ---------------------------------------------------------

        // Measures how long a real jump takes to reach its apex and blends it
        // into m_apexTime, so the timing fixes itself after a few jumps.
        void trackApex(float pcx, float sign) {
            if (!m_jumpActive) return;
            ++m_jumpFrames;
            float vyRel = m_vy * sign;
            if (vyRel > 30.f) m_sawRising = true;
            if (m_sawRising && vyRel <= 0.f) {
                float t = (pcx - m_jumpStartX) / m_speed;
                if (t > 0.1f && t < 0.4f) {
                    m_apexTime = m_apexTime * 0.7f + t * 0.3f;
                    if (debug()) log::info("AutoBot: apex {:.3f}s, using {:.3f}s", t, m_apexTime);
                }
                m_jumpActive = false;
            } else if (m_jumpFrames > 90) {
                m_jumpActive = false;
            }
        }

        // Press while inside a jump orb (only in the air, only once per orb).
        bool orbDecide(CCRect const& pr) {
            auto it = std::lower_bound(m_orbs.begin(), m_orbs.end(), pr.getMinX() - 80.f,
                                       [](Orb const& o, float value) { return o.minX < value; });
            for (; it != m_orbs.end(); ++it) {
                if (it->minX > pr.getMaxX() + kOrbPad) break;
                if (it->used) continue;
                bool overlapX = it->maxX + kOrbPad >= pr.getMinX() && it->minX - kOrbPad <= pr.getMaxX();
                bool overlapY = it->maxY + kOrbPad >= pr.getMinY() && it->minY - kOrbPad <= pr.getMaxY();
                if (overlapX && overlapY) {
                    it->used = true;
                    m_holdFrames = kOrbHoldFrames;
                    m_jumpActive = false;  // an orb jump says nothing about a normal jump
                    if (debug()) log::info("AutoBot: orb at {:.0f},{:.0f}", it->minX, it->minY);
                    return true;
                }
            }
            return false;
        }

        bool cubeDecide(CCRect const& pr, bool onGround) {
            if (m_holdFrames > 0) {
                --m_holdFrames;
                return true;
            }
            if (!onGround && orbDecide(pr)) return true;
            if (!onGround) return false;  // do not waste a group on a jump in the air

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
            float apexDistance = m_speed * m_apexTime;
            if (center - pcx <= apexDistance) {
                m_handledMaxX = cMax;
                m_holdFrames = kCubeHoldFrames;
                m_jumpActive = true;
                m_sawRising = false;
                m_jumpStartX = pcx;
                m_jumpFrames = 0;
                if (debug()) {
                    log::info("AutoBot: cube jump, dist {:.1f}, group {:.0f}-{:.0f}", center - pcx, cMin, cMax);
                }
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

// Pause menu: simple on/off toggle (the full Offbot-style panel comes in phase 4).
class $modify(AutoBotPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();

        auto winSize = CCDirector::get()->getWinSize();
        bool on = Mod::get()->getSettingValue<bool>("enabled");

        auto title = CCLabelBMFont::create("AutoBot", "goldFont.fnt");
        title->setScale(0.8f);
        title->setPosition({70.f, winSize.height - 28.f});
        this->addChild(title);

        auto menu = CCMenu::create();
        menu->setPosition({0.f, 0.f});

        auto toggler = CCMenuItemToggler::createWithStandardSprites(
            this, menu_selector(AutoBotPauseLayer::onAutoBotToggle), 0.8f);
        toggler->toggle(on);
        toggler->setPosition({34.f, winSize.height - 62.f});
        menu->addChild(toggler);

        auto label = CCLabelBMFont::create("Activar", "bigFont.fnt");
        label->setScale(0.45f);
        label->setAnchorPoint({0.f, 0.5f});
        label->setPosition({56.f, winSize.height - 62.f});
        this->addChild(label);

        this->addChild(menu);
    }

    void onAutoBotToggle(CCObject*) {
        bool now = !Mod::get()->getSettingValue<bool>("enabled");
        Mod::get()->setSettingValue<bool>("enabled", now);
    }
};

$execute {
    log::info("AutoBot 0.2.0 loaded.");
}
