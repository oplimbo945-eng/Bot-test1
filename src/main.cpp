#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/ui/Popup.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace geode::prelude;

// AutoBot v0.4.0 - autoplayer for classic mode.
//
// Phase 4: the cube no longer uses a "jump when the group is close" guess.
// Every frame the bot SIMULATES the cube physics a bit into the future
// (gravity, jump, solid blocks you can land on, spikes, jump orbs) and looks
// for a sequence of jumps that survives. It jumps near the end of the safe
// window, so frame-perfect jumps, platforms, chains and orbs are planned
// instead of guessed. Ship keeps the old simple controller for now.
//
// The recorder now detects YOUR jumps by watching the cube leave the ground
// (no input hook needed).
//
// Still missing: pads, portals, blue/green/black/dash orbs, ball, ufo, wave,
// spider, robot, dual mode, flipped gravity, mini, exact slopes, moving objects.

namespace autobot {
    // ---- cube physics (60 Hz "frame" values converted to per second) ----
    constexpr float kGravity = 3449.5f;  // 0.958199 * 60 * 60
    constexpr float kJumpVel = 670.8f;   // 11.180032 * 60
    constexpr float kHalf = 15.f;        // cube half size (30x30)
    constexpr float kSimDt = 1.f / 120.f;
    constexpr int kHorizon = 96;         // simulated steps (0.8 s)
    constexpr int kMaxJumpDelay = 40;    // how far ahead a jump may be delayed (steps)
    constexpr int kJumpStride = 2;       // delay granularity inside the search
    constexpr int kBudgetSteps = 7000;   // max simulated steps per decision
    constexpr int kMarginSteps = 4;      // jump when the safe window is about to close

    constexpr float kClusterGap = 36.f;  // used by the recorder only
    constexpr int kCubeHoldFrames = 3;
    constexpr int kOrbHoldFrames = 2;
    constexpr float kOrbPad = 6.f;       // how much the orb hitbox is widened
    constexpr std::size_t kMaxSamples = 80;
    constexpr float kDefaultSpeed = 311.58f;

    // Menu / options. 3 is the real Geode setting "enabled"; the rest are saved values.
    enum Opt : int { OptHud = 0, OptRecord = 1, OptLearn = 2, OptEnabled = 3 };

    inline bool getOpt(int o) {
        switch (o) {
            case OptHud: return Mod::get()->getSavedValue<bool>("opt_hud", true);
            case OptRecord: return Mod::get()->getSavedValue<bool>("opt_record", false);
            case OptLearn: return Mod::get()->getSavedValue<bool>("opt_learn", true);
            default: return Mod::get()->getSettingValue<bool>("enabled");
        }
    }

    inline void setOpt(int o, bool v) {
        switch (o) {
            case OptHud: Mod::get()->setSavedValue<bool>("opt_hud", v); break;
            case OptRecord: Mod::get()->setSavedValue<bool>("opt_record", v); break;
            case OptLearn: Mod::get()->setSavedValue<bool>("opt_learn", v); break;
            default: Mod::get()->setSettingValue<bool>("enabled", v); break;
        }
    }

    struct Obstacle {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
        bool hazard = false;  // spikes etc. (false = solid block)
    };

    struct Orb {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
        float vel = kJumpVel;  // vertical speed the orb gives (approximate)
        bool used = false;
    };

    struct Pending {
        float t = 0.f;     // seconds between the jump and the group's center
        float cMax = 0.f;  // end of the group; the sample counts once we pass it alive
    };

    struct SimState {
        float x = 0.f;
        float y = 0.f;   // center of the cube
        float vy = 0.f;  // units/second, positive = up
        bool grounded = true;
        float lastOrbX = -1e9f;  // orbs with minX <= this were already handled
    };

    enum class Ev { None, Dead, Landed, Orb };

    class Bot {
    public:
        static Bot& get() {
            static Bot instance;
            return instance;
        }

        std::size_t sampleCount() {
            ensureLoaded();
            return m_samples.size();
        }

        void clearSamples() {
            ensureLoaded();
            m_samples.clear();
            m_pending.clear();
            saveSamples();
            m_lastEvent = "datos borrados";
        }

        void update(PlayLayer* layer, float dt) {
            if (!layer || !layer->m_player1) return;
            ensureLoaded();

            bool botOn = getOpt(OptEnabled);
            bool rec = getOpt(OptRecord);
            bool hud = getOpt(OptHud);

            if (!botOn && !rec && !hud) {
                if (m_pressed) setPressed(layer, false);
                m_hasLast = false;
                hideHud(layer);
                return;
            }
            if (!hud) hideHud(layer);
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
            float pcx = (pr.getMinX() + pr.getMaxX()) * 0.5f;
            float pcy = (pr.getMinY() + pr.getMaxY()) * 0.5f;
            bool onGround = player->m_isOnGround;

            bool isShip = player->m_isShip;
            bool other = player->m_isBird || player->m_isBall || player->m_isDart ||
                         player->m_isSpider || player->m_isRobot;
            bool cube = !isShip && !other;
            m_modeName = isShip ? "nave" : (other ? "no soportado" : "cubo");

            if (cube && onGround) m_floorY = std::min(m_floorY, pcy - kHalf);

            if (botOn) {
                bool want = false;
                if (isShip) {
                    logMode("ship");
                    m_plan = "nave";
                    want = shipDecide(player, pr);
                } else if (other) {
                    logMode("unsupported");
                    m_plan = "-";
                    want = false;
                } else if (player->m_isUpsideDown) {
                    logMode("flipped");
                    m_plan = "gravedad invertida (no soportado)";
                    want = false;
                } else {
                    logMode("cube");
                    want = planCube(pr, pcx, pcy, onGround);
                }
                setPressed(layer, want);
            } else {
                if (m_pressed) setPressed(layer, false);
                m_plan = "bot apagado";
            }

            if (rec) {
                // The player's own jump: the cube leaves the ground going up.
                if (cube && !botOn && m_prevGround && !onGround && m_vy > 40.f) recordTakeoff(pr);
                commitPending(px);
            } else {
                m_pending.clear();
            }
            m_prevGround = onGround;

            if (hud) updateHud(layer, botOn, rec);
        }

    private:
        std::vector<Obstacle> m_obstacles;
        std::vector<Orb> m_orbs;
        std::vector<Obstacle> m_local;  // obstacles near the player (for the simulation)
        std::vector<Orb> m_localOrbs;
        float m_maxW = 30.f;
        float m_floorY = 1e9f;
        PlayLayer* m_cacheLayer = nullptr;
        std::size_t m_cacheCount = 0;

        bool m_hasLast = false;
        bool m_prevGround = true;
        float m_lastX = 0.f;
        float m_lastY = 0.f;
        float m_speed = kDefaultSpeed;
        float m_vy = 0.f;
        float m_step = 1.f / 60.f;

        bool m_pressed = false;
        int m_holdFrames = 0;
        float m_lastUsedOrbX = -1e9f;
        char const* m_lastMode = "";
        char const* m_modeName = "-";
        char const* m_plan = "-";
        int m_dbgS0 = -1;
        int m_dbgSM = -1;

        // recorder
        bool m_loaded = false;
        std::vector<float> m_samples;
        std::vector<Pending> m_pending;
        float m_recHandledMaxX = -1e9f;

        // HUD
        bool m_hudShown = false;
        int m_hudTick = 0;
        std::string m_lastEvent = "-";

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
            m_recHandledMaxX = -1e9f;
            m_lastUsedOrbX = -1e9f;
            m_vy = 0.f;
            m_pending.clear();  // the attempt ended: unconfirmed jumps are discarded
            for (auto& o : m_orbs) o.used = false;
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

        // ---- recorder -------------------------------------------------------
        void ensureLoaded() {
            if (m_loaded) return;
            m_loaded = true;
            std::string s = Mod::get()->getSavedValue<std::string>("samples", "");
            std::size_t pos = 0;
            while (pos < s.size()) {
                std::size_t e = s.find(',', pos);
                if (e == std::string::npos) e = s.size();
                float v = std::strtof(s.substr(pos, e - pos).c_str(), nullptr);
                if (v > 0.f && v < 0.5f) m_samples.push_back(v);
                pos = e + 1;
            }
            if (m_samples.size() > kMaxSamples) {
                m_samples.erase(m_samples.begin(), m_samples.end() - kMaxSamples);
            }
        }

        void saveSamples() {
            std::string s;
            for (std::size_t i = 0; i < m_samples.size(); ++i) {
                if (i) s += ',';
                s += fmt::format("{:.4f}", m_samples[i]);
            }
            Mod::get()->setSavedValue<std::string>("samples", s);
        }

        void recordTakeoff(CCRect const& pr) {
            float cMin = 0.f, cMax = 0.f;
            if (!findCluster(pr, m_recHandledMaxX, cMin, cMax)) {
                m_lastEvent = "salto sin obstaculo";
                return;
            }
            float pcx = (pr.getMinX() + pr.getMaxX()) * 0.5f;
            float center = (cMin + cMax) * 0.5f;
            float t = (center - pcx) / m_speed;
            if (t < 0.f || t > 0.5f) return;
            m_recHandledMaxX = cMax;
            m_pending.push_back(Pending{t, cMax});
            m_lastEvent = fmt::format("grabado t={:.3f}s", t);
            if (debug()) log::info("AutoBot: recorded jump t={:.3f}s", t);
        }

        // A recorded jump only counts once the player got past its group alive.
        void commitPending(float px) {
            bool changed = false;
            for (std::size_t i = 0; i < m_pending.size();) {
                if (px > m_pending[i].cMax + 40.f) {
                    m_samples.push_back(m_pending[i].t);
                    m_pending.erase(m_pending.begin() + static_cast<long>(i));
                    changed = true;
                } else {
                    ++i;
                }
            }
            if (changed) {
                if (m_samples.size() > kMaxSamples) {
                    m_samples.erase(m_samples.begin(), m_samples.end() - kMaxSamples);
                }
                saveSamples();
                m_lastEvent = fmt::format("muestra guardada ({})", m_samples.size());
            }
        }

        // ---- HUD ------------------------------------------------------------
        void hideHud(PlayLayer* layer) {
            if (!m_hudShown) return;
            if (auto node = layer->getChildByID("autobot-hud"_spr)) node->setVisible(false);
            m_hudShown = false;
        }

        void updateHud(PlayLayer* layer, bool botOn, bool rec) {
            if (m_hudShown && (++m_hudTick % 5) != 0) return;

            auto* label = static_cast<CCLabelBMFont*>(layer->getChildByID("autobot-hud"_spr));
            if (!label) {
                label = CCLabelBMFont::create("", "chatFont.fnt");
                label->setID("autobot-hud"_spr);
                label->setAnchorPoint({0.f, 1.f});
                label->setScale(0.6f);
                label->setOpacity(210);
                label->setZOrder(200);
                auto ws = CCDirector::get()->getWinSize();
                label->setPosition({8.f, ws.height - 30.f});
                layer->addChild(label);
            }

            std::string txt = fmt::format(
                "AutoBot {}  modo: {}  vel: {:.0f}\n"
                "plan: {}  ahora:{} +margen:{}\n"
                "grabar: {}  muestras: {}  pend: {}\n"
                "{}",
                botOn ? "ON" : "OFF", m_modeName, m_speed, m_plan, m_dbgS0, m_dbgSM,
                rec ? "ON" : "OFF", m_samples.size(), m_pending.size(), m_lastEvent);
            label->setString(txt.c_str());
            label->setVisible(true);
            m_hudShown = true;
        }

        // ---- level index ------------------------------------------------------
        void ensureObstacles(PlayLayer* layer) {
            std::size_t count = layer->m_objects ? layer->m_objects->count() : 0;
            if (m_cacheLayer == layer && m_cacheCount == count) return;
            m_cacheLayer = layer;
            m_cacheCount = count;
            m_obstacles.clear();
            m_orbs.clear();
            m_floorY = 1e9f;
            m_maxW = 30.f;
            if (!layer->m_objects) return;

            CCArrayExt<GameObject*> objects = layer->m_objects;
            for (auto* obj : objects) {
                if (!obj) continue;
                auto type = obj->m_objectType;

                float orbVel = 0.f;
                if (type == GameObjectType::YellowJumpRing) orbVel = kJumpVel * 1.0f;
                else if (type == GameObjectType::PinkJumpRing) orbVel = kJumpVel * 0.7f;
                else if (type == GameObjectType::RedJumpRing) orbVel = kJumpVel * 1.45f;

                if (orbVel > 0.f) {
                    CCRect const& r = obj->getObjectRect();
                    Orb orb;
                    orb.minX = r.getMinX();
                    orb.maxX = r.getMaxX();
                    orb.minY = r.getMinY();
                    orb.maxY = r.getMaxY();
                    orb.vel = orbVel;
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

                bool hazard = type == GameObjectType::Hazard;
                if (type != GameObjectType::Solid && !hazard && type != GameObjectType::Slope) {
                    continue;
                }
                CCRect const& r = obj->getObjectRect();
                Obstacle o;
                o.minX = r.getMinX();
                o.maxX = r.getMaxX();
                o.minY = r.getMinY();
                o.maxY = r.getMaxY();
                o.hazard = hazard;
                if (o.maxX - o.minX < 0.01f || o.maxY - o.minY < 0.01f) {
                    float x = obj->getPositionX();
                    float y = obj->getPositionY();
                    o.minX = x - 15.f;
                    o.maxX = x + 15.f;
                    o.minY = y - 15.f;
                    o.maxY = y + 15.f;
                }
                m_maxW = std::max(m_maxW, std::min(o.maxX - o.minX, 300.f));
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

        // Group of obstacles ahead in the player's current row (recorder only).
        bool findCluster(CCRect const& pr, float handledMax, float& cMin, float& cMax) const {
            float front = pr.getMaxX();
            bool have = false;
            for (auto it = firstFrom(front - 8.f); it != m_obstacles.end(); ++it) {
                if (it->minX > front + 220.f) break;
                if (it->minX <= handledMax) continue;
                float overlap = std::min(it->maxY, pr.getMaxY()) - std::max(it->minY, pr.getMinY());
                if (overlap <= 4.f) continue;
                if (!have) {
                    have = true;
                    cMin = it->minX;
                    cMax = it->maxX;
                } else if (it->minX - cMax < kClusterGap) {
                    cMax = std::max(cMax, it->maxX);
                } else {
                    break;
                }
            }
            return have;
        }

        // ---- cube simulation ----------------------------------------------------

        // Copies the obstacles/orbs around the player so the simulation only
        // looks at a small list.
        void gatherLocal(float px) {
            m_local.clear();
            m_localOrbs.clear();
            float xa = px - 60.f;
            float xb = px + m_speed * 0.9f + 100.f;
            for (auto it = firstFrom(xa - m_maxW); it != m_obstacles.end() && it->minX <= xb; ++it) {
                if (it->maxX >= xa) m_local.push_back(*it);
            }
            auto ot = std::lower_bound(m_orbs.begin(), m_orbs.end(), xa - 80.f,
                                       [](Orb const& o, float value) { return o.minX < value; });
            for (; ot != m_orbs.end() && ot->minX <= xb; ++ot) {
                if (ot->maxX >= xa) m_localOrbs.push_back(*ot);
            }
        }

        // Highest surface under the cube (a block top or the floor).
        float supportTop(float x, float feet) const {
            float best = m_floorY;
            float l = x - kHalf + 2.f;
            float r = x + kHalf - 2.f;
            for (auto const& o : m_local) {
                if (o.minX > r) break;
                if (o.maxX < l || o.hazard) continue;
                if (o.maxY <= feet + 6.f && o.maxY > best) best = o.maxY;
            }
            return best;
        }

        // One simulated step of 1/120 s.
        Ev stepOnce(SimState& s, bool allowOrbs, int& orbIdx) const {
            float prevBottom = s.y - kHalf;
            s.x += m_speed * kSimDt;

            if (!s.grounded) {
                s.vy -= kGravity * kSimDt;
                s.y += s.vy * kSimDt;
            } else {
                float feet = s.y - kHalf;
                if (supportTop(s.x, feet) < feet - 1.f) {  // rolled off a ledge
                    s.grounded = false;
                    s.vy = 0.f;
                }
            }

            Ev ev = Ev::None;
            float bottom = s.y - kHalf;

            if (!s.grounded && bottom <= m_floorY && s.vy <= 0.f) {  // landed on the floor
                s.y = m_floorY + kHalf;
                s.vy = 0.f;
                s.grounded = true;
                ev = Ev::Landed;
                bottom = s.y - kHalf;
            }

            float left = s.x - kHalf;
            float right = s.x + kHalf;
            float top = s.y + kHalf;

            for (auto const& o : m_local) {
                if (o.minX > right) break;
                if (o.maxX < left) continue;
                float shrink = o.hazard ? 2.f : 0.f;
                bool overlapX = right - shrink > o.minX && left + shrink < o.maxX;
                bool overlapY = top - shrink > o.minY + 0.5f && bottom + shrink < o.maxY - 0.5f;
                if (!overlapX || !overlapY) continue;
                if (o.hazard) return Ev::Dead;
                bool fromAbove = prevBottom >= o.maxY - 6.f;
                if (fromAbove && s.vy <= 0.f) {  // lands on top of the block
                    s.y = o.maxY + kHalf;
                    s.vy = 0.f;
                    if (!s.grounded) {
                        s.grounded = true;
                        ev = Ev::Landed;
                    }
                    bottom = s.y - kHalf;
                    top = s.y + kHalf;
                } else {
                    return Ev::Dead;  // side or underside of a block
                }
            }

            if (ev != Ev::None) return ev;

            if (allowOrbs && !s.grounded) {
                for (std::size_t i = 0; i < m_localOrbs.size(); ++i) {
                    auto const& ob = m_localOrbs[i];
                    if (ob.minX - kOrbPad > right) break;
                    if (ob.minX <= s.lastOrbX) continue;
                    bool ox = ob.maxX + kOrbPad >= left && ob.minX - kOrbPad <= right;
                    bool oy = ob.maxY + kOrbPad >= bottom && ob.minY - kOrbPad <= top;
                    if (ox && oy) {
                        orbIdx = static_cast<int>(i);
                        return Ev::Orb;
                    }
                }
            }
            return Ev::None;
        }

        // Is there a way to survive the next `left` steps? `depth` = how many
        // more jumps may be used. Orbs are a free branch (press or skip).
        bool search(SimState s, int left, int depth, int& budget, bool allowOrbs) const {
            // airborne part
            while (!s.grounded) {
                if (left <= 0) return true;
                if (--budget < 0) return false;
                int orbIdx = -1;
                Ev ev = stepOnce(s, allowOrbs, orbIdx);
                --left;
                if (ev == Ev::Dead) return false;
                if (ev == Ev::Orb) {
                    SimState skip = s;
                    skip.lastOrbX = m_localOrbs[static_cast<std::size_t>(orbIdx)].maxX;
                    if (search(skip, left, depth, budget, allowOrbs)) return true;
                    SimState press = skip;
                    press.vy = m_localOrbs[static_cast<std::size_t>(orbIdx)].vel;
                    return search(press, left, depth, budget, allowOrbs);
                }
            }

            // grounded part
            if (left <= 0) return true;
            SimState path[kMaxJumpDelay + 1];
            int n = 0;
            SimState g = s;
            for (int i = 0; i < left; ++i) {
                if (n <= kMaxJumpDelay) path[n++] = g;
                if (--budget < 0) return false;
                int oi = -1;
                Ev ev = stepOnce(g, false, oi);
                if (ev == Ev::Dead) break;  // dies if it never jumps
                if (!g.grounded) {          // rolled off a ledge: keep going in the air
                    if (search(g, left - i - 1, 0, budget, false)) return true;
                    break;
                }
                if (i == left - 1) return true;  // reached the horizon without jumping
            }
            if (depth <= 0) return false;

            for (int k = 0; k < n; k += kJumpStride) {
                SimState t = path[k];
                t.vy = kJumpVel;
                t.grounded = false;
                if (search(t, left - k, depth - 1, budget, allowOrbs)) return true;
            }
            return false;
        }

        // Wait `k` steps on the ground, jump, then survive (with up to `depth` more jumps).
        bool jumpAt(SimState s, int k, int depth, int& budget, bool allowOrbs) const {
            for (int i = 0; i < k; ++i) {
                if (--budget < 0) return false;
                int oi = -1;
                Ev ev = stepOnce(s, false, oi);
                if (ev == Ev::Dead || !s.grounded) return false;
            }
            s.vy = kJumpVel;
            s.grounded = false;
            return search(s, kHorizon - k, depth, budget, allowOrbs);
        }

        int stepsToDeath(SimState s) const {
            for (int i = 0; i < kHorizon; ++i) {
                int oi = -1;
                Ev ev = stepOnce(s, false, oi);
                if (ev == Ev::Dead) return i;
                if (!s.grounded) {
                    // falling without input: count until it lands or dies
                    continue;
                }
            }
            return kHorizon;
        }

        int latencySteps() const {
            int l = static_cast<int>(std::lround(m_step / kSimDt));
            return std::clamp(l, 1, 4);
        }

        int findOrb(float cx, float cy) const {
            float l = cx - kHalf;
            float r = cx + kHalf;
            float b = cy - kHalf;
            float t = cy + kHalf;
            auto it = std::lower_bound(m_orbs.begin(), m_orbs.end(), l - 80.f,
                                       [](Orb const& o, float value) { return o.minX < value; });
            for (; it != m_orbs.end(); ++it) {
                if (it->minX - kOrbPad > r) break;
                if (it->used || it->minX <= m_lastUsedOrbX) continue;
                bool ox = it->maxX + kOrbPad >= l && it->minX - kOrbPad <= r;
                bool oy = it->maxY + kOrbPad >= b && it->minY - kOrbPad <= t;
                if (ox && oy) return static_cast<int>(it - m_orbs.begin());
            }
            return -1;
        }

        bool planCube(CCRect const& pr, float pcx, float pcy, bool onGround) {
            if (m_holdFrames > 0) {
                --m_holdFrames;
                return true;
            }
            if (m_floorY > 1e8f) m_floorY = pcy - kHalf;

            SimState st;
            st.x = pcx;
            st.y = pcy;
            st.vy = onGround ? 0.f : m_vy;
            st.grounded = onGround;
            st.lastOrbX = m_lastUsedOrbX;
            m_dbgS0 = -1;
            m_dbgSM = -1;

            gatherLocal(pr.getMinX());

            if (!onGround) {
                // In the air: only an orb can change anything.
                int idx = findOrb(pcx, pcy);
                if (idx < 0) {
                    m_plan = "en el aire";
                    return false;
                }
                Orb& ob = m_orbs[static_cast<std::size_t>(idx)];
                int budget = kBudgetSteps;
                SimState skip = st;
                skip.lastOrbX = ob.maxX;
                if (search(skip, kHorizon, 2, budget, true)) {
                    m_plan = "orbe: no hace falta";
                    return false;
                }
                ob.used = true;
                m_lastUsedOrbX = ob.minX;
                m_holdFrames = kOrbHoldFrames;
                m_plan = "orbe";
                m_lastEvent = "orbe";
                if (debug()) log::info("AutoBot: orb at {:.0f},{:.0f}", ob.minX, ob.minY);
                return true;
            }

            int budget = kBudgetSteps;
            if (search(st, kHorizon, 0, budget, false)) {
                m_plan = "libre";
                return false;
            }

            int lat = latencySteps();
            budget = kBudgetSteps;
            bool s0 = jumpAt(st, lat, 2, budget, true);
            m_dbgS0 = s0 ? 1 : 0;

            if (s0) {
                budget = kBudgetSteps;
                bool sm = jumpAt(st, lat + kMarginSteps, 2, budget, true);
                m_dbgSM = sm ? 1 : 0;
                if (sm) {
                    m_plan = "esperar ventana";
                    return false;
                }
                m_holdFrames = kCubeHoldFrames;
                m_plan = "saltar";
                m_lastEvent = "salto planeado";
                if (debug()) log::info("AutoBot: planned jump at x={:.0f}", pcx);
                return true;
            }

            // No plan works right now. If death is about to happen, try anyway.
            int death = stepsToDeath(st);
            if (death <= lat + 1) {
                m_holdFrames = kCubeHoldFrames;
                m_plan = "sin plan (salto de emergencia)";
                m_lastEvent = "salto de emergencia";
                return true;
            }
            m_plan = "esperar";
            return false;
        }

        // ---- ship (old simple controller) -------------------------------------
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

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

class $modify(AutoBotPlayLayer, PlayLayer) {
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        if (PlayLayer::get() == this) autobot::Bot::get().update(this, dt);
    }
};

// ---------------------------------------------------------------------------
// AutoBot menu (popup)
// ---------------------------------------------------------------------------

class AutoBotMenu : public geode::Popup {
protected:
    CCLabelBMFont* m_countLabel = nullptr;

    bool init() {
        if (!Popup::init(320.f, 280.f)) return false;
        this->setTitle("AutoBot");

        struct RowInfo {
            char const* text;
            int id;
        };
        RowInfo rows[] = {
            {"AutoBot activado", autobot::OptEnabled},
            {"Texto en pantalla", autobot::OptHud},
            {"Grabar mis saltos", autobot::OptRecord},
            {"Aprender (pronto)", autobot::OptLearn},
        };

        float y = 62.f;
        for (auto& row : rows) {
            auto toggler = CCMenuItemToggler::createWithStandardSprites(
                this, menu_selector(AutoBotMenu::onToggle), 0.7f);
            toggler->setTag(row.id);
            toggler->toggle(autobot::getOpt(row.id));
            m_buttonMenu->addChildAtPosition(toggler, Anchor::Center, {-125.f, y});

            auto label = CCLabelBMFont::create(row.text, "bigFont.fnt");
            label->setScale(0.5f);
            label->setAnchorPoint({0.f, 0.5f});
            m_mainLayer->addChildAtPosition(label, Anchor::Center, {-100.f, y});

            y -= 36.f;
        }

        m_countLabel = CCLabelBMFont::create("", "chatFont.fnt");
        m_countLabel->setScale(0.7f);
        m_mainLayer->addChildAtPosition(m_countLabel, Anchor::Center, {0.f, -84.f});
        this->refreshCount();

        auto clearBtn = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("Borrar datos"), nullptr, this, menu_selector(AutoBotMenu::onClear));
        clearBtn->setScale(0.7f);
        m_buttonMenu->addChildAtPosition(clearBtn, Anchor::Center, {0.f, -112.f});

        return true;
    }

    void refreshCount() {
        if (!m_countLabel) return;
        auto text = fmt::format("Saltos grabados: {}", autobot::Bot::get().sampleCount());
        m_countLabel->setString(text.c_str());
    }

    void onToggle(CCObject* sender) {
        int id = static_cast<CCNode*>(sender)->getTag();
        autobot::setOpt(id, !autobot::getOpt(id));
    }

    void onClear(CCObject*) {
        autobot::Bot::get().clearSamples();
        this->refreshCount();
    }

public:
    static AutoBotMenu* create() {
        auto ret = new AutoBotMenu();
        if (ret->init()) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }
};

// Pause menu: one button that opens the AutoBot menu.
class $modify(AutoBotPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();

        auto winSize = CCDirector::get()->getWinSize();

        auto menu = CCMenu::create();
        menu->setPosition({0.f, 0.f});

        auto btn = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("AutoBot"), nullptr, this, menu_selector(AutoBotPauseLayer::onAutoBotMenu));
        btn->setScale(0.7f);
        btn->setPosition({70.f, winSize.height - 30.f});
        menu->addChild(btn);

        this->addChild(menu);
    }

    void onAutoBotMenu(CCObject*) {
        if (auto popup = AutoBotMenu::create()) popup->show();
    }
};

$execute {
    log::info("AutoBot 0.4.0 loaded.");
}
EOF
wc -l /mnt/user-data/outputs/main.cpp
# quick brace balance check
python3 - << 'PY'
s=open('/mnt/user-data/outputs/main.cpp').read()
print(s.count('{'), s.count('}'), s.count('('), s.count(')'))
PY
