#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/ui/Popup.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

using namespace geode::prelude;

// AutoBot v0.5.0 - autoplayer for classic mode, all vehicles.
//
// The bot keeps a physics model of every game mode (cube, ship, ball, ufo,
// wave, robot, spider) and plans by search: every frame it simulates up to ~1 s
// into the future, trying "press / release" at every 1/60 s step, and picks the
// first input of a sequence that survives (blocks, slopes, spikes, orbs, pads,
// gravity / size / speed / mode portals). Memoised DFS, so it is cheap.
// Physics constants (jump power, gravity, ship acceleration, wave slope, pad
// strength) start from estimates and are CALIBRATED ONLINE from what the real
// player does, so the model corrects itself while you play.
//
// Not supported: dual mode (only player 1 is planned), dash orbs, teleport
// portals, moving / rotating objects, triggers that move the ground.

namespace autobot {
    constexpr float kDec = 1.f / 60.f;     // one decision = one 60 Hz frame
    constexpr float kSub = 1.f / 120.f;    // simulation sub-step
    constexpr int kHorizon = 64;           // decisions looked ahead (~1.07 s)
    constexpr int kBudget = 16000;         // max sub-steps per frame
    constexpr float kOrbPad = 6.f;         // orb hitbox widening
    constexpr float kLand = 8.f;           // how deep a corner may clip and still count as landing
    constexpr float kSafe = 1.0f;          // extra safety margin on hazards
    constexpr float kDefaultSpeed = 311.58f;
    constexpr float kMinHold = 0.035f;     // cube: minimum hold time of a tap (s)

    enum Opt : int { OptHud = 0, OptEnabled = 3 };

    inline bool getOpt(int o) {
        if (o == OptHud) return Mod::get()->getSavedValue<bool>("opt_hud", true);
        return Mod::get()->getSettingValue<bool>("enabled");
    }
    inline void setOpt(int o, bool v) {
        if (o == OptHud) Mod::get()->setSavedValue<bool>("opt_hud", v);
        else Mod::get()->setSettingValue<bool>("enabled", v);
    }

    enum class Mode : int { Cube = 0, Ship, Ball, Ufo, Wave, Robot, Spider };

    enum class Kind : int {
        Solid, Hazard, Slope,
        PadY, PadP, PadR, PadB,
        OrbY, OrbP, OrbR, OrbB, OrbG, OrbK,
        PCube, PShip, PBall, PUfo, PWave, PRobot, PSpider,
        GravN, GravF, Mini, Big, Speed,
        Count
    };

    inline bool isOrb(Kind k) { return k >= Kind::OrbY && k <= Kind::OrbK; }
    inline bool isPad(Kind k) { return k >= Kind::PadY && k <= Kind::PadB; }
    inline bool isPortal(Kind k) { return k >= Kind::PCube && k <= Kind::Speed; }

    struct Obj {
        float x0 = 0, x1 = 0, y0 = 0, y1 = 0;
        Kind k = Kind::Solid;
        float val = 0.f;      // speed portals: units/s
        int corner = 0;       // slopes: 0 BR, 1 BL (top-facing)
    };

    struct Phys {
        float g = 3449.5f;     // gravity (units/s^2)
        float jump = 670.8f;   // jump / ufo / robot impulse
        float upA = 2300.f;    // ship: acceleration while holding
        float downA = -2500.f; // ship: acceleration while released
        float capUp = 540.f;   // ship/ufo speed caps
        float capDown = 900.f;
        float slope = 1.f;     // wave: |vy| / vx
    };

    struct Sim {
        float x = 0, y = 0, vy = 0;
        Mode mode = Mode::Cube;
        int grav = 1;           // 1: pulled down, -1: pulled up
        bool mini = false;
        float speed = kDefaultSpeed;
        bool grounded = true;
        bool prevPress = false;
        int holdLeft = 0;       // forced press decisions (cube tap)
        std::uint64_t used = 0; // pads/orbs already consumed (window of 64)
        float robotT = 0.f;
        float slopeVy = 0.f;    // speed kept when leaving a rising slope
    };

    inline float halfOf(Mode m, bool mini) {
        float h = (m == Mode::Wave) ? 10.f : 15.f;
        return mini ? h * 0.6f : h;
    }

    inline float slopeH(Obj const& o, float x) {
        float w = std::max(o.x1 - o.x0, 1.f);
        float f = std::clamp((x - o.x0) / w, 0.f, 1.f);
        float h = o.y1 - o.y0;
        return o.corner == 0 ? o.y0 + f * h : o.y1 - f * h;
    }

    class Bot {
    public:
        static Bot& get() {
            static Bot instance;
            return instance;
        }

        Bot() { initPhys(); }

        void resetPhysics() {
            initPhys();
            m_lastEvent = "fisica reiniciada";
        }

        void onReset() { resetRun(); }

        void invalidate() {
            m_cacheLayer = nullptr;
            m_cacheCount = 0;
            m_solids.clear(); m_wide.clear(); m_haz.clear(); m_inter.clear(); m_usedI.clear();
            m_hasLast = false;
            m_floorY = 105.f;
            m_ceilY = 405.f;
            m_pressed = false;
        }

        std::string learnedInfo() const {
            auto const& c = m_ph[0][0];
            auto const& s = m_ph[1][0];
            return fmt::format("cubo salto {:.0f}  nave {:.0f}/{:.0f}", c.jump, s.upA, s.downA);
        }

        void update(PlayLayer* layer, float dt) {
            if (!layer || !layer->m_player1) return;
            bool botOn = getOpt(OptEnabled);
            bool hud = getOpt(OptHud);

            if (!botOn && !hud) {
                if (m_pressed) setPressed(layer, false);
                m_hasLast = false;
                hideHud(layer);
                return;
            }
            if (!hud) hideHud(layer);
            if (dt <= 0.f) return;

            PlayerObject* p = layer->m_player1;
            if (p->m_isDead) return;
            ensureLevel(layer);

            float px = p->getPositionX();
            float py = p->getPositionY();
            float prevVy = m_vy;
            if (!m_hasLast || px < m_lastX - 40.f) {
                resetRun();
            } else {
                measure(px - m_lastX, py - m_lastY, dt);
            }
            m_lastX = px;
            m_lastY = py;
            bool hadLast = m_hasLast;
            m_hasLast = true;

            CCRect pr = p->getObjectRect();
            float pcx = (pr.getMinX() + pr.getMaxX()) * 0.5f;
            float pcy = (pr.getMinY() + pr.getMaxY()) * 0.5f;
            bool onGround = p->m_isOnGround;

            Mode mode = Mode::Cube;
            if (p->m_isShip) mode = Mode::Ship;
            else if (p->m_isBird) mode = Mode::Ufo;
            else if (p->m_isBall) mode = Mode::Ball;
            else if (p->m_isDart) mode = Mode::Wave;
            else if (p->m_isRobot) mode = Mode::Robot;
            else if (p->m_isSpider) mode = Mode::Spider;
            int grav = p->m_isUpsideDown ? -1 : 1;
            bool mini = p->getScale() < 0.8f;
            m_modeName = kModeNames[static_cast<int>(mode)];

            if (hadLast) learn(mode, mini, grav, onGround, prevVy);
            trackSurfaces(mode, grav, onGround, pcx, pcy, prevVy);
            markTouches(pcx, pcy, mode, mini);

            if (botOn) {
                bool want = plan(mode, mini, grav, onGround, pcx, pcy, dt);
                setPressed(layer, want, pcx, pcy, mode, mini);
            } else {
                if (m_pressed) setPressed(layer, false);
                m_plan = "bot apagado";
            }

            m_prevOnGround = onGround;
            m_prevMode = mode;
            if (hud) updateHud(layer, botOn);
        }

    private:
        static constexpr char const* kModeNames[7] = {
            "cubo", "nave", "bola", "ufo", "wave", "robot", "spider"};

        // level index (sorted by x0)
        std::vector<Obj> m_solids, m_wide, m_haz, m_inter;
        std::vector<char> m_usedI;
        float m_maxWs = 30.f, m_maxWh = 30.f, m_maxWi = 40.f;
        PlayLayer* m_cacheLayer = nullptr;
        std::size_t m_cacheCount = 0;

        Phys m_ph[7][2];
        float m_imp[static_cast<int>(Kind::Count)] = {};

        float m_floorY = 105.f;
        float m_ceilY = 405.f;

        bool m_hasLast = false;
        bool m_prevOnGround = true;
        Mode m_prevMode = Mode::Cube;
        float m_lastX = 0.f, m_lastY = 0.f;
        float m_speed = kDefaultSpeed;
        float m_vy = 0.f;
        float m_step = 1.f / 60.f;

        bool m_pressed = false;
        float m_holdTimer = 0.f;
        int m_winBase = 0;
        Kind m_prevTouch = Kind::Solid;

        char const* m_modeName = "-";
        char const* m_plan = "-";
        int m_failStreak = 0;
        int m_usedBudget = 0;

        bool m_hudShown = false;
        int m_hudTick = 0;
        std::string m_lastEvent = "-";

        static bool debug() { return Mod::get()->getSettingValue<bool>("debug_log"); }

        // ---- physics defaults ----------------------------------------------
        void initPhys() {
            for (int m = 0; m < 7; ++m) {
                for (int s = 0; s < 2; ++s) m_ph[m][s] = Phys{};
            }
            auto set = [&](Mode m, float g, float jump) {
                for (int s = 0; s < 2; ++s) {
                    m_ph[static_cast<int>(m)][s].g = g;
                    m_ph[static_cast<int>(m)][s].jump = jump;
                }
            };
            set(Mode::Cube, 3449.5f, 670.8f);
            set(Mode::Ball, 2070.f, 0.f);
            set(Mode::Ufo, 2100.f, 420.f);
            set(Mode::Robot, 3100.f, 560.f);
            set(Mode::Spider, 3449.5f, 0.f);
            m_ph[0][1].jump = 600.f;  // mini cube (calibrated online)
            for (int s = 0; s < 2; ++s) {
                auto& sh = m_ph[static_cast<int>(Mode::Ship)][s];
                sh.upA = 2300.f; sh.downA = -2500.f; sh.capUp = 540.f; sh.capDown = 480.f;
                m_ph[static_cast<int>(Mode::Ufo)][s].capUp = 600.f;
                m_ph[static_cast<int>(Mode::Ufo)][s].capDown = 700.f;
            }
            m_imp[static_cast<int>(Kind::PadY)] = 864.f;
            m_imp[static_cast<int>(Kind::PadP)] = 540.f;
            m_imp[static_cast<int>(Kind::PadR)] = 1230.f;
            m_imp[static_cast<int>(Kind::OrbY)] = 670.8f;
            m_imp[static_cast<int>(Kind::OrbP)] = 470.f;
            m_imp[static_cast<int>(Kind::OrbR)] = 970.f;
            m_imp[static_cast<int>(Kind::OrbG)] = 540.f;
            m_imp[static_cast<int>(Kind::OrbK)] = 1100.f;
        }

        void resetRun() {
            m_pressed = false;
            m_holdTimer = 0.f;
            m_vy = 0.f;
            m_failStreak = 0;
            m_prevTouch = Kind::Solid;
            std::fill(m_usedI.begin(), m_usedI.end(), 0);
        }

        static bool plausibleSpeed(float v) { return v > 150.f && v < 700.f; }

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

        void setPressed(PlayLayer* layer, bool want, float cx = 0.f, float cy = 0.f,
                        Mode mode = Mode::Cube, bool mini = false) {
            if (want == m_pressed) return;
            if (want) {
                // an orb under the player is consumed by this press
                float h = halfOf(mode, mini);
                for (std::size_t i = 0; i < m_inter.size(); ++i) {
                    auto const& o = m_inter[i];
                    if (!isOrb(o.k) || m_usedI[i]) continue;
                    if (o.x0 - kOrbPad > cx + h) break;
                    if (o.x1 + kOrbPad >= cx - h && o.y1 + kOrbPad >= cy - h &&
                        o.y0 - kOrbPad <= cy + h) {
                        m_usedI[i] = 1;
                        break;
                    }
                }
                if (mode == Mode::Cube || mode == Mode::Robot) m_holdTimer = kMinHold;
            }
            m_pressed = want;
            layer->handleButton(want, 1, true);
            if (debug()) log::info("AutoBot: {}", want ? "press" : "release");
        }

        // ---- level index ----------------------------------------------------
        static Kind classify(GameObject* obj) {
            switch (obj->m_objectID) {
                case 200: case 201: case 202: case 203: case 1334: return Kind::Speed;
                default: break;
            }
            switch (obj->m_objectType) {
                case GameObjectType::Solid: return Kind::Solid;
                case GameObjectType::Hazard: return Kind::Hazard;
                case GameObjectType::Slope: return Kind::Slope;
                case GameObjectType::YellowJumpPad: return Kind::PadY;
                case GameObjectType::PinkJumpPad: return Kind::PadP;
                case GameObjectType::RedJumpPad: return Kind::PadR;
                case GameObjectType::GravityPad: return Kind::PadB;
                case GameObjectType::YellowJumpRing: return Kind::OrbY;
                case GameObjectType::PinkJumpRing: return Kind::OrbP;
                case GameObjectType::RedJumpRing: return Kind::OrbR;
                case GameObjectType::GravityRing: return Kind::OrbB;
                case GameObjectType::GreenRing: return Kind::OrbG;
                case GameObjectType::DropRing: return Kind::OrbK;
                case GameObjectType::CubePortal: return Kind::PCube;
                case GameObjectType::ShipPortal: return Kind::PShip;
                case GameObjectType::BallPortal: return Kind::PBall;
                case GameObjectType::UfoPortal: return Kind::PUfo;
                case GameObjectType::WavePortal: return Kind::PWave;
                case GameObjectType::RobotPortal: return Kind::PRobot;
                case GameObjectType::SpiderPortal: return Kind::PSpider;
                case GameObjectType::NormalGravityPortal: return Kind::GravN;
                case GameObjectType::InverseGravityPortal: return Kind::GravF;
                case GameObjectType::MiniSizePortal: return Kind::Mini;
                case GameObjectType::RegularSizePortal: return Kind::Big;
                default: return Kind::Count;  // ignored
            }
        }

        static float speedOf(int id) {
            switch (id) {
                case 200: return 251.16f;
                case 201: return 311.58f;
                case 202: return 387.42f;
                case 203: return 468.0f;
                default: return 576.0f;
            }
        }

        void ensureLevel(PlayLayer* layer) {
            std::size_t count = layer->m_objects ? layer->m_objects->count() : 0;
            if (m_cacheLayer == layer && m_cacheCount == count) return;
            m_cacheLayer = layer;
            m_cacheCount = count;
            m_solids.clear(); m_wide.clear(); m_haz.clear(); m_inter.clear();
            m_maxWs = 30.f; m_maxWh = 30.f; m_maxWi = 40.f;
            m_floorY = 105.f;
            m_ceilY = 405.f;
            if (!layer->m_objects) return;

            CCArrayExt<GameObject*> objects = layer->m_objects;
            for (auto* obj : objects) {
                if (!obj) continue;
                Kind k = classify(obj);
                if (k == Kind::Count) continue;

                auto r = obj->getObjectRect();
                Obj o;
                o.k = k;
                o.x0 = r.getMinX(); o.x1 = r.getMaxX();
                o.y0 = r.getMinY(); o.y1 = r.getMaxY();
                if (o.x1 - o.x0 < 0.01f || o.y1 - o.y0 < 0.01f) {
                    float x = obj->getPositionX(), y = obj->getPositionY();
                    float h = isOrb(k) || isPad(k) ? 18.f : 15.f;
                    o.x0 = x - h; o.x1 = x + h; o.y0 = y - h; o.y1 = y + h;
                }
                if (k == Kind::Speed) o.val = speedOf(obj->m_objectID);

                if (k == Kind::Slope) {
                    // corner holding the solid part: 0 BR, 1 BL, 2 TL, 3 TR
                    int c = 0;
                    if (obj->isFlipX()) c ^= 1;
                    if (obj->isFlipY()) c = 3 - c;
                    float rot = std::fmod(std::fmod(obj->getRotation(), 360.f) + 360.f, 360.f);
                    c = (c + static_cast<int>(std::lround(rot / 90.f))) % 4;
                    if (c <= 1) o.corner = c;       // top-facing: real slope
                    else o.k = Kind::Solid;         // bottom-facing: plain block
                }

                if (o.k == Kind::Solid || o.k == Kind::Slope) {
                    if (o.x1 - o.x0 > 150.f) m_wide.push_back(o);
                    else {
                        m_maxWs = std::max(m_maxWs, o.x1 - o.x0);
                        m_solids.push_back(o);
                    }
                } else if (o.k == Kind::Hazard) {
                    m_maxWh = std::max(m_maxWh, std::min(o.x1 - o.x0, 300.f));
                    m_haz.push_back(o);
                } else {
                    m_maxWi = std::max(m_maxWi, std::min(o.x1 - o.x0, 200.f));
                    m_inter.push_back(o);
                }
            }
            auto cmp = [](Obj const& a, Obj const& b) { return a.x0 < b.x0; };
            std::sort(m_solids.begin(), m_solids.end(), cmp);
            std::sort(m_haz.begin(), m_haz.end(), cmp);
            std::sort(m_inter.begin(), m_inter.end(), cmp);
            m_usedI.assign(m_inter.size(), 0);
            if (debug()) {
                log::info("AutoBot: {} solids, {} wide, {} hazards, {} interactives",
                          m_solids.size(), m_wide.size(), m_haz.size(), m_inter.size());
            }
        }

        template <class F>
        static void scan(std::vector<Obj> const& v, float maxW, float l, float r, F&& f) {
            auto it = std::lower_bound(v.begin(), v.end(), l - maxW,
                                       [](Obj const& o, float val) { return o.x0 < val; });
            for (; it != v.end() && it->x0 <= r; ++it) {
                if (it->x1 < l) continue;
                f(*it);
            }
        }

        template <class F>
        void forSolids(float l, float r, F&& f) const {
            scan(m_solids, m_maxWs, l, r, f);
            for (auto const& o : m_wide) {
                if (o.x1 >= l && o.x0 <= r) f(o);
            }
        }

        // ---- calibration & surface tracking -------------------------------------
        static void ema(float& v, float target, float a) { v += (target - v) * a; }

        void learn(Mode mode, bool mini, int grav, bool onGround, float prevVy) {
            Phys& P = m_ph[static_cast<int>(mode)][mini ? 1 : 0];
            float u = grav > 0 ? 1.f : -1.f;
            float a = (m_vy - prevVy) / m_step * u;  // acceleration relative to "up"
            bool airborne = !onGround && !m_prevOnGround && mode == m_prevMode;

            switch (mode) {
                case Mode::Cube: case Mode::Ball: case Mode::Ufo: case Mode::Robot: case Mode::Spider:
                    if (airborne && !m_pressed && a < -600.f && a > -6000.f &&
                        std::fabs(m_vy) < 850.f) {
                        ema(P.g, -a, 0.08f);
                    }
                    if ((mode == Mode::Cube || mode == Mode::Robot) && m_prevOnGround && !onGround &&
                        m_vy * u > 150.f && m_prevTouch == Kind::Solid) {
                        ema(P.jump, m_vy * u + P.g * m_step, 0.3f);
                    }
                    if (mode == Mode::Ufo && m_pressed && m_vy * u > 150.f && prevVy * u < m_vy * u - 100.f) {
                        ema(P.jump, m_vy * u, 0.3f);
                    }
                    break;
                case Mode::Ship:
                    if (airborne && std::fabs(m_vy) < P.capUp * 0.8f) {
                        if (m_pressed && a > 300.f && a < 6000.f) ema(P.upA, a, 0.08f);
                        else if (!m_pressed && a < -300.f && a > -6000.f) ema(P.downA, a, 0.08f);
                    }
                    break;
                case Mode::Wave:
                    if (m_speed > 150.f && std::fabs(m_vy) > 80.f) {
                        float sl = std::fabs(m_vy) / m_speed;
                        if (sl > 0.4f && sl < 2.5f) ema(P.slope, sl, 0.1f);
                    }
                    break;
            }
            if (mode == Mode::Ship || mode == Mode::Ufo) {
                if (m_vy * u > P.capUp) P.capUp = m_vy * u;
                if (-m_vy * u > P.capDown) P.capDown = -m_vy * u;
            }
            // pad / orb strength: a big jump in vy right after touching one
            if ((isPad(m_prevTouch) || isOrb(m_prevTouch)) && m_prevTouch != Kind::PadB &&
                m_prevTouch != Kind::OrbB && m_prevTouch != Kind::OrbG && m_prevTouch != Kind::OrbK &&
                mode != Mode::Wave) {
                if ((m_vy - prevVy) * u > 250.f) {
                    ema(m_imp[static_cast<int>(m_prevTouch)], m_vy * u + P.g * m_step, 0.4f);
                }
            }
        }

        bool solidEdgeNear(float x, float half, float y, bool top) const {
            bool found = false;
            forSolids(x - half, x + half, [&](Obj const& o) {
                if (found || o.x1 <= x - half + 1.f || o.x0 >= x + half - 1.f) return;
                float e = top ? o.y1 : o.y0;
                if (std::fabs(e - y) < 3.f) found = true;
            });
            return found;
        }

        void trackSurfaces(Mode mode, int grav, bool onGround, float cx, float cy, float prevVy) {
            bool mini = false;
            float half = halfOf(mode, mini);
            if (onGround && (mode == Mode::Cube || mode == Mode::Robot || mode == Mode::Spider ||
                             mode == Mode::Ball)) {
                if (grav > 0) {
                    float b = cy - half;
                    if (!solidEdgeNear(cx, half, b, true) && std::fabs(b - m_floorY) < 40.f) m_floorY = b;
                } else {
                    float t = cy + half;
                    if (!solidEdgeNear(cx, half, t, false) && t > m_floorY + 100.f) m_ceilY = t;
                }
            }
            // flying modes pinned against the ceiling
            float u = grav > 0 ? 1.f : -1.f;
            if ((mode == Mode::Ship || mode == Mode::Ufo || mode == Mode::Wave || mode == Mode::Ball) &&
                prevVy * u > 150.f && std::fabs(m_vy) < 15.f && cy + half > m_floorY + 120.f &&
                !solidEdgeNear(cx, half, cy + half, false)) {
                m_ceilY = cy + half;
            }
        }

        // pads/orbs the real player is already touching count as consumed
        void markTouches(float cx, float cy, Mode mode, bool mini) {
            float h = halfOf(mode, mini);
            m_prevTouch = Kind::Solid;
            scan(m_inter, m_maxWi, cx - h - kOrbPad, cx + h + kOrbPad, [&](Obj const& o) {
                if (isPortal(o.k)) return;
                if (o.y1 + kOrbPad < cy - h || o.y0 - kOrbPad > cy + h) return;
                std::size_t i = static_cast<std::size_t>(&o - m_inter.data());
                if (isPad(o.k)) {
                    if (!m_usedI[i]) m_prevTouch = o.k;
                    m_usedI[i] = 1;
                } else if (m_usedI[i] && m_prevTouch == Kind::Solid) {
                    m_prevTouch = o.k;
                }
            });
        }

        // ---- simulation ---------------------------------------------------------
        enum class Ev { None, Dead };

        bool windowUsed(Sim const& s, std::size_t idx) const {
            long rel = static_cast<long>(idx) - m_winBase;
            return rel >= 0 && rel < 64 && ((s.used >> rel) & 1ULL);
        }
        static void setUsed(Sim& s, long rel) {
            if (rel >= 0 && rel < 64) s.used |= (1ULL << rel);
        }

        void applyImpulse(Sim& s, Kind k) const {
            float u = s.grav > 0 ? 1.f : -1.f;
            bool wave = s.mode == Mode::Wave;
            switch (k) {
                case Kind::PadB: case Kind::OrbB:
                    s.grav = -s.grav;
                    s.vy = 0.f;
                    s.grounded = false;
                    break;
                case Kind::OrbG:
                    s.grav = -s.grav;
                    s.vy = m_imp[static_cast<int>(k)] * (s.grav > 0 ? 1.f : -1.f);
                    s.grounded = false;
                    break;
                case Kind::OrbK:
                    if (!wave) s.vy = -m_imp[static_cast<int>(k)] * u;
                    s.grounded = false;
                    break;
                default:
                    if (!wave) s.vy = m_imp[static_cast<int>(k)] * u;
                    s.grounded = false;
                    break;
            }
        }

        // press edge on an orb: returns true if an orb was used
        bool tryOrb(Sim& s, float half) const {
            bool used = false;
            scan(m_inter, m_maxWi, s.x - half - kOrbPad, s.x + half + kOrbPad, [&](Obj const& o) {
                if (used || !isOrb(o.k)) return;
                std::size_t i = static_cast<std::size_t>(&o - m_inter.data());
                if (windowUsed(s, i)) return;
                if (o.y1 + kOrbPad < s.y - half || o.y0 - kOrbPad > s.y + half) return;
                setUsed(s, static_cast<long>(i) - m_winBase);
                applyImpulse(s, o.k);
                used = true;
            });
            return used;
        }

        void touch(Sim& s, float half) const {
            scan(m_inter, m_maxWi, s.x - half, s.x + half, [&](Obj const& o) {
                if (isOrb(o.k)) return;
                if (o.y1 < s.y - half || o.y0 > s.y + half) return;
                std::size_t i = static_cast<std::size_t>(&o - m_inter.data());
                if (isPad(o.k)) {
                    if (windowUsed(s, i)) return;
                    setUsed(s, static_cast<long>(i) - m_winBase);
                    applyImpulse(s, o.k);
                    return;
                }
                Mode before = s.mode;
                switch (o.k) {
                    case Kind::PCube: s.mode = Mode::Cube; break;
                    case Kind::PShip: s.mode = Mode::Ship; break;
                    case Kind::PBall: s.mode = Mode::Ball; break;
                    case Kind::PUfo: s.mode = Mode::Ufo; break;
                    case Kind::PWave: s.mode = Mode::Wave; break;
                    case Kind::PRobot: s.mode = Mode::Robot; break;
                    case Kind::PSpider: s.mode = Mode::Spider; break;
                    case Kind::GravN:
                        if (s.grav != 1) { s.grav = 1; s.vy *= 0.5f; s.grounded = false; }
                        break;
                    case Kind::GravF:
                        if (s.grav != -1) { s.grav = -1; s.vy *= 0.5f; s.grounded = false; }
                        break;
                    case Kind::Mini: s.mini = true; break;
                    case Kind::Big: s.mini = false; break;
                    case Kind::Speed: s.speed = o.val; break;
                    default: break;
                }
                if (s.mode != before) s.grounded = false;
            });
        }

        // surface under (gravity side of) the cube: block top / slope / floor
        float surfaceUnder(Sim const& s, float half, float& slopeVy) const {
            float u = s.grav > 0 ? 1.f : -1.f;
            slopeVy = 0.f;
            float l = s.x - half + 2.f, r = s.x + half - 2.f;
            float feet = s.y - half * u;
            if (u > 0) {
                float best = m_floorY;
                forSolids(l, r, [&](Obj const& o) {
                    if (o.x1 < l || o.x0 > r) return;
                    float h = o.k == Kind::Slope ? slopeH(o, std::clamp(s.x, o.x0, o.x1)) : o.y1;
                    if (h <= feet + kLand && h > best) {
                        best = h;
                        slopeVy = (o.k == Kind::Slope && o.corner == 0)
                                      ? (o.y1 - o.y0) / std::max(o.x1 - o.x0, 1.f) * s.speed : 0.f;
                    }
                });
                return best;
            }
            float best = m_ceilY;
            forSolids(l, r, [&](Obj const& o) {
                if (o.x1 < l || o.x0 > r || o.k == Kind::Slope) return;
                if (o.y0 >= feet - kLand && o.y0 < best) best = o.y0;
            });
            return best;
        }

        void spiderTeleport(Sim& s, float half) const {
            float u = s.grav > 0 ? 1.f : -1.f;
            float l = s.x - half + 2.f, r = s.x + half - 2.f;
            float target;
            if (u > 0) {  // go up to the nearest underside
                target = m_ceilY;
                float head = s.y + half;
                forSolids(l, r, [&](Obj const& o) {
                    if (o.x1 < l || o.x0 > r || o.k == Kind::Slope) return;
                    if (o.y0 >= head - kLand && o.y0 < target) target = o.y0;
                });
                s.y = target - half;
            } else {
                target = m_floorY;
                float feet = s.y - half;
                forSolids(l, r, [&](Obj const& o) {
                    if (o.x1 < l || o.x0 > r) return;
                    float h = o.k == Kind::Slope ? slopeH(o, std::clamp(s.x, o.x0, o.x1)) : o.y1;
                    if (h <= feet + kLand && h > target) target = h;
                });
                s.y = target + half;
            }
            s.grav = -s.grav;
            s.vy = 0.f;
            s.grounded = true;
        }

        // One simulated sub-step of 1/120 s.
        Ev step(Sim& s, bool press) const {
            Phys const& P = m_ph[static_cast<int>(s.mode)][s.mini ? 1 : 0];
            float half = halfOf(s.mode, s.mini);
            bool edge = press && !s.prevPress;
            s.prevPress = press;
            float prevY = s.y;

            bool acted = edge ? tryOrb(s, half) : false;
            float u = s.grav > 0 ? 1.f : -1.f;
            if (!acted) {
                switch (s.mode) {
                    case Mode::Cube:
                        if (press && s.grounded) { s.vy = P.jump * u; s.grounded = false; }
                        break;
                    case Mode::Ball:
                        if (edge && s.grounded) { s.grav = -s.grav; u = -u; s.vy = 0.f; s.grounded = false; }
                        break;
                    case Mode::Ufo:
                        if (edge) { s.vy = P.jump * u; s.grounded = false; }
                        break;
                    case Mode::Robot:
                        if (press && s.grounded) {
                            s.vy = P.jump * u; s.grounded = false; s.robotT = 0.f;
                        }
                        break;
                    case Mode::Spider:
                        if (edge && s.grounded) { spiderTeleport(s, half); u = -u; }
                        break;
                    default: break;
                }
            }
            touch(s, half);
            u = s.grav > 0 ? 1.f : -1.f;
            half = halfOf(s.mode, s.mini);
            Phys const& Q = m_ph[static_cast<int>(s.mode)][s.mini ? 1 : 0];

            // vertical motion
            switch (s.mode) {
                case Mode::Ship: {
                    float a = press ? Q.upA : Q.downA;
                    s.vy += a * u * kSub;
                    s.vy = std::clamp(s.vy * u, -Q.capDown, Q.capUp) * u;
                    s.y += s.vy * kSub;
                    s.grounded = false;
                    break;
                }
                case Mode::Wave:
                    s.vy = (press ? 1.f : -1.f) * u * Q.slope * s.speed;
                    s.y += s.vy * kSub;
                    s.grounded = false;
                    break;
                default:
                    if (!s.grounded) {
                        float g = Q.g;
                        if (s.mode == Mode::Robot && press && s.robotT < 0.35f) g *= 0.3f;
                        s.robotT += kSub;
                        s.vy -= g * u * kSub;
                        if (-s.vy * u > Q.capDown) s.vy = -Q.capDown * u;
                        s.y += s.vy * kSub;
                    }
                    break;
            }
            s.x += s.speed * kSub;

            bool groundMode = s.mode == Mode::Cube || s.mode == Mode::Robot ||
                              s.mode == Mode::Spider || s.mode == Mode::Ball;
            if (s.grounded && groundMode) {
                float sv = 0.f;
                float sup = surfaceUnder(s, half, sv);
                float gap = u > 0 ? (s.y - half) - sup : sup - (s.y + half);
                if (gap > 1.5f) {  // rolled off
                    s.grounded = false;
                    s.vy = s.slopeVy;
                    s.slopeVy = 0.f;
                } else {
                    s.y = sup + half * u;
                    s.slopeVy = sv;
                }
            }

            return collide(s, prevY, half);
        }

        Ev collide(Sim& s, float prevY, float half) const {
            float u = s.grav > 0 ? 1.f : -1.f;
            bool bounded = s.mode != Mode::Cube && s.mode != Mode::Robot;

            if (s.y - half <= m_floorY && (u > 0 || bounded)) {
                s.y = m_floorY + half;
                if (s.vy < 0.f) s.vy = 0.f;
                if (u > 0) { s.grounded = true; s.slopeVy = 0.f; }
            }
            if (s.y + half >= m_ceilY && (u < 0 || bounded)) {
                s.y = m_ceilY - half;
                if (s.vy > 0.f) s.vy = 0.f;
                if (u < 0) { s.grounded = true; s.slopeVy = 0.f; }
            }

            bool dead = false;
            auto hit = [&](Obj const& o) {
                if (dead) return;
                float l = s.x - half, r = s.x + half, b = s.y - half, t = s.y + half;
                if (!(r > o.x0 && l < o.x1 && t > o.y0 && b < o.y1)) return;
                float prevB = prevY - half, prevT = prevY + half;

                if (o.k == Kind::Slope) {
                    if (u < 0) { dead = true; return; }
                    float h = slopeH(o, std::clamp(s.x, o.x0, o.x1));
                    if (b < h) {
                        if (prevB >= h - 10.f || s.grounded) {
                            if (s.vy <= 0.f || s.grounded) {
                                s.y = h + half;
                                s.vy = 0.f;
                                s.grounded = true;
                                s.slopeVy = o.corner == 0
                                    ? (o.y1 - o.y0) / std::max(o.x1 - o.x0, 1.f) * s.speed : 0.f;
                            }
                        } else {
                            dead = true;
                        }
                    }
                    return;
                }

                if (u > 0) {
                    if (prevB >= o.y1 - kLand) {
                        s.y = o.y1 + half; if (s.vy < 0.f) s.vy = 0.f;
                        s.grounded = true; s.slopeVy = 0.f;
                    } else if (prevT <= o.y0 + kLand) {
                        s.y = o.y0 - half; if (s.vy > 0.f) s.vy = 0.f;
                    } else {
                        dead = true;
                    }
                } else {
                    if (prevT <= o.y0 + kLand) {
                        s.y = o.y0 - half; if (s.vy > 0.f) s.vy = 0.f;
                        s.grounded = true; s.slopeVy = 0.f;
                    } else if (prevB >= o.y1 - kLand) {
                        s.y = o.y1 + half; if (s.vy < 0.f) s.vy = 0.f;
                    } else {
                        dead = true;
                    }
                }
            };
            forSolids(s.x - half, s.x + half, hit);
            if (dead) return Ev::Dead;

            float hh = std::max(half - 2.f, 4.f) + kSafe;
            float l = s.x - hh, r = s.x + hh, b = s.y - hh, t = s.y + hh;
            scan(m_haz, m_maxWh, l, r, [&](Obj const& o) {
                if (r > o.x0 && l < o.x1 && t > o.y0 && b < o.y1) dead = true;
            });
            return dead ? Ev::Dead : Ev::None;
        }

        // One decision = two sub-steps with the same input.
        bool advance(Sim& s, bool press) const {
            if (s.holdLeft > 0 && !press) return false;  // tap lock (caller avoids this)
            bool wasPressed = s.prevPress;
            for (int i = 0; i < 2; ++i) {
                if (step(s, press) == Ev::Dead) return false;
            }
            if (s.holdLeft > 0) --s.holdLeft;
            if (press && !wasPressed && s.mode == Mode::Cube) s.holdLeft = 1;
            return true;
        }

        std::uint64_t keyOf(Sim const& s, int d) const {
            std::uint64_t h = static_cast<std::uint64_t>(d) * 1469598103934665603ULL;
            auto mix = [&](std::int64_t v) {
                h ^= static_cast<std::uint64_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix(static_cast<std::int64_t>(std::floor(s.y / 3.f)));
            mix(static_cast<std::int64_t>(std::floor(s.vy / 20.f)));
            mix(static_cast<int>(s.mode) | (s.grav > 0 ? 8 : 0) | (s.mini ? 16 : 0) |
                (s.grounded ? 32 : 0) | (s.prevPress ? 64 : 0) | (s.holdLeft << 7) |
                (s.robotT > 0.35f ? 1024 : 0));
            mix(static_cast<std::int64_t>(s.used & 0xFFFFFFFFULL));
            mix(static_cast<std::int64_t>(s.speed));
            return h;
        }

        bool dfs(Sim const& s, int d, int& budget, std::unordered_set<std::uint64_t>& bad,
                 int& deepest) const {
            if (d > deepest) deepest = d;
            if (d >= kHorizon) return true;
            if (budget <= 0) return true;  // unknown: assume alive
            std::uint64_t key = keyOf(s, d);
            if (bad.count(key)) return false;

            bool first = false;
            if (s.mode == Mode::Ship || s.mode == Mode::Wave) first = s.prevPress;
            bool forced = s.holdLeft > 0;
            for (int i = 0; i < 2; ++i) {
                bool act = i == 0 ? first : !first;
                if (forced && !act) continue;
                Sim n = s;
                budget -= 2;
                if (!advance(n, act)) continue;
                if (dfs(n, d + 1, budget, bad, deepest)) return true;
            }
            bad.insert(key);
            return false;
        }

        bool plan(Mode mode, bool mini, int grav, bool onGround, float cx, float cy, float dt) {
            if (m_holdTimer > 0.f) m_holdTimer -= dt;

            Sim s;
            s.x = cx; s.y = cy;
            s.vy = onGround ? 0.f : m_vy;
            s.mode = mode; s.mini = mini; s.grav = grav;
            s.speed = m_speed;
            s.grounded = onGround;
            s.prevPress = m_pressed;
            s.holdLeft = m_holdTimer > 0.f ? static_cast<int>(std::ceil(m_holdTimer / kDec)) : 0;
            s.robotT = 0.f;

            // window of pads/orbs for the "already used" bitmask
            m_winBase = static_cast<int>(std::lower_bound(m_inter.begin(), m_inter.end(),
                cx - 120.f, [](Obj const& o, float v) { return o.x0 < v; }) - m_inter.begin());
            for (int i = 0; i < 64; ++i) {
                std::size_t idx = static_cast<std::size_t>(m_winBase + i);
                if (idx >= m_inter.size()) break;
                if (m_usedI[idx]) s.used |= (1ULL << i);
            }

            std::unordered_set<std::uint64_t> bad;
            bad.reserve(2048);
            int budget = kBudget;
            int deepest[2] = {0, 0};
            bool first = false;
            if (mode == Mode::Ship || mode == Mode::Wave) first = m_pressed;
            bool forced = s.holdLeft > 0;

            for (int i = 0; i < 2; ++i) {
                bool act = i == 0 ? first : !first;
                if (forced && !act) continue;
                Sim n = s;
                budget -= 2;
                if (!advance(n, act)) { deepest[act ? 1 : 0] = 0; continue; }
                int dp = 1;
                if (dfs(n, 1, budget, bad, dp)) {
                    m_usedBudget = kBudget - budget;
                    m_failStreak = 0;
                    m_plan = act ? "plan: pulsar" : "plan: soltar";
                    return act || (forced);
                }
                deepest[act ? 1 : 0] = dp;
            }

            m_usedBudget = kBudget - budget;
            // no surviving plan: do whatever lives longest
            ++m_failStreak;
            bool act;
            if (forced) act = true;
            else if (deepest[1] != deepest[0]) act = deepest[1] > deepest[0];
            else act = (mode == Mode::Cube || mode == Mode::Robot) ? onGround : m_pressed;
            m_plan = "sin plan (emergencia)";
            m_lastEvent = fmt::format("sin plan x={:.0f}", cx);
            return act;
        }

        // ---- HUD ------------------------------------------------------------------
        void hideHud(PlayLayer* layer) {
            if (!m_hudShown) return;
            if (auto node = layer->getChildByID("autobot-hud"_spr)) node->setVisible(false);
            m_hudShown = false;
        }

        void updateHud(PlayLayer* layer, bool botOn) {
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
                "{}  presupuesto: {}\n"
                "suelo: {:.0f}  techo: {:.0f}  {}\n"
                "{}",
                botOn ? "ON" : "OFF", m_modeName, m_speed, m_plan, m_usedBudget, m_floorY,
                m_ceilY, learnedInfo(), m_lastEvent);
            label->setString(txt.c_str());
            label->setVisible(true);
            m_hudShown = true;
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

    void resetLevel() {
        PlayLayer::resetLevel();
        autobot::Bot::get().onReset();
    }

    void onQuit() {
        autobot::Bot::get().invalidate();
        PlayLayer::onQuit();
    }
};

// ---------------------------------------------------------------------------
// AutoBot menu (popup)
// ---------------------------------------------------------------------------

class AutoBotMenu : public geode::Popup {
protected:
    bool init() {
        if (!Popup::init(300.f, 200.f)) return false;
        this->setTitle("AutoBot");

        struct RowInfo {
            char const* text;
            int id;
        };
        RowInfo rows[] = {
            {"AutoBot activado", autobot::OptEnabled},
            {"Texto en pantalla", autobot::OptHud},
        };

        float y = 40.f;
        for (auto& row : rows) {
            auto toggler = CCMenuItemToggler::createWithStandardSprites(
                this, menu_selector(AutoBotMenu::onToggle), 0.7f);
            toggler->setTag(row.id);
            toggler->toggle(autobot::getOpt(row.id));
            m_buttonMenu->addChildAtPosition(toggler, Anchor::Center, {-115.f, y});

            auto label = CCLabelBMFont::create(row.text, "bigFont.fnt");
            label->setScale(0.5f);
            label->setAnchorPoint({0.f, 0.5f});
            m_mainLayer->addChildAtPosition(label, Anchor::Center, {-90.f, y});
            y -= 36.f;
        }

        auto clearBtn = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("Reiniciar fisica"), nullptr, this, menu_selector(AutoBotMenu::onReset));
        clearBtn->setScale(0.7f);
        m_buttonMenu->addChildAtPosition(clearBtn, Anchor::Center, {0.f, -60.f});
        return true;
    }

    void onToggle(CCObject* sender) {
        int id = static_cast<CCNode*>(sender)->getTag();
        autobot::setOpt(id, !autobot::getOpt(id));
    }

    void onReset(CCObject*) { autobot::Bot::get().resetPhysics(); }

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
    log::info("AutoBot 0.5.0 loaded.");
}
