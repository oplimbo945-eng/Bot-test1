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

// AutoBot v0.6.0 - autoplayer for classic mode, all vehicles.
//
// How it decides (every frame, from the REAL state of the game):
//  1. The objects around the player are re-read LIVE from the game each frame
//     (position, "no touch", toggled-off groups, moved objects), so fake spikes,
//     fake blocks and move/toggle triggers cannot fool it.
//  2. A physics model of every game mode simulates ~0.8 s ahead at 240 Hz and a
//     memoised search tries "press / release" at every 1/60 s step. The first
//     input of a surviving sequence is used. Nothing is a fixed "jump here" rule.
//  3. Physics constants (jump, gravity, ship acceleration, wave slope, pad / orb
//     strength, dash speed, hitbox size, velocity scale) are calibrated online
//     from the real player, using the exact velocity the game keeps.
//  4. The HUD shows the model's prediction error, so a wrong constant is visible.
//
// Not supported: dual mode (only player 1 is planned, player 2 gets the same
// input), teleport portals, custom triggers that move the ground.

namespace autobot {
    constexpr float kDec = 1.f / 60.f;     // one decision = one 60 Hz frame
    constexpr float kSub = 1.f / 240.f;    // simulation sub-step
    constexpr int kSubPerDec = 4;
    constexpr int kHorizon = 48;           // decisions looked ahead (0.8 s)
    constexpr int kBudget = 16000;         // max sub-steps per frame
    constexpr float kLand = 8.f;           // how deep a corner may clip and still count as landing
    constexpr float kSafe = 1.0f;          // extra margin on hazards
    constexpr float kFlyPad = 3.f;         // extra margin for flying modes against blocks
    constexpr float kDefaultSpeed = 311.58f;
    constexpr float kMinHold = 0.035f;     // cube: minimum hold time of a tap (s)
    constexpr float kPi = 3.14159265f;

    enum Opt : int { OptHud = 0, OptEnabled = 3 };

    inline bool getOpt(int o) {
        if (o == OptHud) return Mod::get()->getSavedValue<bool>("opt_hud", true);
        return Mod::get()->getSettingValue<bool>("enabled");
    }
    inline void setOpt(int o, bool v) {
        if (o == OptHud) Mod::get()->setSavedValue<bool>("opt_hud", v);
        else Mod::get()->setSettingValue<bool>("enabled", v);
    }

    enum class Mode : int { Cube = 0, Ship, Ball, Ufo, Wave, Robot, Spider, Swing };

    enum class Kind : int {
        Solid, Hazard, Slope,
        PadY, PadP, PadR, PadB,
        OrbY, OrbP, OrbR, OrbB, OrbG, OrbK, OrbD, OrbDG,
        PCube, PShip, PBall, PUfo, PWave, PRobot, PSpider, PSwing,
        GravN, GravF, Mini, Big, Speed,
        Count
    };

    inline bool isOrb(Kind k) { return k >= Kind::OrbY && k <= Kind::OrbDG; }
    inline bool isPad(Kind k) { return k >= Kind::PadY && k <= Kind::PadB; }
    inline bool isPortal(Kind k) { return k >= Kind::PCube && k <= Kind::Speed; }

    struct Obj {
        float x0 = 0, x1 = 0, y0 = 0, y1 = 0;
        Kind k = Kind::Solid;
        float val = 0.f;      // speed portals: units/s;  dash orbs: direction x
        float val2 = 0.f;     // dash orbs: direction y
        int corner = 0;       // slopes: 0 rising (BR solid), 1 falling (BL solid)
        float rad = 0.f;      // orb activation radius
        int id = 0;           // stable id (index into the used-flags)
    };

    struct Entry {
        GameObject* o = nullptr;
        float ox = 0.f;
        Kind k = Kind::Solid;
        int id = 0;
    };

    struct Phys {
        float g = 3449.5f;     // gravity (units/s^2)
        float jump = 670.8f;   // jump / ufo / robot impulse
        float upA = 2300.f;    // ship: acceleration while holding
        float downA = -2500.f; // ship: acceleration while released
        float capUp = 540.f;   // speed caps
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
        bool dashing = false;
        float dirX = 1.f, dirY = 0.f;
    };

    inline float rectCircleDist2(float l, float r, float b, float t, float cx, float cy) {
        float dx = std::max({l - cx, 0.f, cx - r});
        float dy = std::max({b - cy, 0.f, cy - t});
        return dx * dx + dy * dy;
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
            m_entries.clear(); m_wideEntries.clear(); m_usedG.clear();
            m_solids.clear(); m_wide.clear(); m_haz.clear(); m_inter.clear();
            m_hasLast = false;
            m_floorY = 105.f;
            m_ceilY = 405.f;
            m_pressed = false;
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
            ensureIndex(layer);

            float px = p->getPositionX();
            float py = p->getPositionY();
            float prevVy = m_vy;
            bool dashing = p->m_isDashing;
            if (!m_hasLast || px < m_lastX - 40.f) {
                resetRun();
            } else {
                measure(px - m_lastX, py - m_lastY, dt);
            }
            m_lastX = px;
            m_lastY = py;
            bool hadLast = m_hasLast;
            m_hasLast = true;

            // exact vertical velocity from the game, scaled by a learned factor
            int ub = p->m_isUpsideDown ? 1 : 0;
            double rel = p->m_yVelocity;
            if (!dashing && std::fabs(rel) > 0.5 && std::fabs(m_vyPos) > 60.f && std::fabs(m_vyPos) < 1500.f) {
                float r = m_vyPos / static_cast<float>(rel);
                if (std::fabs(r) > 20.f && std::fabs(r) < 200.f) {
                    ema(m_vs[ub], r, 0.05f);
                    ++m_vsN[ub];
                }
            }
            m_vy = (m_vsN[ub] >= 20 && !dashing) ? static_cast<float>(rel) * m_vs[ub] : m_vyPos;

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
            else if (p->m_isSwing) mode = Mode::Swing;
            int grav = p->m_isUpsideDown ? -1 : 1;
            bool mini = p->getScale() < 0.8f;
            m_modeName = kModeNames[static_cast<int>(mode)];

            // hitbox learned from the real player
            float wHalf = (pr.getMaxX() - pr.getMinX()) * 0.5f;
            if (wHalf > 4.f && wHalf < 20.f) ema(m_half[static_cast<int>(mode)][mini ? 1 : 0], wHalf, 0.05f);

            m_dual = layer->m_player2 && layer->m_player2->isVisible();

            if (hadLast) {
                compareModel(mode, pcx, pcy, dt);
                learn(mode, mini, grav, onGround, prevVy, dashing, px, py);
            }
            buildWindow(pcx);
            trackSurfaces(mode, mini, grav, onGround, pcx, pcy, prevVy);
            markTouches(pcx, pcy, mode, mini);
            m_prevDashing = dashing;

            if (botOn) {
                bool want = plan(mode, mini, grav, onGround, pcx, pcy, dt, dashing);
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
        static constexpr char const* kModeNames[8] = {
            "cubo", "nave", "bola", "ufo", "wave", "robot", "spider", "swing"};

        // static index (by original x) + live window of the objects near the player
        std::vector<Entry> m_entries, m_wideEntries;
        std::vector<char> m_usedG;
        std::vector<Obj> m_solids, m_wide, m_haz, m_inter;
        float m_maxWs = 30.f, m_maxWh = 30.f, m_maxWi = 40.f;
        PlayLayer* m_cacheLayer = nullptr;
        std::size_t m_cacheCount = 0;

        Phys m_ph[8][2];
        float m_half[8][2];
        float m_imp[static_cast<int>(Kind::Count)] = {};
        float m_vs[2] = {60.f, -60.f};
        int m_vsN[2] = {0, 0};
        float m_dashD = 0.f;       // learned dash speed (0 = unknown)
        float m_dashExit = 0.4f;   // fraction of the dash vy kept when it ends

        float m_floorY = 105.f;
        float m_ceilY = 405.f;

        bool m_hasLast = false;
        bool m_prevOnGround = true;
        bool m_prevDashing = false;
        bool m_dual = false;
        Mode m_prevMode = Mode::Cube;
        float m_lastX = 0.f, m_lastY = 0.f;
        float m_speed = kDefaultSpeed;
        float m_vy = 0.f, m_vyPos = 0.f, m_vxPos = 0.f;
        float m_lastDashVy = 0.f;
        float m_step = 1.f / 60.f;

        bool m_pressed = false;
        float m_holdTimer = 0.f;
        int m_winBase = 0;
        Kind m_prevTouch = Kind::Solid;

        // model prediction check
        bool m_hasPred = false;
        float m_predY = 0.f, m_predVy = 0.f;
        float m_errY = 0.f, m_errVy = 0.f;

        char const* m_modeName = "-";
        char const* m_plan = "-";
        int m_failStreak = 0;
        int m_usedBudget = 0;

        bool m_hudShown = false;
        int m_hudTick = 0;
        std::string m_lastEvent = "-";

        static bool debug() { return Mod::get()->getSettingValue<bool>("debug_log"); }
        static void ema(float& v, float target, float a) { v += (target - v) * a; }

        // ---- defaults ---------------------------------------------------------
        void initPhys() {
            for (int m = 0; m < 8; ++m) {
                for (int s = 0; s < 2; ++s) {
                    m_ph[m][s] = Phys{};
                    float h = (m == static_cast<int>(Mode::Wave)) ? 10.f : 15.f;
                    m_half[m][s] = s ? h * 0.6f : h;
                }
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
            set(Mode::Swing, 3200.f, 0.f);
            m_ph[0][1].jump = 600.f;
            for (int s = 0; s < 2; ++s) {
                auto& sh = m_ph[static_cast<int>(Mode::Ship)][s];
                sh.upA = 2300.f; sh.downA = -2500.f; sh.capUp = 540.f; sh.capDown = 480.f;
                m_ph[static_cast<int>(Mode::Ufo)][s].capUp = 600.f;
                m_ph[static_cast<int>(Mode::Ufo)][s].capDown = 700.f;
                m_ph[static_cast<int>(Mode::Swing)][s].capUp = 600.f;
                m_ph[static_cast<int>(Mode::Swing)][s].capDown = 600.f;
            }
            m_imp[static_cast<int>(Kind::PadY)] = 864.f;
            m_imp[static_cast<int>(Kind::PadP)] = 540.f;
            m_imp[static_cast<int>(Kind::PadR)] = 1230.f;
            m_imp[static_cast<int>(Kind::OrbY)] = 670.8f;
            m_imp[static_cast<int>(Kind::OrbP)] = 470.f;
            m_imp[static_cast<int>(Kind::OrbR)] = 970.f;
            m_imp[static_cast<int>(Kind::OrbG)] = 540.f;
            m_imp[static_cast<int>(Kind::OrbK)] = 1100.f;
            m_dashD = 0.f;
            m_dashExit = 0.4f;
            m_vs[0] = 60.f; m_vs[1] = -60.f;
            m_vsN[0] = m_vsN[1] = 0;
        }

        void resetRun() {
            m_pressed = false;
            m_holdTimer = 0.f;
            m_vy = 0.f; m_vyPos = 0.f;
            m_failStreak = 0;
            m_prevTouch = Kind::Solid;
            m_hasPred = false;
            m_prevDashing = false;
            std::fill(m_usedG.begin(), m_usedG.end(), 0);
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
            m_vxPos = vx;
            m_vyPos = dy / m_step;
        }

        void setPressed(PlayLayer* layer, bool want, float cx = 0.f, float cy = 0.f,
                        Mode mode = Mode::Cube, bool mini = false) {
            if (want == m_pressed) return;
            if (want) {
                // an orb under the player is consumed by this press
                float h = m_half[static_cast<int>(mode)][mini ? 1 : 0];
                for (auto const& o : m_inter) {
                    if (!isOrb(o.k) || m_usedG[static_cast<std::size_t>(o.id)]) continue;
                    if (orbHit(o, cx, cy, h)) {
                        m_usedG[static_cast<std::size_t>(o.id)] = 1;
                        break;
                    }
                }
                if (mode == Mode::Cube || mode == Mode::Robot) m_holdTimer = kMinHold;
            }
            m_pressed = want;
            layer->handleButton(want, 1, true);
            if (m_dual && layer->m_player2) layer->handleButton(want, 1, false);
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
                case GameObjectType::DashRing: return Kind::OrbD;
                case GameObjectType::GravityDashRing: return Kind::OrbDG;
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

        void ensureIndex(PlayLayer* layer) {
            std::size_t count = layer->m_objects ? layer->m_objects->count() : 0;
            if (m_cacheLayer == layer && m_cacheCount == count) return;
            m_cacheLayer = layer;
            m_cacheCount = count;
            m_entries.clear(); m_wideEntries.clear();
            m_floorY = 105.f;
            m_ceilY = 405.f;
            int nextId = 0;
            if (layer->m_objects) {
                CCArrayExt<GameObject*> objects = layer->m_objects;
                for (auto* obj : objects) {
                    if (!obj) continue;
                    Kind k = classify(obj);
                    if (k == Kind::Count) continue;
                    auto r = obj->getObjectRect();
                    Entry e;
                    e.o = obj;
                    e.k = k;
                    e.ox = r.getMinX();
                    e.id = nextId++;
                    if (r.getMaxX() - r.getMinX() > 150.f && (k == Kind::Solid || k == Kind::Slope)) {
                        m_wideEntries.push_back(e);
                    } else {
                        m_entries.push_back(e);
                    }
                }
            }
            std::sort(m_entries.begin(), m_entries.end(),
                      [](Entry const& a, Entry const& b) { return a.ox < b.ox; });
            m_usedG.assign(static_cast<std::size_t>(nextId), 0);
            if (debug()) log::info("AutoBot: indexed {} objects", nextId);
        }

        // Live read of one object. Returns false if it cannot hurt / help the player now.
        bool makeObj(Entry const& e, Obj& out) const {
            GameObject* o = e.o;
            if (o->m_isNoTouch || o->m_isGroupDisabled) return false;
            auto r = o->getObjectRect();
            out = Obj{};
            out.k = e.k;
            out.id = e.id;
            out.x0 = r.getMinX(); out.x1 = r.getMaxX();
            out.y0 = r.getMinY(); out.y1 = r.getMaxY();
            if (out.x1 - out.x0 < 0.01f || out.y1 - out.y0 < 0.01f) {
                float x = o->getPositionX(), y = o->getPositionY();
                float h = (isOrb(e.k) || isPad(e.k)) ? 18.f : 15.f;
                out.x0 = x - h; out.x1 = x + h; out.y0 = y - h; out.y1 = y + h;
            }
            switch (e.k) {
                case Kind::Speed: out.val = speedOf(o->m_objectID); break;
                case Kind::Slope:
                    if (o->m_slopeIsHazard) out.k = Kind::Hazard;
                    else if (o->slopeFloorTop()) out.corner = o->m_slopeUphill ? 0 : 1;
                    else out.k = Kind::Solid;
                    break;
                case Kind::OrbD: case Kind::OrbDG: {
                    float a = -o->getRotation() * kPi / 180.f;
                    out.val = std::cos(a);
                    out.val2 = std::sin(a);
                    if (out.val < 0.f) { out.val = -out.val; out.val2 = -out.val2; }
                    out.rad = o->getObjectRadius();
                    break;
                }
                default:
                    if (isOrb(e.k)) out.rad = o->getObjectRadius();
                    break;
            }
            return true;
        }

        void buildWindow(float px) {
            m_solids.clear(); m_wide.clear(); m_haz.clear(); m_inter.clear();
            float lo = px - 250.f;
            float hi = px + m_speed * (kHorizon * kDec) + 700.f;
            auto it = std::lower_bound(m_entries.begin(), m_entries.end(), lo - 350.f,
                                       [](Entry const& e, float v) { return e.ox < v; });
            float maxWs = 30.f, maxWh = 30.f, maxWi = 40.f;
            Obj o;
            for (; it != m_entries.end() && it->ox <= hi; ++it) {
                if (!makeObj(*it, o)) continue;
                if (o.x1 < lo || o.x0 > hi) continue;
                switch (o.k) {
                    case Kind::Solid: case Kind::Slope:
                        maxWs = std::max(maxWs, o.x1 - o.x0);
                        m_solids.push_back(o);
                        break;
                    case Kind::Hazard:
                        maxWh = std::max(maxWh, o.x1 - o.x0);
                        m_haz.push_back(o);
                        break;
                    default:
                        maxWi = std::max(maxWi, o.x1 - o.x0);
                        m_inter.push_back(o);
                        break;
                }
            }
            for (auto const& e : m_wideEntries) {
                if (makeObj(e, o)) m_wide.push_back(o);
            }
            auto cmp = [](Obj const& a, Obj const& b) { return a.x0 < b.x0; };
            std::sort(m_solids.begin(), m_solids.end(), cmp);
            std::sort(m_haz.begin(), m_haz.end(), cmp);
            std::sort(m_inter.begin(), m_inter.end(), cmp);
            m_maxWs = maxWs; m_maxWh = maxWh; m_maxWi = maxWi;
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

        static bool orbHit(Obj const& o, float cx, float cy, float half) {
            float ocx = (o.x0 + o.x1) * 0.5f, ocy = (o.y0 + o.y1) * 0.5f;
            float rad = std::max(o.rad, 18.f) + 4.f;
            return rectCircleDist2(cx - half, cx + half, cy - half, cy + half, ocx, ocy) <= rad * rad;
        }

        // ---- calibration & surface tracking -------------------------------------
        void learn(Mode mode, bool mini, int grav, bool onGround, float prevVy, bool dashing,
                   float px, float py) {
            (void)px; (void)py;
            Phys& P = m_ph[static_cast<int>(mode)][mini ? 1 : 0];
            float u = grav > 0 ? 1.f : -1.f;
            float a = (m_vy - prevVy) / m_step * u;  // acceleration relative to "up"
            bool airborne = !onGround && !m_prevOnGround && mode == m_prevMode && !dashing && !m_prevDashing;

            // dash: speed while dashing, and the vy kept when it ends
            if (dashing) {
                float v = std::sqrt(m_vxPos * m_vxPos + m_vyPos * m_vyPos);
                if (v > 150.f && v < 1500.f) {
                    m_dashD = m_dashD <= 0.f ? v : m_dashD * 0.8f + v * 0.2f;
                    m_lastDashVy = m_vyPos;
                }
            } else if (m_prevDashing && std::fabs(m_lastDashVy) > 80.f) {
                float k = m_vyPos / m_lastDashVy;
                if (k > -0.5f && k < 1.5f) ema(m_dashExit, k, 0.3f);
            }

            switch (mode) {
                case Mode::Cube: case Mode::Ball: case Mode::Ufo: case Mode::Robot:
                case Mode::Spider: case Mode::Swing:
                    if (airborne && !m_pressed && a < -600.f && a > -6000.f && std::fabs(m_vy) < 850.f) {
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
                        if (m_pressed && a > 300.f && a < 8000.f) ema(P.upA, a, 0.1f);
                        else if (!m_pressed && a < -300.f && a > -8000.f) ema(P.downA, a, 0.1f);
                    }
                    break;
                case Mode::Wave:
                    if (m_speed > 150.f && std::fabs(m_vy) > 80.f) {
                        float sl = std::fabs(m_vy) / m_speed;
                        if (sl > 0.4f && sl < 2.5f) ema(P.slope, sl, 0.1f);
                    }
                    break;
            }
            if (mode == Mode::Ship || mode == Mode::Ufo || mode == Mode::Swing) {
                if (m_vy * u > P.capUp) P.capUp = m_vy * u;
                if (-m_vy * u > P.capDown) P.capDown = -m_vy * u;
            }
            // pad / orb strength: a big jump in vy right after touching one
            if ((isPad(m_prevTouch) || isOrb(m_prevTouch)) && m_prevTouch != Kind::PadB &&
                m_prevTouch != Kind::OrbB && m_prevTouch != Kind::OrbG && m_prevTouch != Kind::OrbK &&
                m_prevTouch != Kind::OrbD && m_prevTouch != Kind::OrbDG && mode != Mode::Wave) {
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

        void trackSurfaces(Mode mode, bool mini, int grav, bool onGround, float cx, float cy,
                           float prevVy) {
            float half = m_half[static_cast<int>(mode)][mini ? 1 : 0];
            bool groundish = mode == Mode::Cube || mode == Mode::Robot || mode == Mode::Spider ||
                             mode == Mode::Ball;
            if (onGround && groundish) {
                if (grav > 0) {
                    float b = cy - half;
                    if (!solidEdgeNear(cx, half, b, true) && std::fabs(b - m_floorY) < 40.f) m_floorY = b;
                } else {
                    float t = cy + half;
                    if (!solidEdgeNear(cx, half, t, false) && t > m_floorY + 100.f) m_ceilY = t;
                }
            }
            // flying modes pinned against the floor / ceiling (world velocity from positions)
            bool flying = mode == Mode::Ship || mode == Mode::Ufo || mode == Mode::Wave ||
                          mode == Mode::Ball || mode == Mode::Swing;
            if (flying && !m_prevDashing) {
                if (prevVy > 150.f && std::fabs(m_vyPos) < 15.f && cy + half > m_floorY + 120.f &&
                    !solidEdgeNear(cx, half, cy + half, false)) {
                    m_ceilY = cy + half;
                }
                if (prevVy < -150.f && std::fabs(m_vyPos) < 15.f && cy - half < m_ceilY - 120.f &&
                    !solidEdgeNear(cx, half, cy - half, true) && cy - half > 0.f) {
                    m_floorY = cy - half;
                }
            }
        }

        // pads/orbs the real player is already touching count as consumed
        void markTouches(float cx, float cy, Mode mode, bool mini) {
            float h = m_half[static_cast<int>(mode)][mini ? 1 : 0];
            m_prevTouch = Kind::Solid;
            scan(m_inter, m_maxWi, cx - h - 40.f, cx + h + 40.f, [&](Obj const& o) {
                if (isPortal(o.k)) return;
                auto idx = static_cast<std::size_t>(o.id);
                if (isPad(o.k)) {
                    if (o.y1 < cy - h || o.y0 > cy + h || o.x1 < cx - h || o.x0 > cx + h) return;
                    if (!m_usedG[idx]) m_prevTouch = o.k;
                    m_usedG[idx] = 1;
                } else if (orbHit(o, cx, cy, h) && m_usedG[idx] && m_prevTouch == Kind::Solid) {
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
            scan(m_inter, m_maxWi, s.x - half - 40.f, s.x + half + 40.f, [&](Obj const& o) {
                if (used || !isOrb(o.k)) return;
                std::size_t i = static_cast<std::size_t>(&o - m_inter.data());
                if (windowUsed(s, i)) return;
                if (!orbHit(o, s.x, s.y, half)) return;
                setUsed(s, static_cast<long>(i) - m_winBase);
                if (o.k == Kind::OrbD || o.k == Kind::OrbDG) {
                    s.dashing = true;
                    s.dirX = o.val; s.dirY = o.val2;
                    if (o.k == Kind::OrbDG) s.grav = -s.grav;
                    s.grounded = false;
                } else {
                    applyImpulse(s, o.k);
                }
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
                    case Kind::PSwing: s.mode = Mode::Swing; break;
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

        float halfOf(Mode m, bool mini) const { return m_half[static_cast<int>(m)][mini ? 1 : 0]; }

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
            if (u > 0) {
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

        // One simulated sub-step of 1/240 s.
        Ev step(Sim& s, bool press) const {
            Phys const& P = m_ph[static_cast<int>(s.mode)][s.mini ? 1 : 0];
            float half = halfOf(s.mode, s.mini);
            bool edge = press && !s.prevPress;
            s.prevPress = press;
            float prevY = s.y;

            // dash ends when the button is released
            if (s.dashing && !press) {
                s.dashing = false;
                float dv = (m_dashD > 0.f ? m_dashD : s.speed) * s.dirY;
                s.vy = dv * m_dashExit;
            }

            bool acted = edge ? tryOrb(s, half) : false;
            float u = s.grav > 0 ? 1.f : -1.f;
            if (!acted && !s.dashing) {
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
                    case Mode::Swing:
                        if (edge) { s.grav = -s.grav; u = -u; s.vy *= 0.3f; }
                        break;
                    default: break;
                }
            }
            touch(s, half);
            u = s.grav > 0 ? 1.f : -1.f;
            half = halfOf(s.mode, s.mini);
            Phys const& Q = m_ph[static_cast<int>(s.mode)][s.mini ? 1 : 0];

            if (s.dashing) {
                float d = m_dashD > 0.f ? m_dashD : s.speed;
                s.x += d * s.dirX * kSub;
                s.y += d * s.dirY * kSub;
                s.vy = d * s.dirY;
                s.grounded = false;
            } else {
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
                    case Mode::Swing: {
                        s.vy -= Q.g * u * kSub;
                        s.vy = std::clamp(s.vy * u, -Q.capDown, Q.capUp) * u;
                        s.y += s.vy * kSub;
                        s.grounded = false;
                        break;
                    }
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
            }

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
            bool flying = s.mode == Mode::Ship || s.mode == Mode::Wave || s.mode == Mode::Ufo ||
                          s.mode == Mode::Swing;
            float ch = half + (flying ? kFlyPad : 0.f);  // collision half-size

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
                float l = s.x - ch, r = s.x + ch, b = s.y - ch, t = s.y + ch;
                if (!(r > o.x0 && l < o.x1 && t > o.y0 && b < o.y1)) return;
                float prevB = prevY - ch, prevT = prevY + ch;

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
                        s.y = o.y1 + ch; if (s.vy < 0.f) s.vy = 0.f;
                        s.grounded = true; s.slopeVy = 0.f;
                    } else if (prevT <= o.y0 + kLand) {
                        s.y = o.y0 - ch; if (s.vy > 0.f) s.vy = 0.f;
                    } else {
                        dead = true;
                    }
                } else {
                    if (prevT <= o.y0 + kLand) {
                        s.y = o.y0 - ch; if (s.vy > 0.f) s.vy = 0.f;
                        s.grounded = true; s.slopeVy = 0.f;
                    } else if (prevB >= o.y1 - kLand) {
                        s.y = o.y1 + ch; if (s.vy < 0.f) s.vy = 0.f;
                    } else {
                        dead = true;
                    }
                }
            };
            forSolids(s.x - ch, s.x + ch, hit);
            if (dead) return Ev::Dead;

            float hh = std::max(half - 2.f, 4.f) + kSafe;
            float l = s.x - hh, r = s.x + hh, b = s.y - hh, t = s.y + hh;
            scan(m_haz, m_maxWh, l, r, [&](Obj const& o) {
                if (r > o.x0 && l < o.x1 && t > o.y0 && b < o.y1) dead = true;
            });
            return dead ? Ev::Dead : Ev::None;
        }

        // One decision = kSubPerDec sub-steps with the same input.
        bool advance(Sim& s, bool press) const {
            if (s.holdLeft > 0 && !press) return false;  // tap lock (caller avoids this)
            bool wasPressed = s.prevPress;
            for (int i = 0; i < kSubPerDec; ++i) {
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
            mix(static_cast<int>(s.mode) | (s.grav > 0 ? 16 : 0) | (s.mini ? 32 : 0) |
                (s.grounded ? 64 : 0) | (s.prevPress ? 128 : 0) | (s.holdLeft << 8) |
                (s.robotT > 0.35f ? 2048 : 0) | (s.dashing ? 4096 : 0));
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
            if (s.mode == Mode::Ship || s.mode == Mode::Wave || s.dashing) first = s.prevPress;
            bool forced = s.holdLeft > 0;
            for (int i = 0; i < 2; ++i) {
                bool act = i == 0 ? first : !first;
                if (forced && !act) continue;
                Sim n = s;
                budget -= kSubPerDec;
                if (!advance(n, act)) continue;
                if (dfs(n, d + 1, budget, bad, deepest)) return true;
            }
            bad.insert(key);
            return false;
        }

        bool plan(Mode mode, bool mini, int grav, bool onGround, float cx, float cy, float dt,
                  bool dashing) {
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
            if (dashing) {
                float v = std::sqrt(m_vxPos * m_vxPos + m_vyPos * m_vyPos);
                s.dashing = v > 1.f;
                if (s.dashing) { s.dirX = m_vxPos / v; s.dirY = m_vyPos / v; }
            }

            // window of pads/orbs for the "already used" bitmask
            m_winBase = static_cast<int>(std::lower_bound(m_inter.begin(), m_inter.end(),
                cx - 160.f, [](Obj const& o, float v) { return o.x0 < v; }) - m_inter.begin());
            for (int i = 0; i < 64; ++i) {
                std::size_t idx = static_cast<std::size_t>(m_winBase + i);
                if (idx >= m_inter.size()) break;
                if (m_usedG[static_cast<std::size_t>(m_inter[idx].id)]) s.used |= (1ULL << i);
            }

            std::unordered_set<std::uint64_t> bad;
            bad.reserve(2048);
            int budget = kBudget;
            int deepest[2] = {0, 0};
            bool first = false;
            if (mode == Mode::Ship || mode == Mode::Wave || s.dashing) first = m_pressed;
            bool forced = s.holdLeft > 0;

            for (int i = 0; i < 2; ++i) {
                bool act = i == 0 ? first : !first;
                if (forced && !act) continue;
                Sim n = s;
                budget -= kSubPerDec;
                if (!advance(n, act)) { deepest[act ? 1 : 0] = 0; continue; }
                int dp = 1;
                if (dfs(n, 1, budget, bad, dp)) {
                    m_usedBudget = kBudget - budget;
                    m_failStreak = 0;
                    m_plan = act ? "plan: pulsar" : "plan: soltar";
                    storePrediction(s, act);
                    return act;
                }
                deepest[act ? 1 : 0] = dp;
            }

            m_usedBudget = kBudget - budget;
            ++m_failStreak;
            bool act;
            if (forced) act = true;
            else if (deepest[1] != deepest[0]) act = deepest[1] > deepest[0];
            else act = (mode == Mode::Cube || mode == Mode::Robot) ? onGround : m_pressed;
            m_plan = "sin plan (emergencia)";
            m_lastEvent = fmt::format("sin plan x={:.0f}", cx);
            storePrediction(s, act);
            return act;
        }

        // ---- model check: where did the model say we would be one frame later? ----
        void storePrediction(Sim s, bool act) {
            m_hasPred = false;
            for (int i = 0; i < kSubPerDec; ++i) {
                if (step(s, act) == Ev::Dead) return;
            }
            m_predY = s.y;
            m_predVy = s.vy;
            m_hasPred = true;
        }

        void compareModel(Mode mode, float cx, float cy, float dt) {
            (void)mode; (void)cx;
            if (!m_hasPred) return;
            m_hasPred = false;
            if (dt < 0.0150f || dt > 0.0185f) return;  // only frames of ~60 fps
            ema(m_errY, std::fabs(cy - m_predY), 0.1f);
            ema(m_errVy, std::fabs(m_vy - m_predVy), 0.1f);
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
            auto const& c = m_ph[0][0];
            auto const& sh = m_ph[1][0];
            std::string txt = fmt::format(
                "AutoBot {}  modo: {}{}  vel: {:.0f}\n"
                "{}  nodos: {}\n"
                "error modelo: y {:.1f}  vy {:.0f}   suelo {:.0f} techo {:.0f}\n"
                "cubo salto {:.0f} g {:.0f} | nave {:.0f}/{:.0f}\n"
                "{}",
                botOn ? "ON" : "OFF", m_modeName, m_dual ? " (dual)" : "", m_speed, m_plan,
                m_usedBudget, m_errY, m_errVy, m_floorY, m_ceilY, c.jump, c.g, sh.upA, sh.downA,
                m_lastEvent);
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
    log::info("AutoBot 0.6.0 loaded.");
}
EOF
python3 - << 'PY'
s=open('/mnt/user-data/outputs/main.cpp').read()
print(s.count('{'), s.count('}'), s.count('('), s.count(')'), s.count('\n'))
PY
