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
     
