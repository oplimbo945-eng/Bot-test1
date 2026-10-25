#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/ui/Popup.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace geode::prelude;

// AutoBot v0.3.1 - reactive autoplayer for classic mode, cube and ship only.
//
// Phase 1: pause-menu access + ground-checked cube jumps.
// Phase 2: jump orbs (yellow/pink/red) + self-calibrating cube jump timing.
// Phase 3: real menu (popup) with options, on-screen text (HUD), and a
//          recorder that stores WHEN the player presses jump relative to the
//          obstacles. The bot can reuse that timing ("learn" option).
//
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
    constexpr std::size_t kMaxSamples = 80;
    constexpr std::size_t kMinSamplesToLearn = 5;

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
    };

    struct Orb {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
        bool used = false;
    };

    struct Pending {
        float t = 0.f;     // seconds between the press and the group's center
        float cMax = 0.f;  // end of the group; the sample counts once we pass it alive
    };

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
            bool isShip = player->m_isShip;
            bool other = player->m_isBird || player->m_isBall || player->m_isDart || player->m_isSpider;
            m_modeName = isShip ? "nave" : (other ? "no soportado" : "cubo");

            if (botOn) {
                bool want = false;
                if (isShip) {
                    logMode("ship");
                    want = shipDecide(player, pr);
                } else if (other) {
                    logMode("unsupported");
                    want = false;
                } else {
                    logMode("cube");
                    float sign = player->m_isUpsideDown ? -1.f : 1.f;
                    trackApex((pr.getMinX() + pr.getMaxX()) * 0.5f, sign);
                    want = cubeDecide(pr, player->m_isOnGround);
                }
                setPressed(layer, want);
            } else {
                if (m_pressed) setPressed(layer, false);
            }

            if (rec) {
                commitPending(px);
            } else {
                m_pending.clear();
            }

            if (hud) updateHud(layer, botOn, rec);
        }

        // Called (from the GJBaseGameLayer hook) when the REAL player presses jump.
        void onPlayerPress(PlayLayer* layer) {
            if (m_injecting || !layer || !layer->m_player1) return;
            if (!getOpt(OptRecord)) return;
            PlayerObject* p = layer->m_player1;
            if (p->m_isShip || p->m_isBird || p->m_isBall || p->m_isDart || p->m_isSpider) return;
            if (!p->m_isOnGround) return;

            ensureObstacles(layer);
            CCRect pr = p->getObjectRect();
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
            if (debug()) log::info("AutoBot: recorded press t={:.3f}s", t);
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
        bool m_injecting = false;
        int m_holdFrames = 0;
        float m_handledMaxX = -1e9f;
        char const* m_lastMode = "";
        char const* m_modeName = "-";

        // self-calibration of the cube jump
        float m_apexTime = kCubeApexTime;
        bool m_jumpActive = false;
        bool m_sawRising = false;
        float m_jumpStartX = 0.f;
        int m_jumpFrames = 0;

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
            m_handledMaxX = -1e9f;
            m_recHandledMaxX = -1e9f;
            m_vy = 0.f;
            m_jumpActive = false;
            m_sawRising = false;
            m_pending.clear();  // the attempt ended: unconfirmed presses are discarded
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
            m_injecting = true;
            layer->handleButton(want, 1, true);
            m_injecting = false;
            if (debug()) log::info("AutoBot: {}", want ? "press" : "release");
        }

        // ---- recorder / learning -----------------------------------------
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

        // A recorded press only counts once the player got past its group alive.
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

        bool learnedActive() const {
            return getOpt(OptLearn) && m_samples.size() >= kMinSamplesToLearn;
        }

        float learnedMedian() const {
            std::vector<float> v = m_samples;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        }

        // Time (seconds) between the jump press and the center of the obstacle group.
        float jumpTime() const {
            if (learnedActive()) return std::clamp(learnedMedian(), 0.12f, 0.30f);
            return m_apexTime;
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
                "salto: {:.3f}s {}\n"
                "grabar: {}  muestras: {}  pend: {}\n"
                "{}",
                botOn ? "ON" : "OFF", m_modeName, m_speed, jumpTime(),
                learnedActive() ? "(aprendido)" : "(medido)", rec ? "ON" : "OFF",
                m_samples.size(), m_pending.size(), m_lastEvent);
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

        // Group of obstacles ahead, in the player's current row.
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
                    m_lastEvent = fmt::format("apice {:.3f}s", t);
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
                    m_lastEvent = "orbe";
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

            float cMin = 0.f, cMax = 0.f;
            if (!findCluster(pr, m_handledMaxX, cMin, cMax)) return false;

            // Jump so the apex of the arc lands over the middle of the group.
            float pcx = (pr.getMinX() + pr.getMaxX()) * 0.5f;
            float center = (cMin + cMax) * 0.5f;
            float apexDistance = m_speed * jumpTime();
            if (center - pcx <= apexDistance) {
                m_handledMaxX = cMax;
                m_holdFrames = kCubeHoldFrames;
                m_jumpActive = true;
                m_sawRising = false;
                m_jumpStartX = pcx;
                m_jumpFrames = 0;
                m_lastEvent = fmt::format("salto dist {:.0f}", center - pcx);
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

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

class $modify(AutoBotPlayLayer, PlayLayer) {
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        if (PlayLayer::get() == this) autobot::Bot::get().update(this, dt);
    }
};

// Sees every jump press (real ones and the bot's own; the bot marks its own).
class $modify(AutoBotBaseLayer, GJBaseGameLayer) {
    void handleButton(bool down, int button, bool isPlayer1) {
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
        if (down && button == 1 && isPlayer1) {
            auto* pl = PlayLayer::get();
            if (pl && static_cast<GJBaseGameLayer*>(pl) == this) {
                autobot::Bot::get().onPlayerPress(pl);
            }
        }
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
            {"Usar lo aprendido", autobot::OptLearn},
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
    log::info("AutoBot 0.3.1 loaded.");
}
